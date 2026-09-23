"""Install and execute the native W4/W8A8 replacement artifact."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from alignquant.kernels.cuda.extension import (
    alignquant_activation_quantize_cuda,
    alignquant_linear_cuda,
)
from alignquant.models.decoder import iter_target_linears
from alignquant.quantization.exact import structured_transform_metadata
from alignquant.quantization.artifact import (
    REPLACEMENT_TENSOR_NAMES,
    load_artifact,
    pack_w4w8_state_bits,
    unpack_w4w8_state_bits,
)


def projection_groups(module_names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for name in sorted(set(str(value) for value in module_names)):
        if name.endswith(
            (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")
        ):
            group = name.rsplit(".", 1)[0] + ".qkv"
        elif name.endswith((".mlp.gate_proj", ".mlp.up_proj")):
            group = name.rsplit(".", 1)[0] + ".gate_up"
        else:
            group = name
        groups.setdefault(group, []).append(name)
    output = {key: tuple(values) for key, values in sorted(groups.items())}
    for group, members in output.items():
        expected = (
            3 if group.endswith(".qkv") else 2 if group.endswith(".gate_up") else 1
        )
        if len(members) != expected:
            raise ValueError(f"Incomplete projection group {group}: {members}.")
    return output


def _device(value: torch.device | str) -> torch.device:
    result = torch.device(value)
    if result.type == "cuda" and result.index is None:
        result = torch.device("cuda", torch.cuda.current_device())
    if result.type != "cuda":
        raise ValueError(
            f"Native W4/W8A8 runtime requires a CUDA device, got {result}."
        )
    return result


def _as_cuda_buffer(
    value: torch.Tensor, device: torch.device, name: str
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Artifact tensor {name} is not a tensor.")
    return value.detach().to(device=device).contiguous()


def _r5_seed_list(raw: Any, *, count: int, name: str) -> tuple[int, ...]:
    if not isinstance(raw, (list, tuple)) or len(raw) != int(count):
        raise ValueError(f"R5 {name} must contain exactly {count} seeds.")
    result = tuple(int(item) for item in raw)
    if any(item < -1 for item in result):
        raise ValueError(f"R5 {name} contains an invalid seed.")
    return result


def _validate_r5(
    row: Mapping[str, Any], *, n_j: int, n_k: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    decision = row.get("r5")
    if not isinstance(decision, Mapping) or str(decision.get("variant", "")) != "R5":
        raise ValueError(
            f"{row.get('layer_name', '<unknown>')} is not a frozen R5 module."
        )
    if int(decision.get("t_seed", -1)) != -1:
        raise ValueError(
            "Native W4/W8A8 R5 runtime rejects a non-identity T transform."
        )
    allowed = {"variant", "t_seed", "v_seeds", "u_seeds"}
    if set(decision) != allowed:
        raise ValueError(
            f"R5 artifact for {row.get('layer_name', '<unknown>')} contains unsupported "
            "diagonal/factor payload or unknown fields."
        )
    return (
        _r5_seed_list(decision["v_seeds"], count=n_k, name="V"),
        _r5_seed_list(decision["u_seeds"], count=n_j, name="U"),
    )


def _metadata_stack(seeds: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.stack(
        [structured_transform_metadata(int(seed), device=device) for seed in seeds],
        dim=0,
    ).contiguous()


class _PreparedGroupOutput:
    __slots__ = (
        "key",
        "original_shape",
        "rows",
        "combined",
        "consumed_members",
    )

    def __init__(
        self,
        key: tuple[Any, ...],
        original_shape: tuple[int, ...],
        rows: int,
        combined: torch.Tensor,
    ) -> None:
        self.key = key
        self.original_shape = original_shape
        self.rows = rows
        self.combined = combined
        self.consumed_members: set[str] = set()


class _ProjectionGroupRuntime:
    """Own and execute one fused QKV, Gate/Up, or singleton projection group."""

    def __init__(
        self,
        group_id: str,
        members: Sequence[str],
        rows: Mapping[str, Mapping[str, Any]],
        tensors: Mapping[str, Mapping[str, torch.Tensor]],
        device: torch.device,
        *,
        reuse_activation: bool = True,
        cast_combined_output: bool = False,
        bias_free_llama: bool = False,
    ) -> None:
        self.group_id = str(group_id)
        self.members = tuple(str(item) for item in members)
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError(
                f"Projection group {self.group_id} has empty or duplicate members."
            )
        if set(rows) != set(self.members) or set(tensors) != set(self.members):
            raise ValueError(
                f"Projection group {self.group_id} has incomplete member artifacts."
            )
        self.device = torch.device(device)
        self.reuse_activation = bool(reuse_activation)
        self.cast_combined_output = bool(cast_combined_output)
        self.bias_free_llama = bool(bias_free_llama)
        self.member_slices: dict[str, tuple[int, int]] = {}

        state_rows: list[torch.Tensor] = []
        w4_payloads: list[torch.Tensor] = []
        w4_scales: list[torch.Tensor] = []
        w8_payloads: list[torch.Tensor] = []
        w8_scales: list[torch.Tensor] = []
        w4_offsets = [0]
        w8_offsets = [0]
        u_seeds: list[int] = []
        v_seeds: tuple[int, ...] | None = None
        output_offset = 0
        n_k: int | None = None

        for member in self.members:
            row = rows[member]
            member_tensors = tensors[member]
            if set(member_tensors) != set(REPLACEMENT_TENSOR_NAMES):
                raise ValueError(f"Unexpected replacement tensors for {member}.")
            member_n_j = int(row["n_j"])
            member_n_k = int(row["n_k"])
            if member_n_j <= 0 or member_n_k <= 0:
                raise ValueError(f"Invalid tile geometry for {member}.")
            member_v_seeds, member_u_seeds = _validate_r5(
                row, n_j=member_n_j, n_k=member_n_k
            )
            if n_k is None:
                n_k = member_n_k
                v_seeds = member_v_seeds
            elif member_n_k != n_k or member_v_seeds != v_seeds:
                raise ValueError(
                    f"Projection group {self.group_id} does not share K geometry and V seeds."
                )

            state_rows.append(
                unpack_w4w8_state_bits(
                    member_tensors["state_bits"], n_j=member_n_j, n_k=member_n_k
                )
            )
            for prefix, payloads, scales, combined_offsets in (
                ("w4", w4_payloads, w4_scales, w4_offsets),
                ("w8", w8_payloads, w8_scales, w8_offsets),
            ):
                payload = _as_cuda_buffer(
                    member_tensors[f"{prefix}_payload"],
                    self.device,
                    f"{member}.{prefix}_payload",
                )
                scale = _as_cuda_buffer(
                    member_tensors[f"{prefix}_scales"],
                    self.device,
                    f"{member}.{prefix}_scales",
                )
                offsets = (
                    member_tensors[f"{prefix}_row_offsets"]
                    .detach()
                    .to(device="cpu", dtype=torch.int64)
                    .contiguous()
                    .view(-1)
                )
                if (
                    int(offsets.numel()) != member_n_j + 1
                    or int(offsets[0]) != 0
                    or bool((offsets[1:] < offsets[:-1]).any())
                    or int(offsets[-1]) != int(payload.shape[0])
                    or int(scale.numel()) != int(payload.shape[0])
                ):
                    raise ValueError(
                        f"Malformed {prefix.upper()} arena or offsets for {member}."
                    )
                base = combined_offsets[-1]
                combined_offsets.extend(base + int(value) for value in offsets[1:])
                payloads.append(payload)
                scales.append(scale)

            u_seeds.extend(member_u_seeds)
            member_width = member_n_j * 64
            self.member_slices[member] = (output_offset, output_offset + member_width)
            output_offset += member_width

        assert n_k is not None and v_seeds is not None
        self.n_k = n_k
        self.v_seeds = v_seeds
        self.out_features = output_offset
        self.state_bits = _as_cuda_buffer(
            pack_w4w8_state_bits(torch.cat(state_rows, dim=0)),
            self.device,
            "state_bits",
        )
        self.w4_payload = torch.cat(w4_payloads, dim=0).contiguous()
        self.w4_scales = torch.cat(w4_scales, dim=0).contiguous()
        self.w8_payload = torch.cat(w8_payloads, dim=0).contiguous()
        self.w8_scales = torch.cat(w8_scales, dim=0).contiguous()
        self.w4_row_offsets = torch.tensor(
            w4_offsets, dtype=torch.int32, device=self.device
        ).contiguous()
        self.w8_row_offsets = torch.tensor(
            w8_offsets, dtype=torch.int32, device=self.device
        ).contiguous()
        self.v_metadata = _metadata_stack(self.v_seeds, self.device)
        self.u_metadata = _metadata_stack(u_seeds, self.device)
        self._prepared: _PreparedGroupOutput | None = None

    @staticmethod
    def _activation_key(activation: torch.Tensor) -> tuple[Any, ...]:
        return (
            activation.device.type,
            activation.device.index,
            activation.data_ptr(),
            int(activation.storage_offset()),
            tuple(int(value) for value in activation.shape),
            tuple(int(value) for value in activation.stride()),
            str(activation.dtype),
        )

    def project_member(
        self, activation: torch.Tensor, *, member: str
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        member_name = str(member)
        if member_name not in self.members:
            raise ValueError(
                f"{member_name} is not a member of projection group {self.group_id}."
            )
        key = self._activation_key(activation)
        if not self.reuse_activation:
            self._prepared = None
        if self._prepared is not None:
            if self._prepared.key != key:
                missing = sorted(set(self.members) - self._prepared.consumed_members)
                raise RuntimeError(
                    f"Projection group {self.group_id} received a new activation before "
                    f"members {missing} consumed the prior combined output."
                )
            if member_name in self._prepared.consumed_members:
                raise RuntimeError(
                    f"Projection group {self.group_id} member {member_name} was called twice "
                    "for the same activation."
                )
            prepared = self._prepared
        else:
            if activation.device != self.device:
                raise ValueError(
                    f"Projection group {self.group_id} is on {self.device}, got {activation.device}."
                )
            if activation.dim() < 1:
                raise ValueError(
                    f"Native W4/W8A8 linear expects [...,K], got {tuple(activation.shape)}."
                )
            original_shape = tuple(int(value) for value in activation.shape)
            width = int(activation.shape[-1])
            if width != self.n_k * 64:
                raise ValueError(
                    f"Activation K mismatch for {self.group_id}: {width} vs {self.n_k * 64}."
                )
            rows = int(activation.numel()) // width
            if rows <= 0:
                raise ValueError("Activation must contain at least one row.")
            flat = activation.reshape(rows, width).contiguous()
            tile_rows = 1 if rows == 1 else 16
            padded_rows = int(math.ceil(rows / tile_rows) * tile_rows)
            if padded_rows == rows:
                padded = flat
            else:
                padded = torch.zeros(
                    (padded_rows, int(flat.shape[1])),
                    dtype=flat.dtype,
                    device=flat.device,
                )
                padded[:rows] = flat
            qx, scales = alignquant_activation_quantize_cuda(
                padded, self.v_metadata, tile_major=padded_rows >= 128
            )
            combined = alignquant_linear_cuda(
                qx,
                scales,
                self.state_bits,
                self.w4_payload,
                self.w4_scales,
                self.w8_payload,
                self.w8_scales,
                self.w4_row_offsets,
                self.w8_row_offsets,
                self.u_metadata,
                decode_splits=0,
                output_dtype=(
                    torch.bfloat16
                    if self.bias_free_llama and activation.dtype == torch.bfloat16
                    and padded_rows >= 128
                    else torch.float32
                ),
            )
            if tuple(combined.shape) != (padded_rows, self.out_features):
                raise RuntimeError(
                    f"Projection group {self.group_id} returned {tuple(combined.shape)}, "
                    f"expected {(padded_rows, self.out_features)}."
                )
            # Bias-free Llama small-batch projections can share one final cast.
            # Keep all FP32 accumulation/inverse-U work before this rounding.
            # Keep accumulation in FP32 for a single activation row.
            if self.cast_combined_output and padded_rows == 16:
                combined = combined.to(dtype=activation.dtype)
            prepared = _PreparedGroupOutput(key, original_shape, rows, combined)
            self._prepared = prepared

        start, stop = self.member_slices[member_name]
        result = prepared.combined[: prepared.rows, start:stop]
        prepared.consumed_members.add(member_name)
        if not self.reuse_activation or prepared.consumed_members == set(self.members):
            self._prepared = None
        return result, prepared.original_shape

    def reset_pending(self) -> None:
        """Discard an incomplete shared projection after an aborted forward."""

        self._prepared = None


class AlignQuantLinear(nn.Module):
    """A frozen artifact-backed linear using native SM80 W4/W8A8 kernels."""

    def __init__(
        self,
        linear: nn.Linear,
        *,
        layer_name: str,
        runtime: _ProjectionGroupRuntime,
    ) -> None:
        super().__init__()
        self.layer_name = str(layer_name)
        object.__setattr__(self, "runtime", runtime)
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        expected_width = (
            runtime.member_slices[self.layer_name][1]
            - runtime.member_slices[self.layer_name][0]
        )
        if self.in_features != runtime.n_k * 64 or self.out_features != expected_width:
            raise ValueError(
                f"Linear geometry disagrees with projection group for {self.layer_name}."
            )
        if linear.bias is None:
            self.register_buffer("bias", None, persistent=False)
        else:
            self.register_buffer(
                "bias",
                linear.bias.detach().to(device=runtime.device).contiguous(),
                persistent=False,
            )

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        result, original_shape = self.runtime.project_member(
            activation, member=self.layer_name
        )
        if self.bias is not None:
            result = result + self.bias.to(dtype=result.dtype)
        result = result.reshape(*original_shape[:-1], self.out_features)
        return result.to(dtype=activation.dtype)


def install_alignquant(
    model: nn.Module,
    artifact: Mapping[str, Any] | str | Path,
    device: torch.device | str | None,
    *,
    reuse_activation: bool = True,
) -> tuple[list[AlignQuantLinear], dict[str, Any]]:
    """Install exactly the artifact-covered target projections and return metadata."""

    target_device = None if device is None else _device(device)
    loaded = (
        artifact
        if isinstance(artifact, Mapping)
        else load_artifact(artifact, device="cpu")
    )
    if set(loaded) != {"root", "manifest", "modules"}:
        raise ValueError(
            "Replacement artifact must contain root, manifest, and modules."
        )
    manifest = loaded["manifest"]
    module_tensors = loaded["modules"]
    if not isinstance(manifest, Mapping) or not isinstance(module_tensors, Mapping):
        raise TypeError("Replacement artifact manifest and modules must be mappings.")
    rows = list(manifest.get("modules", ()))
    if not rows:
        raise ValueError("Replacement artifact must contain at least one module.")
    row_map: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        name = str(row.get("layer_name", ""))
        if not name or name in row_map:
            raise ValueError(f"Duplicate or empty artifact module name: {name!r}.")
        row_map[name] = row
    targets = list(iter_target_linears(model))
    if not targets:
        raise ValueError("Model has no target linear projections.")
    target_names = {name for _parent, _attr, name, _linear in targets}
    if len(target_names) != len(targets) or target_names != set(row_map):
        raise ValueError(
            "Artifact/model target coverage mismatch: "
            f"model={len(targets)} artifact={len(row_map)} "
            f"missing={sorted(target_names - set(row_map))[:3]} "
            f"extra={sorted(set(row_map) - target_names)[:3]}"
        )
    if set(module_tensors) != set(row_map):
        raise ValueError(
            "Loaded replacement tensor coverage differs from manifest coverage."
        )

    # Keep only replacement coordinates and shape/device metadata. Holding the
    # iterator's Linear objects would retain every original BF16 weight while
    # native groups are installed.
    target_records: dict[str, tuple[nn.Module, str, tuple[int, int], torch.device]] = {}
    for parent, attr_name, name, linear in targets:
        expected_shape = (int(linear.out_features), int(linear.in_features))
        row = row_map[name]
        if tuple(int(value) for value in row.get("shape", ())) != expected_shape:
            raise ValueError(
                f"Artifact shape mismatch for {name}: {row.get('shape')} vs {expected_shape}."
            )
        if (
            int(row["n_j"]) != expected_shape[0] // 64
            or int(row["n_k"]) != expected_shape[1] // 64
        ):
            raise ValueError(f"Artifact tile geometry mismatch for {name}.")
        target_records[name] = (
            parent, attr_name, expected_shape, _device(linear.weight.device)
        )
    del targets, linear

    groups = projection_groups(tuple(sorted(target_records)))
    runtimes: dict[str, _ProjectionGroupRuntime] = {}
    installed: list[AlignQuantLinear] = []
    for group_id, members in groups.items():
        member_devices = {target_records[member][3] for member in members}
        if target_device is None:
            if len(member_devices) != 1:
                raise ValueError(
                    f"Projection group {group_id} spans CUDA devices: {member_devices}."
                )
            runtime_device = next(iter(member_devices))
        else:
            runtime_device = target_device
        runtime = _ProjectionGroupRuntime(
            group_id,
            members,
            {member: row_map[member] for member in members},
            {member: module_tensors[member] for member in members},
            runtime_device,
            reuse_activation=reuse_activation,
            cast_combined_output=(
                getattr(model.config, "model_type", None) == "llama"
                and len(members) > 1
                and all(
                    getattr(target_records[name][0], target_records[name][1]).bias is None
                    for name in members
                )
            ),
            bias_free_llama=(
                getattr(model.config, "model_type", None) == "llama"
                and all(
                    getattr(target_records[name][0], target_records[name][1]).bias is None
                    for name in members
                )
            ),
        )
        runtimes[group_id] = runtime
        for name in members:
            parent, attr_name, _expected_shape, _original_device = target_records.pop(name)
            linear = getattr(parent, attr_name)
            module = AlignQuantLinear(linear, layer_name=name, runtime=runtime)
            setattr(parent, attr_name, module)
            installed.append(module)
        del linear
    model._alignquant_projection_groups = tuple(runtimes.values())
    metadata = {
        "module_count": len(installed),
        "projection_group_count": len(runtimes),
        "groups": {group: list(members) for group, members in groups.items()},
        "reuse_activation": bool(reuse_activation),
    }
    return installed, metadata


def reset_alignquant_state(model: nn.Module) -> None:
    """Reset transient shared-projection state between independent forwards."""

    for runtime in getattr(model, "_alignquant_projection_groups", ()):
        runtime.reset_pending()


__all__ = [
    "AlignQuantLinear",
    "install_alignquant",
    "reset_alignquant_state",
]
