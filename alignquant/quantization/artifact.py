"""Frozen Fisher-guided W4/W8 weight tiles with uniform A8 activations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


P4 = 4
P8 = 8
REPLACEMENT_ARTIFACT_SCHEMA = "alignquant_w4w8_a8_v3"
REPLACEMENT_ARTIFACT_VERSION = 3
REPLACEMENT_TENSOR_NAMES = (
    "state_bits",
    "w4_payload",
    "w4_scales",
    "w8_payload",
    "w8_scales",
    "w4_row_offsets",
    "w8_row_offsets",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return _sha256_bytes(value.numpy().tobytes())


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _safe_shard_stem(layer_name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", layer_name).strip("_")
    if not stem:
        raise ValueError("Layer name cannot be empty.")
    return stem


def layout_b_offsets(*, device: torch.device | str | None = None) -> torch.Tensor:
    """Map each logical [K,N] element of a 64x64 tile to SM80 Layout-B."""
    k = torch.arange(64, dtype=torch.int64, device=device).view(64, 1)
    n = torch.arange(64, dtype=torch.int64, device=device).view(1, 64)
    vector_contiguous = k // 16
    vector_strided = n // 4
    tile_contiguous = vector_contiguous // 2
    tile_residual = (vector_contiguous % 2) + (n % 4) * 2
    partition = tile_residual // 4
    permuted = (tile_residual % 4) ^ (vector_strided % 4)
    physical_k = (tile_contiguous * 8 + partition * 4 + permuted) * 16 + k % 16
    return (physical_k + vector_strided * 256).contiguous()


def _layout_b_physical(logical: torch.Tensor) -> torch.Tensor:
    if tuple(logical.shape[-2:]) != (64, 64):
        raise ValueError(f"Expected [...,64,64], got {tuple(logical.shape)}.")
    offsets = layout_b_offsets(device=logical.device).reshape(-1)
    physical = torch.empty_like(logical).reshape(*logical.shape[:-2], 4096)
    physical[..., offsets] = logical.reshape(*logical.shape[:-2], 4096)
    return physical.reshape_as(logical).contiguous()


def unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    """Expand low-nibble-first signed INT4 values into INT8."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"Packed W4 payload must be uint8, got {packed.dtype}.")
    low = (packed & 0xF).to(torch.int16)
    high = ((packed >> 4) & 0xF).to(torch.int16)
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)
    output = torch.empty(
        (*packed.shape[:-1], int(packed.shape[-1]) * 2),
        dtype=torch.int8,
        device=packed.device,
    )
    output[..., 0::2] = low.to(torch.int8)
    output[..., 1::2] = high.to(torch.int8)
    return output.contiguous()


def pack_w4w8_state_bits(states: torch.Tensor) -> torch.Tensor:
    """Pack a [n_j,n_k] P4/P8 map LSB-first, with one meaning W4."""
    flat = torch.as_tensor(states, dtype=torch.uint8, device="cpu").reshape(-1)
    if not bool(torch.all((flat == P4) | (flat == P8))):
        raise ValueError("Weight states may contain only P4 and P8.")
    packed = torch.zeros(math.ceil(int(flat.numel()) / 8), dtype=torch.uint8)
    if flat.numel():
        positions = torch.arange(flat.numel(), dtype=torch.int64)
        packed.scatter_add_(
            0,
            positions // 8,
            ((flat == P4).to(torch.uint8) << (positions % 8)).to(torch.uint8),
        )
    return packed.contiguous()


def unpack_w4w8_state_bits(
    state_bits: torch.Tensor, *, n_j: int, n_k: int
) -> torch.Tensor:
    total = int(n_j) * int(n_k)
    packed = torch.as_tensor(state_bits, dtype=torch.uint8, device="cpu").reshape(-1)
    if int(packed.numel()) != math.ceil(total / 8):
        raise ValueError(f"Expected {math.ceil(total / 8)} state bytes.")
    positions = torch.arange(total, dtype=torch.int64)
    is_w4 = ((packed[positions // 8].to(torch.int64) >> (positions % 8)) & 1).bool()
    return torch.where(is_w4, P4, P8).to(torch.uint8).view(int(n_j), int(n_k))


def pack_w4w8_replacement_module(
    *,
    states: torch.Tensor,
    qweight4: torch.Tensor,
    weight_scale4: torch.Tensor,
    qweight8: torch.Tensor,
    weight_scale8: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Keep exactly one packed weight payload and one scale for each tile."""
    state_map = torch.as_tensor(states, dtype=torch.uint8, device="cpu").contiguous()
    if state_map.ndim != 2:
        raise ValueError(f"states must be [n_j,n_k], got {tuple(state_map.shape)}.")
    n_j, n_k = map(int, state_map.shape)
    expected = (n_j, n_k, 64, 64)
    q4 = qweight4.detach().to(device="cpu", dtype=torch.int8).contiguous()
    q8 = qweight8.detach().to(device="cpu", dtype=torch.int8).contiguous()
    s4 = weight_scale4.detach().to(device="cpu", dtype=torch.float32).contiguous()
    s8 = weight_scale8.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(q4.shape) != expected or tuple(q8.shape) != expected:
        raise ValueError(f"Candidate weights must be {expected}.")
    if tuple(s4.shape) != (n_j, n_k) or tuple(s8.shape) != (n_j, n_k):
        raise ValueError("Candidate scales must be [n_j,n_k].")
    if bool((q4 < -8).any()) or bool((q4 > 7).any()):
        raise ValueError("W4 values must be signed INT4 in [-8,7].")
    if not bool(torch.isfinite(s4).all() and torch.isfinite(s8).all()):
        raise ValueError("Weight scales must be finite.")
    if bool((s4 <= 0).any() or (s8 <= 0).any()):
        raise ValueError("Weight scales must be positive.")

    state_bits = pack_w4w8_state_bits(state_map)
    w4_payloads: list[torch.Tensor] = []
    w8_payloads: list[torch.Tensor] = []
    w4_scales: list[torch.Tensor] = []
    w8_scales: list[torch.Tensor] = []
    w4_offsets = [0]
    w8_offsets = [0]
    for j in range(n_j):
        mask4 = state_map[j] == P4
        mask8 = state_map[j] == P8
        physical4 = _layout_b_physical(q4[j, mask4].transpose(1, 2).contiguous())
        physical8 = _layout_b_physical(q8[j, mask8].transpose(1, 2).contiguous())
        if physical4.numel():
            low = physical4[..., 0::2].to(torch.int16) & 0xF
            high = (physical4[..., 1::2].to(torch.int16) & 0xF) << 4
            w4_payloads.append((low | high).to(torch.uint8).contiguous())
            w4_scales.append(s4[j, mask4])
        if physical8.numel():
            w8_payloads.append(physical8)
            w8_scales.append(s8[j, mask8])
        w4_offsets.append(w4_offsets[-1] + int(mask4.sum()))
        w8_offsets.append(w8_offsets[-1] + int(mask8.sum()))

    return {
        "state_bits": state_bits,
        "w4_payload": torch.cat(w4_payloads)
        if w4_payloads
        else torch.empty((0, 64, 32), dtype=torch.uint8),
        "w4_scales": torch.cat(w4_scales)
        if w4_scales
        else torch.empty(0, dtype=torch.float32),
        "w8_payload": torch.cat(w8_payloads)
        if w8_payloads
        else torch.empty((0, 64, 64), dtype=torch.int8),
        "w8_scales": torch.cat(w8_scales)
        if w8_scales
        else torch.empty(0, dtype=torch.float32),
        "w4_row_offsets": torch.tensor(w4_offsets, dtype=torch.int32),
        "w8_row_offsets": torch.tensor(w8_offsets, dtype=torch.int32),
    }


def validate_w4w8_replacement_module(
    tensors: Mapping[str, torch.Tensor], *, n_j: int, n_k: int
) -> torch.Tensor:
    """Validate one selected-payload shard and return its unpacked state map."""
    if set(tensors) != set(REPLACEMENT_TENSOR_NAMES):
        raise RuntimeError("Unexpected replacement tensors.")
    states = unpack_w4w8_state_bits(tensors["state_bits"], n_j=n_j, n_k=n_k)
    count4 = int((states == P4).sum())
    count8 = int((states == P8).sum())
    expected = {
        "w4_payload": ((count4, 64, 32), torch.uint8),
        "w4_scales": ((count4,), torch.float32),
        "w8_payload": ((count8, 64, 64), torch.int8),
        "w8_scales": ((count8,), torch.float32),
        "w4_row_offsets": ((int(n_j) + 1,), torch.int32),
        "w8_row_offsets": ((int(n_j) + 1,), torch.int32),
    }
    for name, (shape, dtype) in expected.items():
        tensor = tensors[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise RuntimeError(f"Invalid {name}: {tuple(tensor.shape)} {tensor.dtype}.")
    for precision, name in ((P4, "w4_row_offsets"), (P8, "w8_row_offsets")):
        offsets = tensors[name].detach().cpu().to(torch.int64)
        expected_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.int64), (states == precision).sum(1).cumsum(0))
        )
        if not torch.equal(offsets, expected_offsets):
            raise RuntimeError(f"Invalid {name}.")
    for name in ("w4_scales", "w8_scales"):
        scales = tensors[name]
        if not bool(torch.isfinite(scales).all()) or bool((scales <= 0).any()):
            raise RuntimeError(f"Invalid {name}.")
    return states


def _module_r5(module: Any) -> dict[str, Any]:
    plan = module.plan.normalized(n_k=int(module.n_k), n_j=int(module.n_j))
    if str(plan.variant) != "R5":
        raise ValueError(f"Replacement artifact requires R5, got {plan.variant!r}.")
    return {
        "variant": "R5",
        "t_seed": int(plan.t_seed),
        "v_seeds": [int(value) for value in plan.v_seeds],
        "u_seeds": [int(value) for value in plan.u_seeds],
    }


def _validate_r5_decision(value: Mapping[str, Any], *, n_j: int, n_k: int) -> dict[str, Any]:
    """Validate the runtime R5 metadata carried by an explicit candidate set."""

    expected = {"variant", "t_seed", "v_seeds", "u_seeds"}
    if set(value) != expected or str(value.get("variant")) != "R5":
        raise ValueError("Candidate module requires exactly the frozen R5 metadata.")
    if int(value["t_seed"]) != -1:
        raise ValueError("The frozen native runtime supports identity T only.")
    v_seeds = tuple(int(seed) for seed in value["v_seeds"])
    u_seeds = tuple(int(seed) for seed in value["u_seeds"])
    if len(v_seeds) != int(n_k) or len(u_seeds) != int(n_j):
        raise ValueError("R5 seed coverage does not match candidate tile geometry.")
    if any(seed < -1 for seed in (*v_seeds, *u_seeds)):
        raise ValueError("R5 seeds must be at least -1.")
    return {
        "variant": "R5",
        "t_seed": -1,
        "v_seeds": list(v_seeds),
        "u_seeds": list(u_seeds),
    }


def build_artifact(
    root: str | Path,
    *,
    modules: Sequence[Any],
    expected_total_tiles: int | None = None,
    model_identity: Mapping[str, str] | None = None,
) -> Path:
    """Build deterministic R5 shards directly from installed module state maps."""
    target = Path(root).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    rows: list[dict[str, Any]] = []
    seen_layers: set[str] = set()
    policy_digest = hashlib.sha256()
    totals = {"w4": 0, "w8": 0, "tiles": 0, "tensor_bytes": 0, "file_bytes": 0}
    try:
        for index, module in enumerate(modules):
            layer_name = str(module.layer_name)
            if layer_name in seen_layers:
                raise ValueError(f"Duplicate layer name: {layer_name}.")
            seen_layers.add(layer_name)
            if getattr(module, "p4_codebook", None) is not None:
                raise ValueError("Replacement payload requires signed INT4 values.")
            states = module.weight_state_map.detach().to(
                device="cpu", dtype=torch.uint8
            )
            if states.ndim != 2 or tuple(states.shape) != (int(module.n_j), int(module.n_k)):
                raise ValueError(f"Tile-state shape differs for {layer_name}.")
            encoded_name = layer_name.encode("utf-8")
            policy_digest.update(len(encoded_name).to_bytes(4, "little"))
            policy_digest.update(encoded_name)
            policy_digest.update(states.contiguous().numpy().tobytes())
            tensors = pack_w4w8_replacement_module(
                states=states,
                qweight4=module.qweight4,
                weight_scale4=module.weight_scale4,
                qweight8=module.qweight8,
                weight_scale8=module.weight_scale8,
            )
            validate_w4w8_replacement_module(
                tensors, n_j=int(module.n_j), n_k=int(module.n_k)
            )
            shard_name = f"{index:03d}-{_safe_shard_stem(layer_name)}.safetensors"
            shard_path = staging / shard_name
            save_file(tensors, shard_path)
            count4 = int(tensors["w4_scales"].numel())
            count8 = int(tensors["w8_scales"].numel())
            tensor_bytes = sum(_tensor_bytes(tensor) for tensor in tensors.values())
            row = {
                "layer_name": layer_name,
                "shape": [int(module.out_features), int(module.in_features)],
                "n_j": int(module.n_j),
                "n_k": int(module.n_k),
                "tile_counts": {"w4": count4, "w8": count8, "total": count4 + count8},
                "tensor_bytes": tensor_bytes,
                "file": shard_name,
                "file_bytes": int(shard_path.stat().st_size),
                "file_sha256": _file_sha256(shard_path),
                "tensor_sha256": {
                    name: _tensor_sha256(tensors[name])
                    for name in REPLACEMENT_TENSOR_NAMES
                },
                "r5": _module_r5(module),
            }
            rows.append(row)
            totals["w4"] += count4
            totals["w8"] += count8
            totals["tiles"] += count4 + count8
            totals["tensor_bytes"] += tensor_bytes
            totals["file_bytes"] += row["file_bytes"]
        if not rows:
            raise ValueError("At least one projection module is required.")
        if expected_total_tiles is not None and totals["tiles"] != int(
            expected_total_tiles
        ):
            raise ValueError(
                f"Expected {expected_total_tiles} tiles, got {totals['tiles']}."
            )
        payload_bytes = totals["w4"] * 2048 + totals["w8"] * 4096
        uniform_w8_bytes = totals["tiles"] * (4096 + 4) + sum(
            math.ceil(row["tile_counts"]["total"] / 8) + 2 * (row["n_j"] + 1) * 4
            for row in rows
        )
        manifest = {
            "schema": REPLACEMENT_ARTIFACT_SCHEMA,
            "version": REPLACEMENT_ARTIFACT_VERSION,
            "model": dict(model_identity or {}),
            "policy_sha256": policy_digest.hexdigest(),
            "format": {
                "state_bits": "LSB-first flattened (j,k), 1=W4, 0=W8",
                "w4_payload": "SM80 Layout-B [tile,64,32], low nibble first",
                "w8_payload": "SM80 Layout-B [tile,64,64]",
                "scales": "float32 scalar per 64x64 tile",
                "row_offsets": "int32 compact-arena prefix, length n_j+1",
            },
            "totals": {
                **totals,
                "payload_bytes": payload_bytes,
                "payload_bits_per_weight": 8.0
                * payload_bytes
                / (totals["tiles"] * 4096),
                "tensor_storage_ratio_vs_uniform_w8": totals["tensor_bytes"]
                / uniform_w8_bytes,
            },
            "modules": rows,
        }
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        (staging / "manifest.json").write_text(manifest_text, encoding="utf-8")
        (staging / "manifest.sha256").write_text(
            f"{_sha256_bytes(manifest_text.encode())}  manifest.json\n",
            encoding="utf-8",
        )
        staging.replace(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def load_artifact(
    root: str | Path, *, device: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load and fully validate one replacement artifact."""
    target = Path(root).expanduser().resolve()
    manifest_text = (target / "manifest.json").read_text(encoding="utf-8")
    expected_manifest_hash = (
        (target / "manifest.sha256").read_text(encoding="utf-8").split()[0]
    )
    if _sha256_bytes(manifest_text.encode()) != expected_manifest_hash:
        raise RuntimeError("Replacement manifest digest mismatch.")
    manifest = json.loads(manifest_text)
    if (
        manifest.get("schema") != REPLACEMENT_ARTIFACT_SCHEMA
        or manifest.get("version") != REPLACEMENT_ARTIFACT_VERSION
    ):
        raise ValueError("Unsupported W4/W8 replacement artifact.")
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    totals = {"w4": 0, "w8": 0, "tiles": 0, "tensor_bytes": 0, "file_bytes": 0}
    for row in manifest["modules"]:
        shard_path = target / row["file"]
        if _file_sha256(shard_path) != row["file_sha256"]:
            raise RuntimeError(f"Replacement shard digest mismatch: {row['file']}.")
        tensors = load_file(shard_path, device=str(device))
        validate_w4w8_replacement_module(
            tensors, n_j=int(row["n_j"]), n_k=int(row["n_k"])
        )
        for name in REPLACEMENT_TENSOR_NAMES:
            if _tensor_sha256(tensors[name]) != row["tensor_sha256"][name]:
                raise RuntimeError(
                    f"Tensor digest mismatch for {row['layer_name']}/{name}."
                )
        count4 = int(tensors["w4_scales"].numel())
        count8 = int(tensors["w8_scales"].numel())
        if row["tile_counts"] != {"w4": count4, "w8": count8, "total": count4 + count8}:
            raise RuntimeError(f"Tile count mismatch for {row['layer_name']}.")
        loaded[str(row["layer_name"])] = tensors
        totals["w4"] += count4
        totals["w8"] += count8
        totals["tiles"] += count4 + count8
        totals["tensor_bytes"] += sum(
            _tensor_bytes(tensor) for tensor in tensors.values()
        )
        totals["file_bytes"] += int(shard_path.stat().st_size)
    for name, value in totals.items():
        if int(manifest["totals"][name]) != value:
            raise RuntimeError(f"Artifact total mismatch: {name}.")
    return {"root": target, "manifest": manifest, "modules": loaded}
