from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from alignquant.models.decoder import iter_target_linears
from alignquant.quantization.exact import (
    a8_endpoint_quantize_tiles,
    inverse_output_tiles,
    mse_quantize_tiles,
    reconstruct_tiles,
    split_weight_tiles,
    structured_transform_matrix,
    transform_activation_tiles,
    transform_weight_tiles,
)

TILE = 64
SEEDS = (-1, 0, 1, 2, 3, 4, 5, 6)


def _r5_step_plan(
    members: Sequence[str], *, n_k: int, n_j: Mapping[str, int]
) -> tuple[tuple[int, str, str | None, int], ...]:
    result: list[tuple[int, str, str | None, int]] = []
    for sweep in range(2):
        result.extend(((sweep, "v", None, index) for index in range(int(n_k))))
        for member in members:
            result.extend(
                (
                    (sweep, "u", str(member), index)
                    for index in range(int(n_j[str(member)]))
                )
            )
    return tuple(result)


def _projection_groups(module_names: Sequence[str]) -> dict[str, tuple[str, ...]]:
    groups: defaultdict[str, list[str]] = defaultdict(list)
    for name in sorted(set((str(value) for value in module_names))):
        if name.endswith(
            (".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")
        ):
            group = name.rsplit(".", 1)[0] + ".qkv"
        elif name.endswith((".mlp.gate_proj", ".mlp.up_proj")):
            group = name.rsplit(".", 1)[0] + ".gate_up"
        else:
            group = name
        groups[group].append(name)
    result = {name: tuple(values) for name, values in groups.items()}
    for group, members in result.items():
        expected = (
            3 if group.endswith(".qkv") else 2 if group.endswith(".gate_up") else 1
        )
        if len(members) != expected:
            raise ValueError(f"Incomplete projection group {group}: {members}.")
    return result


def _prompt_target(
    record: Mapping[str, Any], *, max_length: int
) -> tuple[list[int], list[int]]:
    prompt = record.get("prompt_ids")
    target = record.get("target_ids")
    if not isinstance(prompt, list) or not isinstance(target, list):
        raise ValueError(
            "Fisher calibration records require explicit prompt_ids and target_ids."
        )
    if (
        not prompt
        or not target
        or (not all((isinstance(value, int) for value in (*prompt, *target))))
    ):
        raise ValueError(
            "Fisher prompt_ids and target_ids must be nonempty integer arrays."
        )
    if len(prompt) + len(target) > int(max_length):
        raise ValueError(
            "Frozen Fisher prompt/target window exceeds max_length; refusing truncation."
        )
    return (list(prompt), list(target))


def deterministic_decode_proxy_indices(
    continuation_tokens: int, requested_steps: int = 4
) -> tuple[int, ...]:
    tokens, count = (int(continuation_tokens), int(requested_steps))
    if tokens < 2 or count <= 0:
        raise ValueError("Decode sensitivity requires two targets and positive steps.")
    available = tuple(range(1, tokens))
    if count >= len(available):
        return available
    if count == 1:
        return (available[(len(available) - 1) // 2],)
    last, denominator = (len(available) - 1, count - 1)
    selected = tuple(
        (
            available[(2 * rank * last + denominator) // (2 * denominator)]
            for rank in range(count)
        )
    )
    if len(set(selected)) != count:
        raise RuntimeError("Deterministic decode proxy sampling produced duplicates.")
    return selected


def _detach_past(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple((_detach_past(item) for item in value))
    if isinstance(value, list):
        return [_detach_past(item) for item in value]
    for layer in value.layers:
        if layer.keys is not None:
            layer.keys = layer.keys.detach()
        if layer.values is not None:
            layer.values = layer.values.detach()
    return value


def _input_rows(
    model: torch.nn.Module,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    max_length: int,
    module_names: Sequence[str] | None = None,
    max_rows: int | None = None,
) -> dict[str, torch.Tensor]:
    captured: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    selected = (
        None if module_names is None else set((str(name) for name in module_names))
    )
    required = (
        selected
        if selected is not None
        else {name for _parent, _attribute, name, _linear in iter_target_linears(model)}
    )
    hooks = []
    for _parent, _attribute, name, linear in iter_target_linears(model):
        if selected is not None and name not in selected:
            continue

        def capture(
            _module: torch.nn.Module, args: tuple[Any, ...], *, key: str = name
        ) -> None:
            value = (
                args[0].detach().reshape(-1, args[0].shape[-1]).to("cpu", torch.float64)
            )
            if max_rows is not None:
                remaining = int(max_rows) - sum(
                    (int(chunk.shape[0]) for chunk in captured[key])
                )
                value = value[: max(remaining, 0)]
            if not value.numel():
                return
            captured[key].append(value)

        hooks.append(linear.register_forward_pre_hook(capture))
    try:
        with torch.no_grad():
            for record in records:
                ids = torch.tensor(
                    record["input_ids"][:max_length], device=device, dtype=torch.long
                )[None, :]
                if int(ids.shape[1]) < 2:
                    continue
                model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                )
                if (
                    max_rows is not None
                    and captured
                    and all(
                        (
                            sum((int(chunk.shape[0]) for chunk in captured[name]))
                            >= int(max_rows)
                            for name in required
                        )
                    )
                ):
                    break
    finally:
        for hook in hooks:
            hook.remove()
    result = {name: torch.cat(rows, dim=0) for name, rows in captured.items() if rows}
    if set(result) != required:
        raise RuntimeError(
            f"Activation capture coverage mismatch: missing={sorted(required - set(result))[:3]}."
        )
    return result


def _endpoint_a8_rows(transformed: torch.Tensor) -> torch.Tensor:
    if transformed.ndim != 3 or int(transformed.shape[-1]) != TILE:
        raise ValueError("A8 rows must be [K,M,64].")
    n_k, rows = map(int, transformed.shape[:2])
    padded_rows = (rows + 15) // 16 * 16
    if padded_rows != rows:
        transformed = F.pad(transformed, (0, 0, 0, padded_rows - rows))
    grouped = transformed.reshape(n_k, -1, 16, TILE)
    quantized, scales = a8_endpoint_quantize_tiles(grouped)
    return reconstruct_tiles(quantized, scales, dtype=torch.float64).reshape(
        n_k, padded_rows, TILE
    )[:, :rows]


def _r5_a4_grid(rows: torch.Tensor) -> torch.Tensor:
    if rows.ndim != 2 or int(rows.shape[1]) % TILE:
        raise ValueError("R5 A4 calibration rows must be [rows,K64].")
    usable = int(rows.shape[0]) // 16 * 16
    if usable == 0:
        raise ValueError("R5 A4 calibration requires at least sixteen activation rows.")
    return (
        rows[:usable]
        .reshape(-1, 16, int(rows.shape[1]) // TILE, TILE)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def _a4_reconstruct(tiles: torch.Tensor) -> torch.Tensor:
    quantized, scales = mse_quantize_tiles(tiles, bits=4)
    return reconstruct_tiles(quantized, scales, dtype=torch.float64)


class _R5A4Cache:
    def __init__(
        self,
        activations: Mapping[str, torch.Tensor],
        weights: Mapping[str, torch.Tensor],
        members: Sequence[str],
        *,
        v_seed: int = -1,
        u_seed: Mapping[str, int] | None = None,
    ) -> None:
        self.members = tuple(members)
        representative = activations[self.members[0]]
        self.a = _r5_a4_grid(representative)
        self.w = {name: split_weight_tiles(weights[name]) for name in self.members}
        self.v = [int(v_seed)] * int(self.w[self.members[0]].shape[1])
        chosen_u = (
            {}
            if u_seed is None
            else {str(name): int(seed) for name, seed in u_seed.items()}
        )
        self.u = {
            name: [chosen_u.get(name, -1)] * int(self.w[name].shape[0])
            for name in self.members
        }
        self.matrices: dict[int, torch.Tensor] = {}
        self.aq: torch.Tensor
        self.wu: dict[str, torch.Tensor] = {}
        self.wq: dict[str, torch.Tensor] = {}
        self.reference: dict[str, torch.Tensor] = {}
        self.error: dict[str, torch.Tensor] = {}
        self.row_sse: dict[str, torch.Tensor] = {}
        self._initialize()

    def matrix(self, seed: int) -> torch.Tensor:
        if seed not in self.matrices:
            self.matrices[seed] = structured_transform_matrix(
                seed, device=self.w[self.members[0]].device
            )
        return self.matrices[seed]

    def _initialize(self) -> None:
        v = torch.stack([self.matrix(seed) for seed in self.v])
        self.aq = _a4_reconstruct(torch.matmul(self.a, v.unsqueeze(0)))
        for name in self.members:
            u = torch.stack([self.matrix(seed) for seed in self.u[name]])
            self.wu[name] = torch.matmul(u.transpose(-1, -2).unsqueeze(1), self.w[name])
            self.wq[name] = _a4_reconstruct(torch.matmul(self.wu[name], v.unsqueeze(0)))
            reference = torch.einsum("skmi,jkoi->sjmo", self.a, self.w[name])
            transformed = torch.einsum("skmi,jkoi->sjmo", self.aq, self.wq[name])
            recovered = torch.matmul(transformed, u.transpose(-1, -2).unsqueeze(0))
            error = recovered - reference
            self.reference[name], self.error[name] = (reference, error)
            self.row_sse[name] = error.square().sum(dim=(0, 2, 3))

    @property
    def total_error(self) -> float:
        return sum((float(value.sum().item()) for value in self.row_sse.values()))

    def score_v(
        self, k: int, seed: int
    ) -> tuple[
        float,
        tuple[torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    ]:
        matrix = self.matrix(seed)
        new_a = _a4_reconstruct(torch.matmul(self.a[:, k], matrix))
        updates: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        total = 0.0
        for name in self.members:
            w = _a4_reconstruct(torch.matmul(self.wu[name][:, k], matrix))
            old = torch.einsum("smi,joi->sjmo", self.aq[:, k], self.wq[name][:, k])
            new = torch.einsum("smi,joi->sjmo", new_a, w)
            u = torch.stack([self.matrix(value) for value in self.u[name]])
            error = self.error[name] + torch.matmul(
                new - old, u.transpose(-1, -2).unsqueeze(0)
            )
            sse = error.square().sum(dim=(0, 2, 3))
            updates[name] = (w, error, sse)
            total += float(sse.sum().item())
        return (total, (new_a, updates))

    def commit_v(
        self,
        k: int,
        seed: int,
        payload: tuple[
            torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ],
    ) -> None:
        activations, updates = payload
        self.v[k] = int(seed)
        for name, (weight, error, sse) in updates.items():
            self.aq[:, k] = activations
            self.wq[name][:, k] = weight
            self.error[name], self.row_sse[name] = (error, sse)

    def score_u(
        self, name: str, j: int, seed: int
    ) -> tuple[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        matrix = self.matrix(seed)
        new_wu = torch.matmul(matrix.transpose(0, 1), self.w[name][j])
        v = torch.stack([self.matrix(value) for value in self.v])
        new_wq = _a4_reconstruct(torch.matmul(new_wu, v))
        transformed = torch.einsum("skmi,koi->smo", self.aq, new_wq)
        error = (
            torch.matmul(transformed, matrix.transpose(0, 1))
            - self.reference[name][:, j]
        )
        sse = error.square().sum()
        return (
            self.total_error - float(self.row_sse[name][j].item()) + float(sse.item()),
            (new_wu, new_wq, error, sse),
        )

    def commit_u(
        self,
        name: str,
        j: int,
        seed: int,
        payload: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        wu, wq, error, sse = payload
        self.u[name][j] = int(seed)
        self.wu[name][j], self.wq[name][j] = (wu, wq)
        self.error[name][:, j], self.row_sse[name][j] = (error, sse)


def _decision(v: Sequence[int], u: Sequence[int]) -> dict[str, Any]:
    return {
        "variant": "R5",
        "t_seed": -1,
        "v_seeds": [int(seed) for seed in v],
        "u_seeds": [int(seed) for seed in u],
    }


def _select_r5_group(
    activations: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    *,
    group: str,
    members: Sequence[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    ordered_members = tuple((str(member) for member in members))
    if not ordered_members:
        raise ValueError("R5 group requires at least one projection.")
    if any(
        (name not in activations or name not in weights for name in ordered_members)
    ):
        raise ValueError(f"R5 group {group} has incomplete activations or weights.")
    if any(
        (
            int(weights[name].shape[1]) != int(weights[ordered_members[0]].shape[1])
            for name in ordered_members
        )
    ):
        raise ValueError(f"R5 group {group} does not share K geometry.")
    n_k = int(weights[ordered_members[0]].shape[1]) // TILE
    n_j = {name: int(weights[name].shape[0]) // TILE for name in ordered_members}
    plan = _r5_step_plan(ordered_members, n_k=n_k, n_j=n_j)
    v_seed = min(
        (
            (
                _R5A4Cache(
                    activations, weights, ordered_members, v_seed=seed
                ).total_error,
                seed,
            )
            for seed in SEEDS
        )
    )[1]
    uniform_u: dict[str, int] = {}
    for name in ordered_members:
        uniform_u[name] = min(
            (
                (
                    _R5A4Cache(
                        activations,
                        weights,
                        ordered_members,
                        v_seed=v_seed,
                        u_seed={**uniform_u, name: seed},
                    ).total_error,
                    seed,
                )
                for seed in SEEDS
            )
        )[1]
    cache = _R5A4Cache(
        activations, weights, ordered_members, v_seed=v_seed, u_seed=uniform_u
    )
    initial_error = cache.total_error
    next_step = 0
    for step_index in range(next_step, len(plan)):
        _sweep, axis, member, coordinate = plan[step_index]
        if axis == "v":
            old = cache.v[coordinate]
            best: tuple[float, int, Any | None] = (cache.total_error, old, None)
            for seed in SEEDS:
                if seed == old:
                    continue
                error, payload = cache.score_v(coordinate, seed)
                if (error, seed) < best[:2]:
                    best = (error, seed, payload)
            if best[1] != old:
                assert best[2] is not None
                cache.commit_v(coordinate, best[1], best[2])
        else:
            assert member is not None
            old = cache.u[member][coordinate]
            best = (cache.total_error, old, None)
            for seed in SEEDS:
                if seed == old:
                    continue
                error, payload = cache.score_u(member, coordinate, seed)
                if (error, seed) < best[:2]:
                    best = (error, seed, payload)
            if best[1] != old:
                assert best[2] is not None
                cache.commit_u(member, coordinate, best[1], best[2])
    decisions = {name: _decision(cache.v, cache.u[name]) for name in ordered_members}
    replay = _R5A4Cache(activations, weights, ordered_members)
    replay.v = list(cache.v)
    replay.u = {name: list(cache.u[name]) for name in ordered_members}
    replay._initialize()
    drift = abs(replay.total_error - cache.total_error)
    if drift > max(1e-10, abs(cache.total_error) * 1e-10):
        raise RuntimeError(f"R5 affected-coordinate replay drift for {group}: {drift}.")
    return (
        decisions,
        {
            "uniform_initial_error": float(initial_error),
            "blockwise_final_error": float(cache.total_error),
            "replay_drift": float(drift),
        },
    )


def _candidate_tensors(
    weight: torch.Tensor, r5: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    transformed = transform_weight_tiles(
        split_weight_tiles(weight), v_seeds=r5["v_seeds"], u_seeds=r5["u_seeds"]
    )
    q4, s4 = mse_quantize_tiles(transformed, bits=4)
    q8, s8 = mse_quantize_tiles(transformed, bits=8)
    return {
        "qweight4": q4.to(torch.int8).cpu(),
        "weight_scale4": s4.to(torch.float32).cpu(),
        "qweight8": q8.to(torch.int8).cpu(),
        "weight_scale8": s8.to(torch.float32).cpu(),
    }


def diagonal_fisher_w8_to_w4_risk(
    a8: torch.Tensor,
    output_gradient: torch.Tensor,
    delta: torch.Tensor,
    *,
    u_seeds: Sequence[int],
    token_count: int = 1,
) -> torch.Tensor:
    if a8.ndim != 3 or delta.ndim != 4 or int(a8.shape[1]) != int(delta.shape[1]):
        raise ValueError("Fisher operands have incompatible K64 geometry.")
    if (
        int(output_gradient.shape[0]) != int(a8.shape[0])
        or int(output_gradient.shape[1]) != int(delta.shape[0]) * TILE
    ):
        raise ValueError("Fisher output gradient does not match output tiles.")
    if len(u_seeds) != int(delta.shape[0]) or int(token_count) <= 0:
        raise ValueError("Fisher U coverage and token count must be valid.")
    grad = output_gradient.to(torch.float64).reshape(output_gradient.shape[0], -1, TILE)
    transformed_grad = torch.stack(
        [
            grad[:, j] @ structured_transform_matrix(u_seeds[j], device=grad.device)
            for j in range(len(u_seeds))
        ],
        dim=1,
    )
    result = torch.zeros(delta.shape[:2], dtype=torch.float64, device=delta.device)
    for k in range(delta.shape[1]):
        perturbation = torch.einsum("mi,joi->mjo", a8[:, k], delta[:, k])
        result[:, k] = (
            0.5
            * float(token_count) ** 2
            * (transformed_grad.square() * perturbation.square()).sum((0, 2))
        )
    return result


class _ExactW8Linear(torch.nn.Module):
    def __init__(
        self,
        original: torch.nn.Linear,
        values: Mapping[str, torch.Tensor],
        row: Mapping[str, Any],
        *,
        execution_device: torch.device,
    ) -> None:
        super().__init__()
        if original.weight.device.type != "cpu":
            raise ValueError(
                "Exact W8 Fisher requires the frozen weight to be host resident."
            )
        original.requires_grad_(False)
        self.frozen_weight_cpu = original.weight.detach()
        self.q4 = values["qweight4"].detach().to(device="cpu")
        self.s4 = values["weight_scale4"].detach().to(device="cpu")
        self.q8 = values["qweight8"].detach().to(device=execution_device)
        self.s8 = values["weight_scale8"].detach().to(device=execution_device)
        self.numeric_bias = (
            original.bias.detach().to(device=execution_device, dtype=torch.float64)
            if original.bias is not None
            else None
        )
        self.v = tuple((int(seed) for seed in row["r5"]["v_seeds"]))
        self.u = tuple((int(seed) for seed in row["r5"]["u_seeds"]))
        self.last_a8: torch.Tensor | None = None
        self.last_gradient: torch.Tensor | None = None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original_shape = tuple(value.shape)
        flat = value.reshape(-1, int(value.shape[-1]))
        rows, width = (int(flat.shape[0]), int(flat.shape[1]))
        if width % TILE:
            raise ValueError("Exact W8A8 requires K64 input.")
        padded = (16 - rows % 16) % 16
        work = F.pad(flat, (0, 0, 0, padded)) if padded else flat
        n_k = width // TILE
        x = work.reshape(-1, n_k, TILE).permute(1, 0, 2)
        transformed = transform_activation_tiles(x, v_seeds=self.v).to(
            device=value.device
        )
        a8 = _endpoint_a8_rows(transformed).permute(1, 0, 2)[:rows]
        weight = reconstruct_tiles(self.q8, self.s8, dtype=torch.float64)
        transformed_output = torch.einsum("mki,jkoi->mjo", a8, weight)
        output = (
            inverse_output_tiles(transformed_output.permute(1, 0, 2), u_seeds=self.u)
            .permute(1, 0, 2)
            .reshape(rows, -1)
        )
        if self.numeric_bias is not None:
            output = output + self.numeric_bias
        numeric = output.to(dtype=value.dtype).reshape(*original_shape[:-1], -1)
        result = _FrozenLinearInputGradient.apply(
            value, numeric.detach(), self.frozen_weight_cpu
        )
        self.last_a8 = a8.detach().to(device="cpu")
        self.last_gradient = None
        if result.requires_grad:
            result.register_hook(self._capture_gradient)
        return result

    def _capture_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        self.last_gradient = (
            gradient.detach()
            .reshape(-1, gradient.shape[-1])
            .to(device="cpu", dtype=torch.float64)
        )
        return gradient

    def risk(self, *, token_count: int = 1) -> torch.Tensor:
        if self.last_a8 is None or self.last_gradient is None:
            raise RuntimeError(
                "Fisher capture is missing activation or output gradient."
            )
        device = self.q8.device
        try:
            q4 = self.q4.to(device=device)
            s4 = self.s4.to(device=device)
            delta = reconstruct_tiles(q4, s4, dtype=torch.float64)
            delta.sub_(reconstruct_tiles(self.q8, self.s8, dtype=torch.float64))
            result = diagonal_fisher_w8_to_w4_risk(
                self.last_a8.to(device=device),
                self.last_gradient.to(device=device),
                delta,
                u_seeds=self.u,
                token_count=token_count,
            )
            return result.to(device="cpu")
        finally:
            self.last_a8 = None
            self.last_gradient = None


class _FrozenLinearInputGradient(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        value: torch.Tensor,
        numeric: torch.Tensor,
        frozen_weight_cpu: torch.Tensor,
    ) -> torch.Tensor:
        if frozen_weight_cpu.device.type != "cpu" or frozen_weight_cpu.requires_grad:
            raise ValueError("Frozen STE weights must be non-gradient host tensors.")
        if (
            value.ndim == 0
            or numeric.shape[:-1] != value.shape[:-1]
            or int(value.shape[-1]) != int(frozen_weight_cpu.shape[1])
            or (int(numeric.shape[-1]) != int(frozen_weight_cpu.shape[0]))
        ):
            raise ValueError(
                "Frozen STE input, output, and weight shapes are incompatible."
            )
        ctx.save_for_backward(frozen_weight_cpu)
        ctx.input_shape = tuple(value.shape)
        return numeric

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        (frozen_weight_cpu,) = ctx.saved_tensors
        weight = frozen_weight_cpu.to(device=gradient.device)
        flat = gradient.reshape(-1, int(gradient.shape[-1]))
        grad_input = flat.matmul(weight).reshape(ctx.input_shape)
        return (grad_input, None, None)


def _score_captures(
    replacements: Mapping[str, _ExactW8Linear], *, token_count: int
) -> dict[str, torch.Tensor]:
    return {
        name: module.risk(token_count=token_count)
        for name, module in replacements.items()
    }


def _fisher_record_values(
    model: torch.nn.Module,
    replacements: Mapping[str, _ExactW8Linear],
    record: Mapping[str, Any],
    *,
    phase: str,
    device: torch.device,
    max_length: int,
    decode_proxy_steps: int,
) -> tuple[dict[str, torch.Tensor], list[int], int]:
    if phase == "prefill":
        model.zero_grad(set_to_none=True)
        loss, tokens = _loss_for_record(
            model, record, device=device, max_length=max_length
        )
        loss.backward()
        _assert_no_parameter_grads(model)
        return (_score_captures(replacements, token_count=tokens), [], tokens)
    if phase != "decode":
        raise ValueError(f"Unsupported Fisher phase: {phase!r}.")
    prompt_ids, target_ids = _prompt_target(record, max_length=max_length)
    if len(target_ids) < 2:
        raise ValueError("Decode Fisher calibration needs at least two target tokens.")
    values = {
        name: torch.zeros(
            (module.q8.shape[0], module.q8.shape[1]), dtype=torch.float64, device="cpu"
        )
        for name, module in replacements.items()
    }
    selected = set(
        deterministic_decode_proxy_indices(len(target_ids), decode_proxy_steps)
    )
    prompt = torch.tensor(prompt_ids, device=device, dtype=torch.long)[None, :]
    with torch.no_grad():
        initial = model(
            input_ids=prompt, attention_mask=torch.ones_like(prompt), use_cache=True
        )
    past = _detach_past(initial.past_key_values)
    for index in range(1, len(target_ids)):
        token = torch.tensor([[target_ids[index - 1]]], device=device, dtype=torch.long)
        attention = torch.ones(
            (1, len(prompt_ids) + index), device=device, dtype=torch.long
        )
        if index in selected:
            model.zero_grad(set_to_none=True)
            output = model(
                input_ids=token,
                attention_mask=attention,
                past_key_values=past,
                use_cache=True,
            )
            label = torch.tensor([target_ids[index]], device=device)
            F.cross_entropy(output.logits[:, -1].float(), label).backward()
            _assert_no_parameter_grads(model)
            for name, value in _score_captures(replacements, token_count=1).items():
                values[name].add_(value)
        else:
            with torch.no_grad():
                output = model(
                    input_ids=token,
                    attention_mask=attention,
                    past_key_values=past,
                    use_cache=True,
                )
        past = _detach_past(output.past_key_values)
    return (values, sorted(selected), len(target_ids))


def _prepare_fisher_model(model: torch.nn.Module) -> None:
    model.requires_grad_(False)
    enable = getattr(model, "enable_input_require_grads", None)
    if not callable(enable):
        raise ValueError(
            "Frozen Fisher model must expose enable_input_require_grads()."
        )
    enable()


def _assert_no_parameter_grads(model: torch.nn.Module) -> None:
    if any((parameter.grad is not None for parameter in model.parameters())):
        raise RuntimeError("Fisher STE allocated a model-parameter gradient.")


def _loss_for_record(
    model: torch.nn.Module,
    record: Mapping[str, Any],
    *,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, int]:
    prompt, target = _prompt_target(record, max_length=max_length)
    ids = torch.tensor(prompt + target, device=device, dtype=torch.long)[None, :]
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    start = len(prompt) - 1
    labels = torch.tensor(target, device=device, dtype=torch.long)
    logits = output.logits[:, start : start + len(target)].float()
    return (
        F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels),
        int(labels.numel()),
    )
