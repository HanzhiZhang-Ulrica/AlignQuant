"""Numerical primitives shared by calibration and the CUDA runtime."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


TILE = 64
IDENTITY_SEED = -1
MSE_SCALE_RATIOS = tuple(0.25 + index / 128.0 for index in range(129))


def structured_transform_metadata(
    seed: int, *, device: torch.device | str | None = None
) -> torch.Tensor:
    """Return ``(mode, permutation, left sign, right sign)`` for one R5 seed."""

    value = int(seed)
    if value < IDENTITY_SEED:
        raise ValueError(f"Transform seed must be >= {IDENTITY_SEED}, got {value}.")
    metadata = torch.empty((4, TILE), dtype=torch.int32)
    metadata[0].fill_(0 if value == IDENTITY_SEED else 1)
    metadata[1] = torch.arange(TILE, dtype=torch.int32)
    metadata[2:].fill_(1)
    if value > 0:
        generator = torch.Generator(device="cpu").manual_seed(0x52464451 + value)
        metadata[1] = torch.randperm(TILE, generator=generator).to(torch.int32)
        for row in (2, 3):
            metadata[row] = (
                torch.randint(0, 2, (TILE,), generator=generator, dtype=torch.int64)
                .to(torch.int32)
                .mul_(2)
                .sub_(1)
            )
    return metadata.to(device=device) if device is not None else metadata


def normalized_hadamard64(
    *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None
) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float64, device=device)
    while int(matrix.shape[0]) < TILE:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return (matrix / math.sqrt(TILE)).to(dtype=dtype)


def structured_transform_matrix(
    seed: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    metadata = structured_transform_metadata(seed)
    if int(metadata[0, 0]) == 0:
        return torch.eye(TILE, dtype=dtype, device=device)
    matrix = normalized_hadamard64(dtype=torch.float64, device="cpu")
    matrix = (
        metadata[2].to(torch.float64)[:, None]
        * matrix[metadata[1].to(torch.long), :]
        * metadata[3].to(torch.float64)[None, :]
    )
    return matrix.to(dtype=dtype, device=device)


def split_weight_tiles(weight: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError(f"Weight must be rank two, got {tuple(weight.shape)}.")
    n, k = map(int, weight.shape)
    if n % TILE or k % TILE:
        raise ValueError(f"Weight shape must be N64/K64 aligned, got {(n, k)}.")
    return (
        weight.reshape(n // TILE, TILE, k // TILE, TILE)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def dense_weight_from_tiles(tiles: torch.Tensor) -> torch.Tensor:
    if tiles.ndim != 4 or tuple(tiles.shape[-2:]) != (TILE, TILE):
        raise ValueError(f"Weight tiles must be [J,K,64,64], got {tuple(tiles.shape)}.")
    n_j, n_k = map(int, tiles.shape[:2])
    return tiles.permute(0, 2, 1, 3).reshape(n_j * TILE, n_k * TILE).contiguous()


def transform_weight_tiles(
    tiles: torch.Tensor, *, v_seeds: Sequence[int], u_seeds: Sequence[int]
) -> torch.Tensor:
    if tiles.ndim != 4 or tuple(tiles.shape[-2:]) != (TILE, TILE):
        raise ValueError(f"Weight tiles must be [J,K,64,64], got {tuple(tiles.shape)}.")
    n_j, n_k = map(int, tiles.shape[:2])
    if len(v_seeds) != n_k or len(u_seeds) != n_j:
        raise ValueError("R5 seed coverage does not match weight tile geometry.")
    work = tiles.to(torch.float64)
    v = torch.stack(
        [
            structured_transform_matrix(seed, device=work.device)
            for seed in v_seeds
        ]
    )
    u = torch.stack(
        [
            structured_transform_matrix(seed, device=work.device)
            for seed in u_seeds
        ]
    )
    return torch.matmul(
        torch.matmul(u.transpose(-1, -2).unsqueeze(1), work), v.unsqueeze(0)
    ).contiguous()


def transform_activation_tiles(
    tiles: torch.Tensor, *, v_seeds: Sequence[int]
) -> torch.Tensor:
    if tiles.ndim < 3 or int(tiles.shape[-1]) != TILE:
        raise ValueError(f"Activation tiles must end in [K,*,64], got {tuple(tiles.shape)}.")
    n_k = int(tiles.shape[-3])
    if len(v_seeds) != n_k:
        raise ValueError("R5 V seed coverage does not match activation K tiles.")
    work = tiles.to(torch.float64)
    v = torch.stack(
        [
            structured_transform_matrix(seed, device=work.device)
            for seed in v_seeds
        ]
    )
    return torch.matmul(work, v).contiguous()


def inverse_output_tiles(
    tiles: torch.Tensor, *, u_seeds: Sequence[int]
) -> torch.Tensor:
    if tiles.ndim < 3 or int(tiles.shape[-1]) != TILE:
        raise ValueError(f"Output tiles must end in [J,*,64], got {tuple(tiles.shape)}.")
    n_j = int(tiles.shape[-3])
    if len(u_seeds) != n_j:
        raise ValueError("R5 U seed coverage does not match output J tiles.")
    work = tiles.to(torch.float64)
    u = torch.stack(
        [
            structured_transform_matrix(seed, device=work.device)
            for seed in u_seeds
        ]
    )
    return torch.matmul(work, u.transpose(-1, -2)).contiguous()


def integer_bounds(bits: int) -> tuple[int, int]:
    value = int(bits)
    if value == 4:
        return -8, 7
    if value == 8:
        return -128, 127
    raise ValueError(f"Only signed W4/W8--A8 are supported, got {bits} bits.")


def mse_scales(
    tiles: torch.Tensor,
    *,
    bits: int,
    ratios: Sequence[float] = MSE_SCALE_RATIOS,
    scale_floor: float = 1.0e-12,
    max_workspace_elements: int = 4 * 1024 * 1024,
) -> torch.Tensor:
    """Choose one endpoint-aware MSE scale per complete operand tile."""

    if tiles.ndim < 2 or int(tiles.shape[-1]) != TILE:
        raise ValueError("Quantized operands must end in complete *x64 tiles.")
    qmin, qmax = integer_bounds(bits)
    values = tiles.detach().to(torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Quantized operands contain non-finite values.")
    leading = tuple(int(value) for value in values.shape[:-2])
    flat = values.reshape(-1, int(values.shape[-2]) * TILE)
    positive = flat.clamp_min(0).amax(dim=1) / float(qmax)
    negative = (-flat.clamp_max(0)).amax(dim=1) / float(abs(qmin))
    nonzero = (flat != 0).any(dim=1)
    base = torch.maximum(positive, negative).clamp_min(float(scale_floor))
    ratio_values = torch.as_tensor(tuple(ratios), dtype=torch.float64, device=flat.device)
    if ratio_values.numel() != 129 or not bool(torch.isfinite(ratio_values).all()):
        raise ValueError("The frozen quantizer requires exactly 129 finite scale ratios.")
    if bool((ratio_values <= 0).any()):
        raise ValueError("Scale ratios must be positive.")
    best_error = torch.full_like(base, float("inf"))
    best_scale = base.clone()
    chunk = max(
        1,
        min(
            int(ratio_values.numel()),
            int(max_workspace_elements) // max(int(flat.numel()), 1),
        ),
    )
    for start in range(0, int(ratio_values.numel()), chunk):
        candidates = (
            base[:, None] * ratio_values[start : start + chunk][None, :]
        ).clamp_min(float(scale_floor))
        quantized = torch.round(flat[:, None, :] / candidates[:, :, None]).clamp(
            qmin, qmax
        )
        error = ((flat[:, None, :] - quantized * candidates[:, :, None]) ** 2).mean(2)
        local_error, local_index = error.min(1)
        local_scale = candidates.gather(1, local_index[:, None]).squeeze(1)
        better = local_error < best_error
        best_error = torch.where(better, local_error, best_error)
        best_scale = torch.where(better, local_scale, best_scale)
    best_scale = torch.where(
        nonzero, best_scale, torch.full_like(best_scale, float(scale_floor))
    )
    return best_scale.reshape(leading).contiguous()


def quantize_tiles(
    tiles: torch.Tensor, scales: torch.Tensor, *, bits: int
) -> torch.Tensor:
    if tiles.ndim < 2 or int(tiles.shape[-1]) != TILE:
        raise ValueError("Quantized operands must end in complete *x64 tiles.")
    expected = tuple(int(value) for value in tiles.shape[:-2])
    scale = torch.as_tensor(scales, dtype=torch.float64, device=tiles.device)
    if tuple(scale.shape) != expected:
        raise ValueError(f"Scale grid must have shape {expected}, got {tuple(scale.shape)}.")
    if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise ValueError("Every complete tile scale must be finite and positive.")
    qmin, qmax = integer_bounds(bits)
    return torch.round(tiles.to(torch.float64) / scale[..., None, None]).clamp(
        qmin, qmax
    ).to(torch.int8).contiguous()


def mse_quantize_tiles(
    tiles: torch.Tensor, *, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = mse_scales(tiles, bits=bits)
    return quantize_tiles(tiles, scales, bits=bits), scales


def a8_endpoint_scales(
    tiles: torch.Tensor, *, scale_floor: float = 1.0e-12
) -> torch.Tensor:
    """Return the frozen endpoint-aware A8 scale for each M16xK64 tile.

    Unlike weights, activations do not use the 129-candidate MSE search.  The
    CUDA path uses one scale over each complete 16-by-64 operand tile with
    asymmetric signed endpoints: positive values divide by 127 and negative
    values divide by 128.
    """

    if tiles.ndim < 2 or tuple(tiles.shape[-2:]) != (16, TILE):
        raise ValueError("A8 endpoint quantization requires complete [...,16,64] tiles.")
    values = tiles.detach().to(torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Activation tiles contain non-finite values.")
    positive = values.clamp_min(0).amax(dim=(-2, -1)) / 127.0
    negative = (-values.clamp_max(0)).amax(dim=(-2, -1)) / 128.0
    return torch.maximum(positive, negative).clamp_min(float(scale_floor)).contiguous()


def a8_endpoint_quantize_tiles(
    tiles: torch.Tensor, *, scale_floor: float = 1.0e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize complete M16xK64 activation operands for the native A8 path."""

    scales = a8_endpoint_scales(tiles, scale_floor=scale_floor)
    quantized = torch.round(tiles.to(torch.float64) / scales[..., None, None]).clamp(
        -128, 127
    ).to(torch.int8)
    return quantized.contiguous(), scales


def reconstruct_tiles(
    quantized: torch.Tensor, scales: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    return (
        quantized.to(torch.float64)
        * torch.as_tensor(scales, dtype=torch.float64, device=quantized.device)[
            ..., None, None
        ]
    ).to(dtype=dtype)


__all__ = [
    "IDENTITY_SEED",
    "MSE_SCALE_RATIOS",
    "TILE",
    "a8_endpoint_quantize_tiles",
    "a8_endpoint_scales",
    "dense_weight_from_tiles",
    "integer_bounds",
    "inverse_output_tiles",
    "mse_quantize_tiles",
    "mse_scales",
    "normalized_hadamard64",
    "quantize_tiles",
    "reconstruct_tiles",
    "split_weight_tiles",
    "structured_transform_matrix",
    "structured_transform_metadata",
    "transform_activation_tiles",
    "transform_weight_tiles",
]
