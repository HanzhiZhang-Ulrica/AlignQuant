from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import transformers

from alignquant.models.decoder import iter_target_linears
from alignquant.quantization.artifact import build_artifact, unpack_w4w8_state_bits
from alignquant.quantization.exact import (
    mse_quantize_tiles,
    split_weight_tiles,
    transform_weight_tiles,
)


class TransformPlan:
    def __init__(self, row: dict) -> None:
        self.v_seeds = row["r5"]["v_seeds"]
        self.u_seeds = row["r5"]["u_seeds"]

    def normalized(self, *, n_k: int, n_j: int) -> SimpleNamespace:
        if len(self.v_seeds) != n_k or len(self.u_seeds) != n_j:
            raise ValueError("Calibration transform geometry differs from model.")
        return SimpleNamespace(
            variant="R5",
            t_seed=-1,
            v_seeds=self.v_seeds,
            u_seeds=self.u_seeds,
        )


def build_from_calibration(
    model_path: str, calibration_path: str, output_path: str
) -> Path:
    calibration = json.loads(Path(calibration_path).read_text())
    if calibration.get("format") != "alignquant-calibration-v1":
        raise ValueError("Unsupported calibration format.")
    if calibration.get("tile_size") != 64:
        raise ValueError("Calibration tile size must be 64.")
    if not 0.5999 <= float(calibration.get("w4_fraction", -1)) <= 0.6001:
        raise ValueError("Calibration must use the 60% W4 precision budget.")
    loader = getattr(transformers, calibration["loader"])
    model = loader.from_pretrained(
        model_path,
        revision=calibration["revision"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).eval()
    checkpoint_revision = getattr(model.config, "_commit_hash", None)
    if checkpoint_revision and checkpoint_revision != calibration["revision"]:
        raise ValueError(
            "Loaded checkpoint revision differs from the calibration result."
        )
    linears = {name: linear for _, _, name, linear in iter_target_linears(model)}
    rows = {row["name"]: row for row in calibration["modules"]}
    if linears.keys() != rows.keys():
        raise ValueError("Calibration/model projection coverage differs.")

    def modules():
        for name in sorted(linears):
            linear = linears[name]
            row = rows[name]
            shape = tuple(int(value) for value in row["shape"])
            if tuple(linear.weight.shape) != shape:
                raise ValueError(f"Projection shape differs for {name}.")
            n_j, n_k = shape[0] // 64, shape[1] // 64
            states = unpack_w4w8_state_bits(
                torch.tensor(
                    list(bytes.fromhex(row["w4_state_bits_lsb_hex"])), dtype=torch.uint8
                ),
                n_j=n_j,
                n_k=n_k,
            )
            weight = linear.weight.detach().to(device="cpu", dtype=torch.float64)
            transformed = transform_weight_tiles(
                split_weight_tiles(weight),
                v_seeds=row["r5"]["v_seeds"],
                u_seeds=row["r5"]["u_seeds"],
            )
            q4, s4 = mse_quantize_tiles(transformed, bits=4)
            q8, s8 = mse_quantize_tiles(transformed, bits=8)
            yield SimpleNamespace(
                layer_name=name,
                weight_state_map=states,
                qweight4=q4.to(torch.int8),
                weight_scale4=s4.to(torch.float32),
                qweight8=q8.to(torch.int8),
                weight_scale8=s8.to(torch.float32),
                n_j=n_j,
                n_k=n_k,
                in_features=shape[1],
                out_features=shape[0],
                p4_codebook=None,
                plan=TransformPlan(row),
            )

    total_tiles = sum(
        (int(row["shape"][0]) // 64) * (int(row["shape"][1]) // 64)
        for row in calibration["modules"]
    )

    def checked_modules():
        w4_tiles = 0
        for module in modules():
            w4_tiles += int((module.weight_state_map == 4).sum())
            yield module
        if abs(w4_tiles / total_tiles - 0.6) > 1.0e-5:
            raise ValueError("Calibration tile states do not meet the 60% W4 budget.")

    return build_artifact(
        output_path,
        modules=checked_modules(),
        expected_total_tiles=total_tiles,
        model_identity={
            "id": calibration["model"],
            "revision": calibration["revision"],
            "loader": calibration["loader"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an AlignQuant inference artifact."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--calibration",
        default=str(
            Path(__file__).resolve().parent / "data/llama-3.2-3b-instruct.json"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(build_from_calibration(args.model, args.calibration, args.output))


if __name__ == "__main__":
    main()
