from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import transformers
from safetensors.torch import load_file, save_file

from alignquant.models.decoder import iter_target_linears
from alignquant.quantization.artifact import pack_w4w8_state_bits
from alignquant.quantization.calibration import (
    _ExactW8Linear,
    _candidate_tensors,
    _fisher_record_values,
    _input_rows,
    _prepare_fisher_model,
    _projection_groups,
    _select_r5_group,
)


def calibration_records(samples: Path, tokenizer) -> dict[str, list[dict]]:
    with samples.open() as stream:
        source = [json.loads(line) for line in stream if line.strip()]
    if len(source) != 64:
        raise ValueError("Provide 64 ordered document/summary records.")
    rows = []
    for row in source:
        prompt_text = (
            "Summarize the following text.\n\nText:\n"
            + row["document"].strip()
            + "\n\nSummary:\n"
        )
        prompt = tokenizer.encode(
            prompt_text, add_special_tokens=False, truncation=True, max_length=4096
        )
        target = tokenizer.encode(
            row["summary"].strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=128,
        )
        if tokenizer.bos_token_id is not None:
            prompt = [tokenizer.bos_token_id, *prompt]
        if tokenizer.eos_token_id is not None:
            target.append(tokenizer.eos_token_id)
        if not prompt or len(target) < 2:
            raise ValueError(
                "Calibration requires nonempty prompts and at least two target tokens."
            )
        rows.append(
            {"prompt_ids": prompt, "target_ids": target, "input_ids": prompt + target}
        )
    windows = []
    used = set()
    for bucket, length in (
        ("extra_long", 2049),
        ("long", 1024),
        ("medium", 512),
        ("short", 128),
    ):
        selected = 0
        for index, row in enumerate(rows):
            sequence = row["input_ids"]
            if index in used or selected == 16 or len(sequence) < length + 32:
                continue
            windows.append(
                (
                    bucket,
                    index,
                    {
                        "prompt_ids": sequence[:length],
                        "target_ids": sequence[length : length + 32],
                        "input_ids": sequence[: length + 32],
                    },
                )
            )
            used.add(index)
            selected += 1
        if selected != 16:
            raise ValueError(
                f"Insufficient documents for the {bucket} calibration bucket."
            )
    windows.sort(key=lambda item: (item[0], item[1]))
    return {"prefill": rows[:24], "decode": [item[2] for item in windows[:24]]}


def allocate(scores: dict[str, dict[str, torch.Tensor]], modules: list[dict]) -> float:
    regimes = {
        phase: {
            f"{name}:{j}:{k}": float(values[j, k])
            for name, values in layers.items()
            for j in range(values.shape[0])
            for k in range(values.shape[1])
        }
        for phase, layers in scores.items()
    }
    normalized = {}
    for phase, values in regimes.items():
        total = sum(values.values())
        if (
            not math.isfinite(total)
            or total <= 0
            or any(v < 0 or not math.isfinite(v) for v in values.values())
        ):
            raise ValueError(f"Invalid Fisher scores for {phase}.")
        normalized[phase] = {key: value / total for key, value in values.items()}
    if normalized["prefill"].keys() != normalized["decode"].keys():
        raise ValueError("Prefill/decode Fisher coverage differs.")
    risk = {
        key: max(values[key] for values in normalized.values())
        for key in normalized["prefill"]
    }
    selected = set(
        sorted(risk, key=lambda key: (risk[key], key))[: round(len(risk) * 0.6)]
    )
    for row in modules:
        n_j, n_k = (width // 64 for width in row["shape"])
        states = torch.full((n_j, n_k), 8, dtype=torch.uint8)
        for j in range(n_j):
            for k in range(n_k):
                if f"{row['name']}:{j}:{k}" in selected:
                    states[j, k] = 4
        row["w4_state_bits_lsb_hex"] = bytes(
            pack_w4w8_state_bits(states).tolist()
        ).hex()
    return len(selected) / len(risk)


def calibrate(args) -> None:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    preset = json.loads(Path(args.preset).read_text())
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        revision=preset["revision"],
        trust_remote_code=False,
    )
    records = calibration_records(Path(args.samples), tokenizer)
    device = torch.device(args.device)
    model = (
        getattr(transformers, preset["loader"])
        .from_pretrained(
            args.model,
            revision=preset["revision"],
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        .to(device)
        .eval()
    )
    revision = getattr(model.config, "_commit_hash", None)
    if revision and revision != preset["revision"]:
        raise ValueError("Checkpoint revision differs from the selected model preset.")
    targets = list(iter_target_linears(model))
    linears = {name: linear for _, _, name, linear in targets}
    if not linears:
        raise ValueError("No supported decoder projections found.")
    expected_shapes = {row["name"]: tuple(row["shape"]) for row in preset["modules"]}
    if {
        name: tuple(layer.weight.shape) for name, layer in linears.items()
    } != expected_shapes:
        raise ValueError("Model projection geometry differs from the selected preset.")
    decisions = {}
    for group, members in _projection_groups(tuple(linears)).items():
        print(f"R5: {group}", flush=True)
        captured = _input_rows(
            model,
            records["prefill"],
            device=device,
            max_length=2048,
            module_names=(members[0],),
            max_rows=1024,
        )
        activations = {name: captured[members[0]] for name in members}
        weights = {
            name: linears[name].weight.detach().to("cpu", torch.float64)
            for name in members
        }
        chosen, _ = _select_r5_group(activations, weights, group=group, members=members)
        decisions.update(chosen)
        del captured, activations, weights
    modules = [
        {
            "name": name,
            "shape": list(linears[name].weight.shape),
            "r5": {key: decisions[name][key] for key in ("v_seeds", "u_seeds")},
        }
        for name in sorted(linears)
    ]
    with TemporaryDirectory(prefix="alignquant-", dir=args.work_dir) as temporary:
        candidate_paths = {}
        for index, row in enumerate(modules):
            name = row["name"]
            print(f"Candidates: {name}", flush=True)
            values = _candidate_tensors(
                linears[name].weight.detach().to("cpu", torch.float64), row["r5"]
            )
            candidate_paths[name] = str(Path(temporary) / f"{index}.safetensors")
            save_file(values, candidate_paths[name])
            del values
        _prepare_fisher_model(model)
        replacements = {}
        for parent, attribute, name, linear in targets:
            linear.to("cpu")
            replacement = _ExactW8Linear(
                linear,
                load_file(candidate_paths[name]),
                {"r5": decisions[name]},
                execution_device=device,
            )
            setattr(parent, attribute, replacement)
            replacements[name] = replacement
        scores = {"prefill": {}, "decode": {}}
        for phase, phase_records in records.items():
            for index, record in enumerate(phase_records):
                print(f"Fisher {phase}: {index + 1}/{len(phase_records)}", flush=True)
                values, _, _ = _fisher_record_values(
                    model,
                    replacements,
                    record,
                    phase=phase,
                    device=device,
                    max_length=4226,
                    decode_proxy_steps=4,
                )
                for name, value in values.items():
                    scores[phase][name] = (
                        scores[phase].get(name, torch.zeros_like(value)) + value
                    )
        fraction = allocate(scores, modules)
    result = {
        "format": "alignquant-calibration-v1",
        "tile_size": 64,
        "w4_fraction": fraction,
        "model": preset["model"],
        "revision": preset["revision"],
        "loader": preset["loader"],
        "samples_sha256": hashlib.sha256(Path(args.samples).read_bytes()).hexdigest(),
        "modules": modules,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute R5 transforms and global Fisher tile allocation."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument(
        "--preset",
        default=str(Path(__file__).parent / "data/llama-3.2-3b-instruct.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", default=None)
    calibrate(parser.parse_args())


if __name__ == "__main__":
    main()
