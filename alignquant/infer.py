from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer

from alignquant.kernels.cuda.extension import load_extension
from alignquant.quantization.model import install_alignquant, reset_alignquant_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Run greedy generation with AlignQuant.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("AlignQuant inference requires CUDA.")
    manifest = json.loads((Path(args.artifact) / "manifest.json").read_text())
    model_info = manifest.get("model", {})
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=model_info.get("revision"),
        trust_remote_code=False,
    )
    loader = getattr(transformers, model_info.get("loader", "AutoModelForCausalLM"))
    model = loader.from_pretrained(
        args.model,
        revision=model_info.get("revision"),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to("cuda").eval()
    install_alignquant(model, args.artifact, "cuda")
    load_extension()
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    reset_alignquant_state(model)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
