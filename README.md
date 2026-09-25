# AlignQuant

**Tile-Aligned Mixed-Precision Quantization for Efficient LLM Generation**

AlignQuant uses 64×64 weight tiles as the common unit of precision allocation, scale sharing, packed storage, and GPU computation. Joint prefill/decode calibration scores loss-gradient-weighted W8-to-W4 projection-output changes and assigns W4 to the lowest-scoring 60% of tiles model-wide. Prefill and decode share one packed W4/W8 model with A8 activations and INT8 arithmetic.

Requires Python 3.10+, an SM80 GPU, CUDA toolkit, a C++ compiler, and CUTLASS headers.

```bash
pip install -e .
export ALIGNQUANT_CUTLASS_PATH=/path/to/cutlass

python -m alignquant.calibrate --model /path/to/model \
  --samples samples.jsonl --output calibration.json --work-dir /path/to/scratch
python -m alignquant.build --model /path/to/model \
  --calibration calibration.json --output artifacts/model
python -m alignquant.infer --model /path/to/model --artifact artifacts/model \
  --prompt "Explain quantization." --max-new-tokens 128
```

`samples.jsonl` contains 64 ordered records with `document` and `summary` strings. The commands select tile-aligned transforms, compute the joint precision map, pack the selected weights and scales, and generate text. The first inference compiles the CUDA extension.

The default preset is Llama-3.2-3B-Instruct. For Qwen3-4B, Ministral-3-8B, or Qwen3-14B, pass the corresponding `alignquant/data/*.json` to `calibrate --preset` and use the checkpoint revision recorded there. Bundled precision maps can also be used directly with `build --calibration`.
