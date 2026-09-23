# AlignQuant

64×64 tile-aligned W4/W8 allocation, scales, and native execution with A8 activations.
Requires Python 3.10+, CUDA toolkit, a C++ compiler, and CUTLASS headers. The current native kernel targets SM80.
Use `ALIGNQUANT_CXX`, `ALIGNQUANT_CUDA_HOME`, and `TORCH_EXTENSIONS_DIR` to select the compiler, CUDA toolkit, and build-cache directory when needed.

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

`samples.jsonl`: 64 ordered JSON lines with `document` and `summary` strings (the reference corpus uses `alexfabbri/multi_news` training rows 544–607). Calibration tokenizes them, selects R5 transforms, constructs W4/W8 candidates, computes prefill/decode Fisher risk, and assigns the globally lowest-risk 60% of tiles to W4. It uses 24 prefill records and the first 24 sorted matched-prefix decode windows; temporary candidates are removed on exit.

The default model preset is Llama-3.2-3B-Instruct. For another supported model, pass its `alignquant/data/*.json` file via `--preset`; the exact checkpoint revision is recorded there. To skip recalibration, pass that file directly to `alignquant.build --calibration`. Model weights and raw samples are not bundled. The first inference compiles the CUDA extension.
