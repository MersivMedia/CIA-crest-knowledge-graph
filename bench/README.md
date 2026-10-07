# Benchmark harness

Reproduces [`BENCHMARK_RESULTS.md`](../BENCHMARK_RESULTS.md) on any single 24 GB+ NVIDIA GPU.

```bash
# on the GPU box, from the repo root copied to /root/bench
python3 -m venv /root/venv && . /root/venv/bin/activate
pip install "sentence-transformers[image]>=6.1.0" "transformers>=5.19" pymupdf pillow
bash bench/pod_vlm.sh                # fetch pages, run Qwen3-VL 2B/4B/8B-FP8, compare, embed
python bench/cost_model.py /root/out3 /root/out/embed_bench.json
```

`bench_candidates.json` is a fixed, stratified sample of manifest rows. Pages are fetched from
archive.org first and the Wayback copy of the cia.gov PDF second; every page's `citation` is the
official `https://www.cia.gov/readingroom/docs/<doc>.pdf#page=N`.

On pods without `nvcc`, set `VLLM_USE_FLASHINFER_SAMPLER=0` (the script does) or vLLM fails while
JIT-compiling its sampler. Keep model weights off a small container disk: 2B + 4B + 8B-FP8 is ~19 GB.
