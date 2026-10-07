#!/bin/bash
# Runs ON the bench pod (tmux). Page-reader bake-off: Qwen3-VL 2B / 4B / 8B-FP8 on the same CREST pages, then
# EmbeddingGemma 2 retrieval on each reader's page chunks. Writes /root/out3/.
# Needs: a 24 GB+ NVIDIA GPU, /root/bench/{schema,fetch_pages,vlm_pages,vlm_compare,embed_bench}.py,
# bench_candidates.json, and a /root/venv with sentence-transformers>=6.1 + transformers>=5.19 + pymupdf.
set -x
O=/root/out3; mkdir -p $O /root/pages2; cd /root/bench
export HF_HOME=/workspace/hf PIP_DISABLE_PIP_VERSION_CHECK=1 HF_HUB_ENABLE_HF_TRANSFER=0
# no nvcc on the pod: FlashInfer JIT (sampler) cannot compile -> use the PyTorch sampler
export VLLM_USE_FLASHINFER_SAMPLER=0
[ -s /root/pages2/pages.json ] || ( . /root/venv/bin/activate; python3 -u fetch_pages.py --candidates bench_candidates.json --out-dir /root/pages2 \
    --max-pages 8 --workers 8 > $O/fetch.log 2>&1 ) &
FETCH=$!
[ -x /root/vvenv/bin/vllm ] || { python3 -m venv /root/vvenv && /root/vvenv/bin/pip install -q "vllm==0.31.0" > $O/pip_vllm.log 2>&1; }
. /root/vvenv/bin/activate
python3 -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)" > $O/versions.txt 2>&1
wait $FETCH
python3 - <<'PY' > $O/sample.txt
import json; p=json.load(open('/root/pages2/pages.json')); print(len(p), 'pages', len({x['doc_id'] for x in p}), 'docs')
PY
[ -s $O/vlm_2b.json ] && echo "2b already done"
for M in "2b Qwen/Qwen3-VL-2B-Instruct" "4b Qwen/Qwen3-VL-4B-Instruct" "8b Qwen/Qwen3-VL-8B-Instruct-FP8"; do
  set -- $M; T=$1; NAME=$2
  [ -s $O/vlm_$T.json ] && continue
  vllm serve $NAME --port 8000 --max-model-len 12288 --gpu-memory-utilization 0.90 --max-num-seqs 64 \
     --limit-mm-per-prompt '{"image":1}' > $O/serve_$T.log 2>&1 &
  SV=$!
  for i in $(seq 1 120); do curl -sf localhost:8000/v1/models >/dev/null && break; kill -0 $SV 2>/dev/null || { echo "serve $T died"; break; }; sleep 10; done
  curl -sf localhost:8000/v1/models >/dev/null || { echo "SKIP $T: server not up" >> $O/failed.txt; continue; }
  date +%s > $O/t_${T}_start
  python3 -u vlm_pages.py --pages /root/pages2/pages.json --out $O/vlm_$T.json --model $NAME --concurrency 48 > $O/vlm_$T.log 2>&1
  date +%s > $O/t_${T}_end
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/vram_$T.txt
  kill $SV; wait $SV 2>/dev/null; sleep 5
done
python3 vlm_compare.py --ref $O/vlm_8b.json $O/vlm_2b.json $O/vlm_4b.json --out $O > $O/compare.log 2>&1
deactivate; . /root/venv/bin/activate
for T in 2b 4b 8b; do
  python3 -u embed_bench.py --sample $O/embed_pages_$T.json --out $O/embed_$T.json --query-field abbyy --batch 32 > $O/embed_$T.log 2>&1
done
echo DONE > $O/DONE
