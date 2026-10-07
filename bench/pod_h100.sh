#!/bin/bash
# Runs ON a single-H100 pod (tmux). Sample 2 (docs never read before): baseline Qwen3-VL-8B FP8 at full res and at
# 1 MP, plus the two strongest new general vision-language candidates from MODEL_RESEARCH.md. OCR-only models are
# excluded by decision: the reader must see images, maps and other visual content on the page, not just transcribe. EmbeddingGemma 2 on every reader's output.
# Page images are tarred so they can be published and reused. Writes /root/out/.
set -x
O=/root/out; mkdir -p $O /root/pages; cd /root/bench
export HF_HOME=/root/hf PIP_DISABLE_PIP_VERSION_CHECK=1 VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $O/gpu.txt
python3 -m venv /root/venv && /root/venv/bin/pip install -q "sentence-transformers[image]>=6.1.0" "transformers>=5.19" pymupdf pillow > $O/pip_embed.log 2>&1 &
EMB=$!
python3 -m venv /root/vvenv && /root/vvenv/bin/pip install -q "vllm==0.31.0" pymupdf pillow > $O/pip_vllm.log 2>&1
. /root/vvenv/bin/activate
python3 -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)" > $O/versions.txt 2>&1
python3 -u fetch_pages.py --candidates bench_candidates_s2.json --out-dir /root/pages --max-pages 5 --workers 12 > $O/fetch.log 2>&1
cp /root/pages/pages.json $O/pages.json
( cd /root && tar cf $O/page_images.tar pages ) &
run() {  # tag model max_pixels extra_serve_args extra_client_args
  local T=$1 NAME=$2 MP=$3 SARGS=$4 CARGS=$5
  vllm serve $NAME --port 8000 --max-model-len 12288 --gpu-memory-utilization 0.92 --max-num-seqs 160 \
     --limit-mm-per-prompt '{"image":1}' $SARGS > $O/serve_$T.log 2>&1 &
  local SV=$!
  for i in $(seq 1 150); do curl -sf localhost:8000/v1/models >/dev/null && break; kill -0 $SV 2>/dev/null || break; sleep 10; done
  if ! curl -sf localhost:8000/v1/models >/dev/null; then echo "SKIP $T (server did not start)" >> $O/failed.txt; kill $SV 2>/dev/null; return; fi
  date +%s > $O/t_${T}_start
  python3 -u vlm_pages.py --pages /root/pages/pages.json --out $O/vlm_$T.json --model $NAME --concurrency 160 \
     --max-tokens 4000 --max-pixels $MP $CARGS > $O/vlm_$T.log 2>&1
  date +%s > $O/t_${T}_end
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/vram_$T.txt
  kill $SV; wait $SV 2>/dev/null; sleep 5
}
run q3vl8b_fp8      Qwen/Qwen3-VL-8B-Instruct-FP8 0       "" ""
run q3vl8b_fp8_1mp  Qwen/Qwen3-VL-8B-Instruct-FP8 1048576 "" ""
run q35_9b          Qwen/Qwen3.5-9B               0       "" "--no-think"
run q36_35b_a3b     Qwen/Qwen3.6-35B-A3B-FP8      0       "" "--no-think"
ls $O/vlm_*.json > /dev/null 2>&1 && python3 vlm_compare.py --ref $O/vlm_q3vl8b_fp8.json \
   $(ls $O/vlm_*.json | grep -v -e vlm_q3vl8b_fp8.json -e vlm_compare) --out $O > $O/compare.log 2>&1
deactivate; wait $EMB; . /root/venv/bin/activate
for F in $O/embed_pages_*.json; do
  T=$(basename $F .json | sed 's/embed_pages_//')
  python3 -u embed_bench.py --sample $F --out $O/embed_$T.json --query-field abbyy --batch 64 > $O/embed_$T.log 2>&1
done
wait
echo DONE > $O/DONE
