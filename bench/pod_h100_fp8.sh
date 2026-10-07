#!/bin/bash
# Follow-up on the H100 pod: waits for pod_h100.sh to finish, then re-runs the FP8 readers with DeepGEMM disabled
# (the image's NVCC < 12.9 makes DeepGEMM's JIT assert on Hopper). Fallback: full-precision Qwen3-VL-8B baseline.
set -x
O=/root/out; cd /root/bench
export HF_HOME=/root/hf VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
while [ ! -f $O/DONE ]; do sleep 20; done
mv $O/DONE $O/DONE_pass1
. /root/vvenv/bin/activate
run() {  # tag model max_pixels extra_serve_args extra_client_args
  local T=$1 NAME=$2 MP=$3 SARGS=$4 CARGS=$5
  [ -f $O/serve_$T.log ] && mv $O/serve_$T.log $O/serve_$T.fail_deepgemm.log
  vllm serve $NAME --port 8000 --max-model-len 12288 --gpu-memory-utilization 0.92 --max-num-seqs 160 \
     --limit-mm-per-prompt '{"image":1}' $SARGS > $O/serve_$T.log 2>&1 &
  local SV=$!
  for i in $(seq 1 150); do curl -sf localhost:8000/v1/models >/dev/null && break; kill -0 $SV 2>/dev/null || break; sleep 10; done
  if ! curl -sf localhost:8000/v1/models >/dev/null; then echo "SKIP2 $T (server did not start, DeepGEMM off)" >> $O/failed.txt; kill $SV 2>/dev/null; wait $SV 2>/dev/null; return 1; fi
  date +%s > $O/t_${T}_start
  python3 -u vlm_pages.py --pages /root/pages/pages.json --out $O/vlm_$T.json --model $NAME --concurrency 160 \
     --max-tokens 4000 --max-pixels $MP $CARGS > $O/vlm_$T.log 2>&1
  date +%s > $O/t_${T}_end
  nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O/vram_$T.txt
  kill $SV; wait $SV 2>/dev/null; sleep 5
}
if run q3vl8b_fp8 Qwen/Qwen3-VL-8B-Instruct-FP8 0 "" ""; then
  run q3vl8b_fp8_1mp Qwen/Qwen3-VL-8B-Instruct-FP8 1048576 "" ""
  REF=q3vl8b_fp8
else
  run q3vl8b_bf16     Qwen/Qwen3-VL-8B-Instruct 0       "" ""
  run q3vl8b_bf16_1mp Qwen/Qwen3-VL-8B-Instruct 1048576 "" ""
  REF=q3vl8b_bf16
fi
run q36_35b_a3b Qwen/Qwen3.6-35B-A3B-FP8 0 "" "--no-think"
python3 vlm_compare.py --ref $O/vlm_$REF.json $(ls $O/vlm_*.json | grep -v -e "vlm_$REF.json" -e vlm_compare) --out $O > $O/compare.log 2>&1
deactivate; . /root/venv/bin/activate
for F in $O/embed_pages_*.json; do
  T=$(basename $F .json | sed 's/embed_pages_//')
  python3 -u embed_bench.py --sample $F --out $O/embed_$T.json --query-field abbyy --batch 64 > $O/embed_$T.log 2>&1
done
echo DONE > $O/DONE
