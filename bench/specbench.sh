#!/bin/bash
# Stops prod, benchmarks speculative configs on :8001 one at a time, ALWAYS restores prod (trap).
cleanup() { docker rm -f lt >/dev/null 2>&1; /srv/llm/bin/gpu serve llama >/dev/null 2>&1; echo "prod restored $(date +%T)"; }
trap cleanup EXIT; trap "exit 1" HUP INT TERM
K=$(docker exec llama-server printenv LLAMA_API_KEY)
until [ "$(curl -s localhost:8000/slots -H "Authorization: Bearer $K" | grep -o "\"is_processing\":true" | wc -l)" = 0 ]; do sleep 5; done
/srv/llm/bin/gpu stop >/dev/null 2>&1; echo "prod stopped $(date +%T)"
d=$(dirname $(ls /sys/class/drm/card*/device/mem_info_vram_used | head -1))
BASE="-m /models/Qwen3.8-27B-Q4_K_M-attnQ6.gguf -ngl 99 --fit off -fa on --jinja -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -c 131072 -np -1 --kv-unified --spec-draft-ngl 99"
MTP="--model-draft /models/mtp-Qwen3.8-27B-Q4_0.gguf"; DF="--model-draft /cache/custom/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"; DS="--model-draft /cache/custom/Qwen3.8-27B-DSpark-Q8_0.gguf"
run() { L=$1; shift; docker rm -f lt >/dev/null 2>&1
  docker run -d --name lt --network host --device /dev/dri --group-add 44 --group-add 992 -e VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json -e GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 -v /srv/llm/data/llama.cpp/models:/models:ro -v /srv/llm/data/llama.cpp/cache:/cache sayso/llama.cpp:308883b-vulkan --host 127.0.0.1 --port 8001 $BASE "$@" >/dev/null
  for i in $(seq 1 100); do curl -sf localhost:8001/health >/dev/null && break; docker ps -q -f name=^lt$ | grep -q . || break; sleep 3; done
  if ! curl -sf localhost:8001/health >/dev/null; then echo "== $L  FAILED: $(docker logs lt 2>&1 | grep -E " E |error|failed" | tail -2 | tr "\n" " " | cut -c1-250)"; return; fi
  echo "== $L  vram=$(( $(cat $d/mem_info_vram_used)/1048576 ))MiB gtt=$(( $(cat $d/mem_info_gtt_used)/1048576 ))MiB"
  python3 /srv/llm/run/specbench.py || echo "   bench error"
}
run mtp3             $MTP --spec-type draft-mtp --spec-draft-n-max 3
run mtp3+ngram       $MTP --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3
run dflash2-n7       $DF  --spec-type draft-dflash --spec-draft-n-max 7
run dflash2-n15      $DF  --spec-type draft-dflash --spec-draft-n-max 15
run dflash2-n7+ngram $DF  --spec-type draft-dflash,ngram-mod --spec-draft-n-max 7
run dspark-n7        $DS  --spec-type draft-dspark --spec-draft-n-max 7
run mtp3-bs          $MTP --spec-type draft-mtp --spec-draft-n-max 3 --backend-sampling
