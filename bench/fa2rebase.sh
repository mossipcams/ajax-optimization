#!/bin/bash
# A/B: dev-fa2 (308883b+FA2) vs d280808-fa2 (d280808 upstream + same FA2 patch).
# Run as: gpu run bench -- bash /srv/llm/run/fa2verify.sh
set -u
d=/sys/class/drm/card0/device
cleanup() { docker rm -f fb >/dev/null 2>&1; }
trap cleanup EXIT; trap 'exit 1' HUP INT TERM
DOCK="--device /dev/dri --group-add 44 --group-add 992 -e VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json -e GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 -v /srv/llm/data/llama.cpp/models:/models:ro -v /srv/llm/data/llama.cpp/cache:/cache:ro"
FLAGS="-m /models/Qwen3.8-27B-UD-Q4_K_M.gguf --model-draft /models/mtp-M2-L8-O4.gguf --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 -ngl 99 --spec-draft-ngl 99 --fit off -fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -c 131072 -np -1 --kv-unified --no-cache-idle-slots --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0"
run() {
  local label=$1 img=$2
  cleanup
  docker run -d --name fb --network host $DOCK $img --host 127.0.0.1 --port 8001 $FLAGS >/dev/null
  for _ in $(seq 1 150); do curl -sf localhost:8001/health >/dev/null && break; docker ps -q -f name=^fb$ | grep -q . || break; sleep 2; done
  if ! curl -sf localhost:8001/health >/dev/null; then echo "== $label FAILED: $(docker logs fb 2>&1 | grep -iE ' E |error' | tail -1 | cut -c1-200)"; cleanup; return; fi
  echo "== $label  vram=$(( $(cat $d/mem_info_vram_used) / 1048576 ))MiB gtt=$(( $(cat $d/mem_info_gtt_used) / 1048576 ))MiB"
  python3 /srv/llm/run/knobbench.py "$label" 2>&1
  cleanup
}
echo "== start $(date +%T)"
run "dev-fa2" sayso/llama.cpp:dev-fa2
run "d280808-fa2" sayso/llama.cpp:d280808-fa2
run "dev-fa2" sayso/llama.cpp:dev-fa2
run "d280808-fa2" sayso/llama.cpp:d280808-fa2
echo "== done $(date +%T)"
