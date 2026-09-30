#!/bin/bash
# Multi-slot decode benchmark at ~30k depth. Run ONLY as a GPU turn:
#   gpu run bench -- bash /srv/llm/run/slotbench.sh
# The lock stops prod before this starts and restores it afterwards; this script only owns
# the test container `sb` on 127.0.0.1:8001 and always removes it.
set -u
d=/sys/class/drm/card0/device
cleanup() { docker rm -f sb >/dev/null 2>&1; }
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

BASE="-m /models/Qwen3.8-27B-Q4_K_M-attnQ6.gguf --model-draft /models/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 -ngl 99 --spec-draft-ngl 99 --fit off -fa on -ctkd q8_0 -ctvd q8_0 -c 131072"
run() {
  local label=$1; shift
  cleanup
  docker run -d --name sb --network host --device /dev/dri --group-add 44 --group-add 992 \
    -e VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json -e GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 \
    -v /srv/llm/data/llama.cpp/models:/models:ro -v /srv/llm/data/llama.cpp/cache:/cache:ro sayso/llama.cpp:308883b-vulkan \
    --host 127.0.0.1 --port 8001 $BASE "$@" >/dev/null
  for _ in $(seq 1 120); do
    curl -sf localhost:8001/health >/dev/null && break
    docker ps -q -f name=^sb$ | grep -q . || break
    sleep 2
  done
  if ! curl -sf localhost:8001/health >/dev/null; then
    echo "== $label FAILED: $(docker logs sb 2>&1 | grep -iE ' E |error' | tail -1 | cut -c1-200)"
    return
  fi
  echo "== $label  vram=$(( $(cat $d/mem_info_vram_used) / 1048576 ))MiB gtt=$(( $(cat $d/mem_info_gtt_used) / 1048576 ))MiB  ($(date +%T))"
  python3 /srv/llm/run/slotbench.py "$label" 2>&1
  cleanup
}
echo "== start $(date +%T)"
run "A prod: auto(4) unified q8/q8"   -np -1 --kv-unified -ctk q8_0 -ctv q8_0
run "B 2 slots unified q8/q8"         -np 2 --kv-unified -ctk q8_0 -ctv q8_0
run "C 2 slots split 64k q8/q8"       -np 2 --no-kv-unified -ctk q8_0 -ctv q8_0
run "D 2 slots split 64k q8/q4"       -np 2 --no-kv-unified -ctk q8_0 -ctv q4_0
echo "== done $(date +%T)"
