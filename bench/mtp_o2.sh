#!/bin/bash
# Draft-head variants, MTP only (no ngram-mod), prod layout. Run as: gpu run bench -- bash /srv/llm/run/mtp_ab.sh
set -u
d=/sys/class/drm/card0/device
IMG=sayso/llama.cpp:d280808-fa2
cleanup() { docker rm -f sb >/dev/null 2>&1; }
trap cleanup EXIT; trap 'exit 1' HUP INT TERM
DOCK="--device /dev/dri --group-add 44 --group-add 992 -e VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json -e GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 -v /srv/llm/data/llama.cpp/models:/models:ro -v /srv/llm/data/llama.cpp/cache:/cache:ro"
FLAGS="-m /models/Qwen3.8-27B-UD-Q4_K_M.gguf --spec-type draft-mtp --spec-draft-n-max 3 -ngl 99 --spec-draft-ngl 99 --fit off -fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -c 131072 -np -1 --kv-unified --no-cache-idle-slots"
run() {
  local label=$1 draft=$2; shift 2
  cleanup
  docker run -d --name sb --network host $DOCK $IMG --host 127.0.0.1 --port 8001 --model-draft "$draft" $FLAGS >/dev/null
  for _ in $(seq 1 150); do curl -sf localhost:8001/health >/dev/null && break; docker ps -q -f name=^sb$ | grep -q . || break; sleep 2; done
  curl -sf localhost:8001/health >/dev/null || { echo "== $label FAILED: $(docker logs sb 2>&1 | grep -iE ' E |error' | tail -1 | cut -c1-200)"; cleanup; return; }
  echo "== $label  vram=$(( $(cat $d/mem_info_vram_used) / 1048576 ))MiB gtt=$(( $(cat $d/mem_info_gtt_used) / 1048576 ))MiB"
  for mode in "$@"; do python3 /srv/llm/run/mtpbench.py "$label" $mode 2>&1; done
  cleanup
}
echo "== start $(date +%T)"
run "O4" /cache/mtp/mtp-M2-L8-O4.gguf ""
run "O2" /cache/mtp/mtp-M2-L8-O2.gguf ""
run "O4" /cache/mtp/mtp-M2-L8-O4.gguf ""
run "O2" /cache/mtp/mtp-M2-L8-O2.gguf ""
echo "== done $(date +%T)"
