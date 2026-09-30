#!/bin/bash
# Quant candidates vs prod, greedy speed (gbench.py) + perplexity. Run ONLY as a GPU turn:
#   gpu run bench -- bash /srv/llm/run/quant_ab.sh
# Prod layout/flags: 4 slots, unified 128k, q8 KV, MTP n3 + ngram-mod, --no-cache-idle-slots, model-card sampling.
set -u
d=/sys/class/drm/card0/device
IMG=sayso/llama.cpp:308883b-vulkan
cleanup() { docker rm -f sb sbppl >/dev/null 2>&1; }
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
DOCK="--device /dev/dri --group-add 44 --group-add 992 -e VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json -e GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 -v /srv/llm/data/llama.cpp/models:/models:ro -v /srv/llm/data/llama.cpp/cache:/cache:ro -v /srv/llm/run:/run_dir:ro"
FLAGS="--spec-type draft-mtp,ngram-mod --spec-draft-n-max 3 -ngl 99 --spec-draft-ngl 99 --fit off -fa on -ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -c 131072 -np -1 --kv-unified --no-cache-idle-slots --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0"
run() {
  local label=$1 model=$2 draft=$3
  cleanup
  docker run -d --name sb --network host $DOCK $IMG --host 127.0.0.1 --port 8001 -m "$model" --model-draft "$draft" $FLAGS >/dev/null
  for _ in $(seq 1 150); do
    curl -sf localhost:8001/health >/dev/null && break
    docker ps -q -f name=^sb$ | grep -q . || break
    sleep 2
  done
  if ! curl -sf localhost:8001/health >/dev/null; then
    echo "== $label FAILED: $(docker logs sb 2>&1 | grep -iE ' E |error' | tail -1 | cut -c1-200)"; cleanup; return
  fi
  echo "== $label  vram=$(( $(cat $d/mem_info_vram_used) / 1048576 ))MiB gtt=$(( $(cat $d/mem_info_gtt_used) / 1048576 ))MiB"
  python3 /srv/llm/run/gbench.py "$label" 2>&1
  cleanup
}
ppl() {
  local label=$1 model=$2
  docker run --rm --name sbppl $DOCK --entrypoint /app/llama $IMG perplexity -m "$model" -f /run_dir/ppl_corpus.txt \
    -ngl 99 -fa on -c 2048 --chunks 30 2>&1 | grep -E "Final estimate" | sed "s/^/[$label] PPL /"
}
[ -s /srv/llm/run/ppl_corpus.txt ] || { git -C /srv/llm/sayso ls-files '*.py' | head -80 | sed 's#^#/srv/llm/sayso/#' | xargs cat > /srv/llm/run/ppl_corpus.txt; }
echo "== start $(date +%T)  corpus $(wc -c < /srv/llm/run/ppl_corpus.txt) bytes"
M=/models/Qwen3.8-27B-Q4_K_M-attnQ6.gguf
GD=/models/mtp-Qwen3.8-27B-Q4_0.gguf
U=/cache/unsloth
run "A prod attnQ6 + ggml MTP"          $M                                 $GD
run "B UD-Q4_K_M + ggml MTP"            $U/Qwen3.8-27B-UD-Q4_K_M.gguf     $GD
run "C UD-IQ4_XS + ggml MTP"            $U/Qwen3.8-27B-UD-IQ4_XS.gguf     $GD
run "D attnQ6 + unsloth MTP"            $M                                 $U/unsloth-mtp-Qwen3.8-27B-Q4_0.gguf
ppl "A attnQ6"     $M
ppl "B UD-Q4_K_M"  $U/Qwen3.8-27B-UD-Q4_K_M.gguf
ppl "C UD-IQ4_XS"  $U/Qwen3.8-27B-UD-IQ4_XS.gguf
echo "== done $(date +%T)"
