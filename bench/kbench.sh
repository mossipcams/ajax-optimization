#!/bin/bash
# Kernel-level speed (no speculation): tg64 at depth 0/30k and pp4 (MTP verify batch) at 30k. Run as: gpu run bench -- bash ...
set -u
IMG=sayso/llama.cpp:308883b-vulkan
cleanup() { docker rm -f kb >/dev/null 2>&1; }
trap cleanup EXIT; trap 'exit 1' HUP INT TERM
DOCK="--device /dev/dri --group-add 44 --group-add 992 -e VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json -e GGML_VK_SUBALLOCATION_BLOCK_SIZE=4294967296 -v /srv/llm/data/llama.cpp/models:/models:ro -v /srv/llm/data/llama.cpp/cache:/cache:ro"
b() { docker run --rm --name kb $DOCK --entrypoint /app/llama $IMG bench -m "$2" -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -r 3 "${@:3}" 2>&1 | grep -E "^\| qwen" | awk -F'|' -v l="$1" '{gsub(/ +/," "); print "[" l "]" $(NF-2) "|" $(NF-1)}'; }
echo "== start $(date +%T)"
for m in "A attnQ6|/models/Qwen3.8-27B-Q4_K_M-attnQ6.gguf" "B UD-Q4_K_M|/cache/unsloth/Qwen3.8-27B-UD-Q4_K_M.gguf" "C UD-IQ4_XS|/cache/unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf"; do
  l=${m%%|*}; p=${m#*|}
  b "$l" "$p" -p 0 -n 64 -d 0,30000
  b "$l" "$p" -p 4,8 -n 0 -d 30000
done
echo "== done $(date +%T)"
