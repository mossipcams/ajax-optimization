#!/bin/bash
# Per-op Vulkan GPU timings during a 2-slot (PAIR) decode at ~30k depth each. Run inside a gpu turn.
set -u
IMG=${1:-sayso/llama.cpp:d280808-fa2}
eval "$(grep -E "^(DOCK|FLAGS)=" /srv/llm/run/fa2rebase.sh)"
docker rm -f fb >/dev/null 2>&1
docker run -d --name fb --network host $DOCK -e GGML_VK_PERF_LOGGER=1 -e GGML_VK_PERF_LOGGER_FREQUENCY=25 $IMG --host 127.0.0.1 --port 8001 $FLAGS >/dev/null
for _ in $(seq 1 150); do curl -sf localhost:8001/health >/dev/null && break; sleep 2; done
PAIR_ONLY=1 python3 /srv/llm/run/knobbench.py "prof $IMG"
docker logs fb > /srv/llm/run/pairprof.log 2>&1
docker rm -f fb >/dev/null
