cd /srv/llm/run
E=""; for a in "$@"; do E="$E -e $a"; done
nohup /srv/llm/bin/gpu run fatest -- docker exec -e FA_ONLY=1 $E kd /app/build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b Vulkan0 > fatest.out 2>&1 &
sleep 3
while pgrep -f "gpu run fatest" >/dev/null; do sleep 3; done
echo "OK=$(grep -c 'OK' fatest.out) ERR=$(grep -c 'ERR =' fatest.out)"
grep "ERR =" fatest.out | sed -E "s/.*kv=([0-9]+),nb=([0-9]+).*type_K=([a-z0-9]+).*/kv=\1 nb=\2 \3/" | sort | uniq -c | head -20
