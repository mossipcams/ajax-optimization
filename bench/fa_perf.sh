# usage: fa.sh [ENV=val ...]   runs FA perf inside a gpu turn
cd /srv/llm/run
E=""; for a in "$@"; do E="$E -e $a"; done
nohup /srv/llm/bin/gpu run fabench -- docker exec -e FA_ONLY=1 $E kd /app/build/bin/test-backend-ops perf -o FLASH_ATTN_EXT > fabench.out 2>&1 &
sleep 3
until grep -q -E "Backend .* (OK|FAIL)|^\s*[0-9]+/[0-9]+ tests|Failed" fabench.out || ! pgrep -f "gpu run fabench" >/dev/null; do sleep 4; done
grep -E "FLASH_ATTN_EXT|us/run" fabench.out | sed -E 's/.*hsk=256,hsv=256,nh=4,nr23=\[6,1\],kv=([0-9]+),nb=([0-9]+).*: +([0-9.]+) us\/run.*/kv=\1 nb=\2 \3us/' | head -12
