#!/usr/bin/env python3
"""Single + 2-concurrent decode at ~30k depth on :8001, with acceptance-independent cycle time.
usage: knobbench.py <label>"""
import json, statistics as st, subprocess, sys, threading, time, urllib.request
URL, DEPTH, REPO = "http://127.0.0.1:8001", 30000, "/srv/llm/sayso"
BUSY = "/sys/class/drm/card0/device/gpu_busy_percent"
S = dict(temperature=0.6, top_k=20, top_p=0.95, min_p=0, cache_prompt=True)
label = sys.argv[1] if len(sys.argv) > 1 else "?"
def post(path, body):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=1800))
files = subprocess.run(["git", "-C", REPO, "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
def prompt(skip):
    text = ""
    for f in files[skip:]:
        text += f"\n### {f}\n" + open(f"{REPO}/{f}", errors="ignore").read()
        if len(text) > 170_000: break
    n = len(post("/tokenize", {"content": text})["tokens"]); text = text[: len(text) * DEPTH // n]
    return ("<|im_start|>user\n" + text + "\n\nReview the code above and find the three most serious bugs; "
            "explain each.<|im_end|>\n<|im_start|>assistant\n<think>\n")
prompts = [prompt(0), prompt(120)]
for slot, p in enumerate(prompts): post("/completion", dict(S, prompt=p, id_slot=slot, n_predict=1))
def run(slots, seeds):
    samples, stop, out = [], threading.Event(), []
    def sampler():
        while not stop.is_set(): samples.append(int(open(BUSY).read())); time.sleep(0.05)
    def gen(slot, seed):
        out.append(post("/completion", dict(S, prompt=prompts[slot], id_slot=slot, seed=seed, n_predict=300))["timings"])
    sm = threading.Thread(target=sampler); sm.start()
    th = [threading.Thread(target=gen, args=(s, sd)) for s, sd in zip(slots, seeds)]
    [t.start() for t in th]; [t.join() for t in th]; stop.set(); sm.join()
    return out, st.mean(samples)
def cyc(t):  # ms per speculative cycle = decode time / (tokens - accepted drafts)
    return t["predicted_ms"] / max(1, t["predicted_n"] - (t.get("draft_n_accepted") or 0))
import os
solo = [] if os.environ.get("PAIR_ONLY") else [run([0], [s])[0][0] for s in (11, 12, 13, 14)]
if solo:
  tps = [t["predicted_per_second"] for t in solo]; acc = [t["draft_n_accepted"] / max(1, t["draft_n"]) for t in solo]
  print(f"[{label}] SOLO  tok/s mean {st.mean(tps):5.1f} (runs {[round(x, 1) for x in tps]})  accept {st.mean(acc):.2f}  "
      f"cycle {st.mean(cyc(t) for t in solo):5.1f} ms", flush=True)
for sd in ((21, 31), (22, 32)):
    out, busy = run([0, 1], sd)
    print(f"[{label}] PAIR  combined {sum(t['predicted_per_second'] for t in out):5.1f} t/s  per-stream "
          f"{[round(t['predicted_per_second'], 1) for t in out]}  cache_n {[t.get('cache_n') for t in out]}  "
          f"cycle {st.mean(cyc(t) for t in out):5.1f} ms  GPU {busy:.0f}%", flush=True)
