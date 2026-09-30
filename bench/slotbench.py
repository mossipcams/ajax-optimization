#!/usr/bin/env python3
"""1 vs 2 concurrent streams at ~30k depth against a test server on :8001.

usage: slotbench.py <label>
Prompts are real SaySo source (~30k tokens each), cached on slots 0 and 1 first.
Sampling matches Pi (temp 0.6, top_p 0.95, top_k 20).
"""
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
        if len(text) > 170_000:
            break
    n = len(post("/tokenize", {"content": text})["tokens"])
    text = text[: len(text) * DEPTH // n]
    return ("<|im_start|>user\n" + text + "\n\nReview the code above and find the three most serious bugs; "
            "explain each.<|im_end|>\n<|im_start|>assistant\n<think>\n")


prompts = [prompt(0), prompt(120)]
t0 = time.time()
for slot, p in enumerate(prompts):
    post("/completion", dict(S, prompt=p, id_slot=slot, n_predict=1))
print(f"[{label}] cached 2 x ~{DEPTH // 1000}k prompts in {time.time() - t0:.0f}s", flush=True)


def run(slots, seeds):
    samples, stop, out = [], threading.Event(), []

    def sampler():
        while not stop.is_set():
            samples.append(int(open(BUSY).read()))
            time.sleep(0.05)

    def gen(slot, seed):
        out.append(post("/completion", dict(S, prompt=prompts[slot], id_slot=slot, seed=seed, n_predict=300))["timings"])

    sm = threading.Thread(target=sampler)
    sm.start()
    th = [threading.Thread(target=gen, args=(s, seed)) for s, seed in zip(slots, seeds)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    stop.set()
    sm.join()
    tps = [round(t["predicted_per_second"], 1) for t in out]
    acc = sum(t.get("draft_n_accepted") or 0 for t in out) / max(1, sum(t.get("draft_n") or 0 for t in out))
    print(f"[{label}] streams={len(slots)} per-stream {tps} total {sum(tps):.1f} t/s  accept {acc:.2f}  "
          f"GPU busy {st.mean(samples):.0f}%", flush=True)


for rnd in (1, 2):
    run([0], [10 + rnd])
    run([0, 1], [20 + rnd, 30 + rnd])
