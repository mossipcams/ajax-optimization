import json, subprocess, threading, time, urllib.request
KEY = ""
URL, REPO, DEPTH = "http://127.0.0.1:8001", "/srv/llm/sayso", 30000
def req(path, body=None, auth=False):
    h = {"Content-Type": "application/json"}
    if auth: h["Authorization"] = "Bearer " + KEY
    r = urllib.request.Request(URL + path, json.dumps(body).encode() if body else None, h)
    return json.load(urllib.request.urlopen(r, timeout=1800))
for _ in range(90):
    try:
        req("/health", auth=False); break
    except Exception: time.sleep(2)

files = subprocess.run(["git", "-C", REPO, "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
def prompt(skip):
    text = ""
    for f in files[skip:]:
        text += f"\n### {f}\n" + open(f"{REPO}/{f}", errors="ignore").read()
        if len(text) > 170_000: break
    n = len(req("/tokenize", {"content": text})["tokens"]); text = text[: len(text) * DEPTH // n]
    return "<|im_start|>user\n" + text + "\n\nReview the code above and find the three most serious bugs; explain each.<|im_end|>\n<|im_start|>assistant\n<think>\n"
S = dict(temperature=0.6, top_k=20, top_p=0.95, min_p=0, cache_prompt=True)
P = [prompt(0), prompt(120)]
for slot, p in enumerate(P): req("/completion", dict(S, prompt=p, id_slot=slot, n_predict=1))
import sys
LABEL = sys.argv[1] if len(sys.argv) > 1 else "?"
def pair(tag, pinned, seeds):
    tag = f"[{LABEL}] {tag}"
    out = [None, None]
    def gen(i):
        body = dict(S, prompt=P[i], seed=seeds[i], n_predict=300)
        if pinned: body["id_slot"] = i
        t0 = time.time(); r = req("/completion", body); out[i] = (r, time.time() - t0)
    th = [threading.Thread(target=gen, args=(i,)) for i in (0, 1)]; [t.start() for t in th]; [t.join() for t in th]
    for i, (r, wall) in enumerate(out):
        t = r["timings"]
        print(f"{tag} req{i}: slot={r.get('id_slot')} predicted_n={t['predicted_n']} predicted_ms={t['predicted_ms']:.0f} "
              f"tok/s={t['predicted_per_second']:.1f} prompt_n={t['prompt_n']} prompt_ms={t['prompt_ms']:.0f} cache_n={t.get('cache_n')} "
              f"draft={t.get('draft_n_accepted')}/{t.get('draft_n')} stop={r.get('stop_type')} wall={wall:.1f}s", flush=True)
pair("pinned#1  ", True, (21, 31))
pair("unpinned#1", False, (22, 32))
pair("pinned#2  ", True, (23, 33))
