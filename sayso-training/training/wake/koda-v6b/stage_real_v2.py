# v6b real-data staging from spool_labels_v2.json (judged wakes + repeat-after-miss).
# Locked test sessions are reused verbatim from data/splits.json (never reselected).
import json, re, sys, numpy as np, soundfile as sf
from pathlib import Path
from livekit.wakeword.eval import stream as S
R = Path("/srv/llm/data/wake/runs/koda-phonics-v6-20261018"); W = Path("/srv/llm/data/wake/runs/fp-acoustics-20261016")
OUT = R / "output" / sys.argv[1]; HD = R / "staging" / f"hard_real_{sys.argv[1]}"
SP = W / "spool/wake-mining/records"; SR = 16000; WIN = 32000; PRE = 8000
L = {r["id"]: r for r in json.load(open(W / "spool_labels.json"))}
V2 = {o["id"]: o for o in json.load(open(W / "spool_labels_v2.json"))}
CONFIRMED_FP = {"73ae828a", "4f441174", "64c7369f", "da503688", "9423ec5c"}
KODA_PHONES = re.compile(r"\b(k|kh|t|th|tʃ)\s+(o\S*|ɔ\S*|ɑ\S*|aʊ)\s+(d|t|th|ɾ|r|l|n)\b")
splits = json.load(open(R / "data/splits.json")); test_g = set(splits["test_pool_groups"])
mel, emb = S.build_feature_pipeline()
v5 = S.build_onnx_scorer(Path("/srv/llm/data/wake/runs/koda-phonics-v5-20261010/output/koda-phonics-v5/koda-phonics-v5.onnx"))
score = lambda x: float(v5(S.window_features(mel, emb, x.astype(np.float32))))
def context(rid):
    d = SP / rid; w = sf.read(d / "window.wav", dtype="float32")[0]
    pre = sf.read(d / "pre.wav", dtype="float32")[0] if (d / "pre.wav").exists() else np.zeros(0, np.float32)
    post = sf.read(d / "post.wav", dtype="float32")[0] if (d / "post.wav").exists() else np.zeros(0, np.float32)
    pre = np.pad(pre, (PRE - len(pre), 0)) if len(pre) < PRE else pre[-PRE:]
    post = np.pad(post, (0, PRE - len(post))) if len(post) < PRE else post[:PRE]
    return np.concatenate([pre, w, post])
rows = []
for i, o in V2.items():
    r = L[i]
    if o["y"] is None or not (SP / i / "window.wav").exists(): continue
    r["y"], r["src"], r["group"] = o["y"], o["source"], "spool:" + (r.get("session") or r["utc"][:8]); rows.append(r)
new = {r["group"] for r in rows} - set(splits["groups"])
for g in new: splits["groups"][g] = "train"               # unseen groups train; locked test untouched
json.dump(splits, open(R / "data/splits.json", "w"), indent=2, sort_keys=True)
pos = [r for r in rows if r["y"] == 1]; neg = [r for r in rows if r["y"] == 0]
print("labelled", len(rows), "pos", len(pos), "neg", len(neg), "| locked test groups", len(test_g),
      "test pos", sum(r["group"] in test_g for r in pos), "test room negs", sum(r["group"] in test_g and r["koda_era"] and not r["tv_session"] for r in neg))
out = OUT / "real_positive_train"; out.mkdir(parents=True, exist_ok=True)
for f in out.glob("clip_*.wav"): f.unlink()
man = []; k = 0
for r in pos:
    if r["group"] in test_g: continue
    c = context(r["id"]); missed = r["src"] in ("repeat-after-miss", "koda-in-window") or score(c[PRE:PRE + WIN]) < 0.20
    variants = [(s, 1.0) for s in ((-0.16, -0.08, 0.0, 0.08, 0.16) if missed else (-0.08, 0.0, 0.08))] + ([(0.0, 2.0), (0.0, 0.5)] if missed else [])
    for s, gain in variants:
        st = PRE + int(round(s * SR)); x = np.clip(c[st:st + WIN] * gain, -1, 1).astype(np.float32)
        p = out / f"clip_{k:06d}.wav"; sf.write(p, x, SR, subtype="PCM_16"); k += 1
        man.append(dict(clip=p.name, capture_id=r["id"], source=r["src"], shift_s=s, gain=gain, missed=bool(missed)))
print("real_positive_train", k, "clips from", len({m["capture_id"] for m in man}), "takes; missed", len({m["capture_id"] for m in man if m["missed"]}))
HD.mkdir(parents=True, exist_ok=True)
for f in HD.glob("*"): f.unlink()
hits = []; skipped = 0
for r in neg:
    if r["group"] in test_g or r["tv_session"] or not r["koda_era"]: continue
    i = r["id"][:8]; strong = i in CONFIRMED_FP or r["src"] == "judged-false-wake"
    c = context(r["id"]); s5 = score(c[PRE:PRE + WIN])
    if not strong and s5 >= 0.10 and KODA_PHONES.search(r["phones"] or ""): skipped += 1; continue   # unjudged + Koda-shaped + scored: too risky
    f = HD / f"{r['id']}.wav"; sf.write(f, c, SR, subtype="PCM_16")
    for s in ((-0.16, -0.08, 0.0, 0.08, 0.16) if strong else (0.0,)):
        hits.append(dict(file=f.name, timestamp_seconds=round(1.5 + s, 3), score=s5, group_id=r["group"]))
with open(HD / "hits.jsonl", "w") as fh:
    for h in hits: fh.write(json.dumps(h) + "\n")
print("hard negative hits", len(hits), "from", len({h["file"] for h in hits}), "room windows; skipped risky", skipped)
json.dump(dict(real_positive=man, hits=len(hits)), open(R / f"staging/real_manifest_{sys.argv[1]}.json", "w"), indent=1)
