# v5 vs v6 vs v6b on locked never-trained sessions, v2 labels. Window scores; bootstrap over clips.
import json, numpy as np, soundfile as sf
from pathlib import Path
from sklearn.metrics import roc_auc_score
from livekit.wakeword.eval import stream as S
R=Path("/srv/llm/data/wake/runs/koda-phonics-v6-20261018"); W=Path("/srv/llm/data/wake/runs/fp-acoustics-20261016"); SP=W/"spool/wake-mining/records"
L={r["id"]:r for r in json.load(open(W/"spool_labels.json"))}; V2=json.load(open(W/"spool_labels_v2.json"))
test=set(json.load(open(R/"data/splits.json"))["test_pool_groups"])
rows=[(o,L[o["id"]]) for o in V2 if o["y"] is not None and ("spool:"+(L[o["id"]].get("session") or L[o["id"]]["utc"][:8])) in test]
rows=[(o,r) for o,r in rows if o["y"]==1 or (r["koda_era"] and not r["tv_session"])]   # Koda vs room negatives
y=np.array([o["y"] for o,_ in rows]); src=np.array([o["source"] for o,_ in rows])
M={"v5":"/srv/llm/data/wake/runs/koda-phonics-v5-20261010/output/koda-phonics-v5/koda-phonics-v5.onnx",
   "v6":str(R/"output/koda-phonics-v6/koda-phonics-v6.onnx"),"v6b":str(R/"output/koda-phonics-v6b/koda-phonics-v6b.onnx")}
mel,emb=S.build_feature_pipeline(); sc={k:S.build_onnx_scorer(Path(v)) for k,v in M.items()}
F=[S.window_features(mel,emb,sf.read(SP/r["id"]/"window.wav",dtype="float32")[0]) for _,r in rows]
s={k:np.array([float(m(f)) for f in F]) for k,m in sc.items()}
print("locked test: %d real Koda (%d misses by repeat/kw) vs %d room negatives"%((y==1).sum(),np.isin(src,["repeat-after-miss","koda-in-window"]).sum(),(y==0).sum()))
def caught(v,yy,k):
    rn=np.sort(v[yy==0])[::-1]; thr=rn[k] if k<len(rn) else -1; return int(((v>thr)&(yy==1)).sum()), thr
np.savez(R/"staging/eval_v2_scores.npz",y=y,src=src,**s)
print("%-4s %-6s  %s"%("","AUC","Koda caught when 0 / 2 / 5 / 10 room negatives pass (threshold)"))
for k,v in s.items():
    print("%-4s %.3f   %s   | at live 0.20: caught %d, room negs passing %d"%(k,roc_auc_score(y,v),"  ".join("%d@%.2f"%caught(v,y,n) for n in (0,2,5,10)),int(((v>=.2)&(y==1)).sum()),int(((v>=.2)&(y==0)).sum())))
rng=np.random.default_rng(0); P=np.flatnonzero(y==1); N=np.flatnonzero(y==0)
for a,b in (("v6b","v5"),("v6b","v6"),("v6","v5")):
    D={n:[] for n in (2,5)}; A=[]
    for _ in range(3000):
        ii=np.concatenate([rng.choice(P,len(P)),rng.choice(N,len(N))]); yb=y[ii]
        A.append(roc_auc_score(yb,s[a][ii])-roc_auc_score(yb,s[b][ii]))
        for n in D: D[n].append(caught(s[a][ii],yb,n)[0]-caught(s[b][ii],yb,n)[0])
    f=lambda x: "%+.2f [%+.2f, %+.2f]"%(np.mean(x),*np.percentile(x,[2.5,97.5]))
    print("%s - %s: AUC %s | caught@2 %s | caught@5 %s  (of %d)"%(a,b,f(A),f(D[2]),f(D[5]),len(P)))
for k in ("v5","v6","v6b"):
    miss=np.isin(src,["repeat-after-miss","koda-in-window"])
    print("%s on the %d labelled misses: caught at live 0.20 = %d"%(k,miss.sum(),int((s[k][miss]>=.2).sum())))
