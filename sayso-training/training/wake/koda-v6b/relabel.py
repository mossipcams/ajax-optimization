# v2 labels: Claude-judged addressedness of each room fired wake (by index in utc order, see /tmp dump),
# repeat-after-miss for near-threshold room windows, trusted sources unchanged.
import json, re, datetime as dt, collections
W="/srv/llm/data/wake/runs/fp-acoustics-20261016"
L=json.load(open(W+"/spool_labels.json")); byid={r["id"]:r for r in L}
rows=sorted([r for r in byid.values() if r["fired"] and r["koda_era"] and not r["tv_session"]], key=lambda r:r["utc"])
A=set(); N=set(); U=set()
def rng(a,b): return list(range(a,b+1))
A|=set(rng(1,9))|{15,20,21,22,24,25,26,28,29,33,34}|set(rng(36,46))|set(rng(54,65))|{75}|set(rng(77,87))|{89,90,91,108,120,122}|set(rng(124,141))|set(rng(143,146))|set(rng(148,158))|{2}
U|={12,16,19,23,27,31,48,50,52,66,73,74,76,88,111,112,118,121,142,160}
N|=set(range(len(rows)))-A-U
assert len(rows)==165 and not (A&U) and not (A&N) and not (U&N)
judged={rows[i]["id"]:("A" if i in A else "U" if i in U else "N") for i in range(len(rows))}
H=json.load(open("/srv/llm/data/wake/runs/koda-v2-20261005/real_pos_final.json")); HUMAN=set(H["train_ids"])|set(H["gate_ids"])
KW=re.compile(r"\b(koda|coda|kota|cota|kodah|codah|kda)\b",re.I)
t=lambda r: dt.datetime.strptime(r["utc"],"%Y%m%dT%H%M%SZ").timestamp()
wakesA=sorted(t(byid[i]) for i,v in judged.items() if v=="A")
import bisect
out=[]; src=collections.Counter()
for r in L:
    i=r["id"]
    if i in HUMAN: y,s=1,"human"
    elif r["tv_session"] or not r["koda_era"]: y,s=0,"trusted-neg"
    elif i in judged: y,s={"A":(1,"judged-wake"),"N":(0,"judged-false-wake"),"U":(None,"judged-unknown")}[judged[i]]
    elif KW.search(r["window_text"] or ""): y,s=1,"koda-in-window"
    else:
        k=bisect.bisect_right(wakesA,t(r)); nxt=wakesA[k] if k<len(wakesA) else None
        if nxt is not None and 0<nxt-t(r)<=10: y,s=1,"repeat-after-miss"
        else: y,s=0,"room-unfired"
    out.append(dict(id=i,y=y,source=s)); src[(s,y)]+=1
json.dump(out,open(W+"/spool_labels_v2.json","w"),indent=0)
for k,v in sorted(src.items(),key=lambda x:str(x)): print("  %-20s y=%s  %d"%(k[0],k[1],v))
print("positives:",sum(1 for o in out if o["y"]==1),"negatives:",sum(1 for o in out if o["y"]==0),"unknown:",sum(1 for o in out if o["y"] is None))
