import json,sys,time,urllib.request
# Coding workloads at Qwen coding sampling; reports decode t/s and draft acceptance per task.
src="".join(open("/srv/llm/sayso/scripts/llm-host/benchmark_inference.py").readlines()[:140])
S=dict(temperature=0.6,top_k=20,top_p=0.95,min_p=0,seed=42,cache_prompt=False)
T=[("gen","Write a Python module that parses nginx access log lines into dataclasses, aggregates status codes per hour, and includes pytest unit tests.",False,500),
   ("edit","Here is a file:\n```python\n"+src+"\n```\nRename every occurrence of the variable `args` to `cli_args` and return the COMPLETE updated file in one code block, nothing else.",False,1800),
   ("think","Implement an LRU cache with TTL expiry in Rust, with tests. Explain your design briefly first.",True,700)]
def chat(u,think): return "<|im_start|>user\n"+u+"<|im_end|>\n<|im_start|>assistant\n"+("<think>\n" if think else "<think>\n\n</think>\n\n")
for name,u,think,n in T:
    body=dict(S,prompt=chat(u,think),n_predict=n)
    t=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8001/completion",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900))["timings"]
    dn,da=t.get("draft_n") or 0,t.get("draft_n_accepted") or 0
    print("   %-5s %6.1f t/s  n=%4d  accept %s" % (name,t["predicted_per_second"],t["predicted_n"],"%d/%d=%.2f"%(da,dn,da/dn) if dn else "-"),flush=True)
