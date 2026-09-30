#!/usr/bin/env python3
"""Draft-head comparison on :8001: 10 prompts x 2 seeds x 300 tokens, non-thinking. usage: mtpbench.py <label> [greedy]"""
import json, sys, urllib.request
label = sys.argv[1]; greedy = len(sys.argv) > 2
SAMP = dict(temperature=0, top_k=1) if greedy else dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0)
P = ["write a python function that merges two sorted lists into one sorted list, with docstring.",
     "explain the difference between mmap and read for loading large files, one paragraph.",
     "write a bash script that watches a directory and prints new files as they appear.",
     "Explain step by step how a B-tree handles node splits during insertion, with an example.",
     "Walk through how TCP congestion control reacts to packet loss.",
     "Write a Rust function that parses an ISO 8601 date string without external crates.",
     "Summarize the causes and consequences of the 2008 financial crisis in five bullet points.",
     "Write a SQL query that finds the top 3 customers by revenue per month, with explanation.",
     "Describe how a transformer attention layer works to a new engineer.",
     "Write a Go HTTP server with graceful shutdown and a health endpoint."]
chat = lambda u: "<|im_start|>user\n" + u + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
def post(body):
    r = urllib.request.Request("http://127.0.0.1:8001/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))["timings"]
post(dict(SAMP, prompt=chat("hi"), n_predict=16))
ts = [post(dict(SAMP, prompt=chat(p), n_predict=300, seed=s, cache_prompt=False)) for p in P for s in (1, 2)]
n = sum(t["predicted_n"] for t in ts); acc = sum(t["draft_n_accepted"] for t in ts); dr = sum(t["draft_n"] for t in ts)
ms = sum(t["predicted_ms"] for t in ts)
print(f"[{label}{' greedy' if greedy else ''}] accept {acc / dr:.3f}  tokens/cycle {n / (n - acc):.2f}  cycle {ms / (n - acc):.1f} ms  tok/s {n / ms * 1000:.1f}", flush=True)
