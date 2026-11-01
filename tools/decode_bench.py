#!/usr/bin/env python3
"""Single-stream decode speed through a running server: 8 prompts (code, prose, math, structured output, a long quote),
fixed seeds, tokens per second after the first streamed token (prefill excluded), per prompt and the geometric mean.
Start a fresh server per run when comparing builds: the draft head learns as it serves.

usage: decode_bench.py [url] [extra request JSON] [tokens]
  decode_bench.py http://127.0.0.1:18023                       # the model's default sampling, thinking on
  decode_bench.py http://127.0.0.1:18023 '{"temperature": 0}'  # greedy"""
import json, math, os, sys, time, urllib.request

url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18023") + "/v1/chat/completions"
extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
n_tok = int(sys.argv[3]) if len(sys.argv) > 3 else 800
doc = open(os.path.join(os.path.dirname(__file__), "..", "src", "sessions.rs")).read()[:12000]
PROMPTS = {
    "code": "Write a complete Python implementation of a red-black tree with insert, delete and search, "
            "with docstrings.",
    "rust": "Implement an LRU cache in Rust with generics, a doubly linked list over a slab, and unit tests.",
    "prose": "Write a long, detailed essay about the history of the printing press and its effects on European "
             "society.",
    "story": "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "math": "Solve step by step: a train leaves at 3pm at 80 km/h, another at 4pm at 110 km/h from the same station in "
            "the same direction. When and where does the second catch the first? Then generalize.",
    "explain": "Explain how TCP congestion control works, including slow start, congestion avoidance, fast retransmit "
               "and fast recovery.",
    "json": "Generate a JSON array of 15 fictional employees with fields id, name, email, department, salary, "
            "start_date and skills.",
    "quote": "Here is a Rust file:\n\n" + doc + "\n\nRepeat the file above verbatim, then explain it briefly.",
}


def run(prompt, seed):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": n_tok, "stream": True, "ignore_eos": True,
            "seed": seed, "stream_options": {"include_usage": True}, **extra}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    first, usage = None, None
    for line in urllib.request.urlopen(req, timeout=900):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("usage"):
            usage = chunk["usage"]
        if chunk.get("choices") and first is None:
            first = time.time()
    return (usage["completion_tokens"] - 1) / (time.time() - first)


run("hi", 0)  # warm-up
res = {name: run(p, i + 1) for i, (name, p) in enumerate(PROMPTS.items())}
geo = math.exp(sum(map(math.log, res.values())) / len(res))
print(" | ".join(f"{k} {v:.1f}" for k, v in res.items()), f"| geomean {geo:.1f} tok/s")
