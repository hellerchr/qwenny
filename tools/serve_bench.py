#!/usr/bin/env python3
"""Decode speed through the server: 1, 4, 8 and 16 concurrent replies (400 tokens unless the request JSON says
otherwise) to Fjordhaus tickets.
usage: serve_bench.py <test.jsonl> [url] [extra request JSON]"""
import json, sys, threading, time, urllib.request

tests = [json.loads(l) for l in open(sys.argv[1])]
url = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:18023") + "/v1/chat/completions"
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}


def one(t, seed, out):
    msgs = [m for m in t["messages"] if m["role"] != "assistant"]
    body = {"messages": msgs, "max_tokens": 400, "ignore_eos": True, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}, **extra}
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out.append(json.load(urllib.request.urlopen(req, timeout=600))["usage"]["completion_tokens"])


def run(conc, first):
    out = []
    th = [threading.Thread(target=one, args=(tests[first + i], i, out)) for i in range(conc)]
    t0 = time.time()
    for x in th: x.start()
    for x in th: x.join()
    return sum(out) / (time.time() - t0)


run(1, 0)  # warm-up
print(" | ".join(f"{c} streams: {run(c, 1 + c):.1f} tok/s" for c in (1, 1, 4, 8, 16)))
