#!/usr/bin/env python3
"""Decode speed through a running server, on 8 prompts (code, prose, math, structured output, a long quote) with fixed
seeds. Start a fresh server per run when comparing builds: the draft head learns as it serves.

One stream (default): each prompt streamed, tokens per second from the first streamed token to the end of the reply
(prefill excluded; client wall clock, time.perf_counter), per prompt and the geometric mean.
--streams=1,4,8,16: concurrent replies, one run per count: all their tokens divided by the wall time from sending the
requests to the last reply (prefill included).

usage: decode_bench.py [url] [extra request JSON] [tokens] [--streams=N,N,...]
  decode_bench.py http://127.0.0.1:18023                       # the model's default sampling, thinking on
  decode_bench.py http://127.0.0.1:18023 '{"temperature": 0}'  # greedy
  decode_bench.py http://127.0.0.1:18023 '{"chat_template_kwargs": {"enable_thinking": false}}' 400 --streams=1,4,8,16"""
import argparse
import concurrent.futures
import json
import math
import os
import time

from qwenny_client import Client

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


def has_output(chunk):
    """The initial role-only SSE event is sent before prefill and is not a generated token."""
    return any(
        choice.get("delta", {}).get(key)
        for choice in chunk.get("choices", [])
        for key in ("content", "reasoning_content", "tool_calls")
    )


def request(prompt, seed, tokens, extra):
    return {"messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "ignore_eos": True, "seed": seed,
            **extra}


def stream_speed(client, prompt, seed, tokens, extra):
    first, usage = None, None
    for chunk in client.events(**request(prompt, seed, tokens, extra)):
        usage = chunk.get("usage") or usage
        if first is None and has_output(chunk):
            first = time.perf_counter()
    return (usage["completion_tokens"] - 1) / (time.perf_counter() - first)


def concurrent_speed(client, streams, tokens, extra):
    prompts = list(PROMPTS.values())
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(streams) as pool:
        replies = list(pool.map(lambda i: client.chat(**request(prompts[i % len(prompts)], i, tokens, extra)),
                                range(streams)))
    return sum(r["usage"]["completion_tokens"] for r in replies) / (time.perf_counter() - start)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", nargs="?", default="http://127.0.0.1:18023")
    p.add_argument("extra", nargs="?", default="{}", help="more fields of the request, as JSON")
    p.add_argument("tokens", nargs="?", type=int, default=800)
    p.add_argument("--streams", help="concurrent replies to measure, e.g. 1,4,8,16 (default: one stream per prompt)")
    a = p.parse_args()
    client, extra = Client(url=a.url), json.loads(a.extra)
    stream_speed(client, "hi", 0, a.tokens, extra)  # warm-up
    if a.streams:
        speeds = {n: concurrent_speed(client, n, a.tokens, extra) for n in map(int, a.streams.split(","))}
        print(" | ".join(f"{n} streams: {v:.1f} tok/s" for n, v in speeds.items()))
    else:
        res = {name: stream_speed(client, prompt, i + 1, a.tokens, extra) for i, (name, prompt) in enumerate(PROMPTS.items())}
        geo = math.exp(sum(map(math.log, res.values())) / len(res))
        print(" | ".join(f"{k} {v:.1f}" for k, v in res.items()), f"| geomean {geo:.1f} tok/s")


if __name__ == "__main__":
    main()
