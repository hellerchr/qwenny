#!/usr/bin/env python3
"""Require CUDA and MLX to return identical response bytes for identical requests.

Runs a fresh qwenny server per backend. Transport metadata and token timings are ignored; message content,
reasoning, tool calls and finish reasons are compared. Streaming is reassembled before comparison so different
kernel/step boundaries do not change the result. Standard library only.

  python3 tools/backend_diff.py
  python3 tools/backend_diff.py --model=Qwen/Qwen3.5-4B --bf16
  python3 tools/backend_diff.py --model=RedHatAI/Qwen3.5-4B-quantized.w4a16
"""

import argparse
import concurrent.futures
import difflib
import json
from pathlib import Path

from qwenny_client import server


def canonical(response):
    choice = response["choices"][0]
    message = choice["message"]
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
    }


def cases():
    tool = [{
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Read the current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                "required": ["city"],
            },
        },
    }]
    paragraph = "The small observatory measured the comet each clear night and recorded every instrument change. "
    return [
        ("greedy", {
            "messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}],
            "max_tokens": 64, "temperature": 0, "seed": 11,
        }),
        ("seeded", {
            "messages": [{"role": "user", "content": "Write a compact story about a robot learning to bake rye bread."}],
            "max_tokens": 80, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
            "presence_penalty": 1.5, "seed": 314159,
        }),
        ("long-prompt", {
            "messages": [{"role": "user", "content": "Summarize the observation log in two sentences.\n\n" + paragraph * 55}],
            "max_tokens": 64, "temperature": 0, "seed": 12,
        }),
        ("no-thinking-penalties", {
            "messages": [{"role": "user", "content": "List four imaginative names for a quiet library café."}],
            "max_tokens": 64, "enable_thinking": False, "temperature": 0.7, "top_p": 0.8,
            "top_k": 20, "repetition_penalty": 1.1, "presence_penalty": 1.5, "seed": 2718,
        }),
        ("tool", {
            "messages": [{"role": "user", "content": "Use the tool to check Paris weather for two days."}],
            "tools": tool, "max_tokens": 64, "temperature": 0, "seed": 13,
        }),
    ]


def run_suite(client):
    results = {}
    first = None
    for name, body in cases():
        results[name] = canonical(client.chat(**body))
        if name == "greedy":
            first = results[name]
        print(f"  {name}: {len(results[name]['content'].encode())} content bytes", flush=True)

    # Reassembled streaming bytes, independent of the number of engine steps and SSE chunks.
    results["streaming"] = canonical(client.stream(
        messages=[{"role": "user", "content": "Give a two-paragraph answer about tides, with a short title."}],
        max_tokens=72, temperature=0.9, top_p=0.95, top_k=20, seed=1618))

    assistant = {"role": "assistant", "content": first["content"]}
    if first["reasoning_content"]:
        assistant["reasoning_content"] = first["reasoning_content"]
    results["continuation"] = canonical(client.chat(
        messages=[{"role": "user", "content": "Explain in three sentences why the sky is blue."}, assistant,
                  {"role": "user", "content": "Now give one simple analogy for a child."}],
        max_tokens=64, temperature=0, seed=14))

    def concurrent_body(i):
        return {
            "messages": [{"role": "user", "content": f"Concurrent task {i}: describe a useful workshop tool in two sentences."}],
            "max_tokens": 56, "temperature": 0.8, "top_p": 0.9, "top_k": 20, "seed": 500 + i,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda i: canonical(client.chat(**concurrent_body(i))), range(4)))
    for i, reply in enumerate(replies):
        results[f"concurrent-{i}"] = reply
    return results


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--bin", default="target/release/qwenny")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--cache-mem", type=float, default=4, help="GiB of device memory for the slots caches")
    args = parser.parse_args()
    binary = str(Path(args.bin).resolve())
    results = {}
    for backend in ("cuda", "mlx"):
        print(f"== {backend}", flush=True)
        flags = [f"--backend={backend}", "--ctx=4096", "--slots=4", "--drafts=0", f"--cache-mem={args.cache_mem}"]
        with server(binary, args.model, *flags, *(["--bf16"] if args.bf16 else [])) as client:
            results[backend] = run_suite(client)

    bad = []
    for name in results["cuda"]:
        a, b = encoded(results["cuda"][name]), encoded(results["mlx"][name])
        if a == b:
            print(f"OK    {name}: {len(a)} bytes identical")
            continue
        bad.append(name)
        at = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"FAIL  {name}: first differing byte {at}; CUDA {len(a)} bytes, MLX {len(b)} bytes")
        diff = "\n".join(difflib.unified_diff(
            [json.dumps(results["cuda"][name], ensure_ascii=False, indent=2)],
            [json.dumps(results["mlx"][name], ensure_ascii=False, indent=2)],
            fromfile="cuda", tofile="mlx",
        ))
        print(diff[:6000])
    if bad:
        raise SystemExit(f"response bytes differ: {', '.join(bad)}")
    print(f"all {len(results['cuda'])} CUDA / MLX response comparisons are byte-identical")


if __name__ == "__main__":
    main()
