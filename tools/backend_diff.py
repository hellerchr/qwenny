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
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def canonical(response):
    choice = response["choices"][0]
    message = choice["message"]
    return {
        "content": message.get("content") or "",
        "reasoning_content": message.get("reasoning_content") or "",
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
    }


def post(port, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error


def streamed(port, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({**body, "stream": True}, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    content, reasoning, finish = [], [], None
    tools = {}
    with urllib.request.urlopen(req, timeout=900) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            choice = event["choices"][0]
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            finish = choice.get("finish_reason") or finish
            for part in delta.get("tool_calls") or []:
                index = part.get("index", 0)
                tool = tools.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                tool["id"] += part.get("id") or ""
                function = part.get("function") or {}
                tool["function"]["name"] += function.get("name") or ""
                tool["function"]["arguments"] += function.get("arguments") or ""
    return {
        "content": "".join(content),
        "reasoning_content": "".join(reasoning),
        "tool_calls": [tools[i] for i in sorted(tools)],
        "finish_reason": finish,
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


def run_suite(port):
    results = {}
    first = None
    for name, body in cases():
        results[name] = canonical(post(port, body))
        if name == "greedy":
            first = results[name]
        print(f"  {name}: {len(results[name]['content'].encode())} content bytes", flush=True)

    # Reassembled streaming bytes, independent of the number of engine steps and SSE chunks.
    results["streaming"] = streamed(port, {
        "messages": [{"role": "user", "content": "Give a two-paragraph answer about tides, with a short title."}],
        "max_tokens": 72, "temperature": 0.9, "top_p": 0.95, "top_k": 20, "seed": 1618,
    })

    assistant = {"role": "assistant", "content": first["content"]}
    if first["reasoning_content"]:
        assistant["reasoning_content"] = first["reasoning_content"]
    results["continuation"] = canonical(post(port, {
        "messages": [{"role": "user", "content": "Explain in three sentences why the sky is blue."}, assistant,
                     {"role": "user", "content": "Now give one simple analogy for a child."}],
        "max_tokens": 64, "temperature": 0, "seed": 14,
    }))

    def concurrent_body(i):
        return {
            "messages": [{"role": "user", "content": f"Concurrent task {i}: describe a useful workshop tool in two sentences."}],
            "max_tokens": 56, "temperature": 0.8, "top_p": 0.9, "top_k": 20, "seed": 500 + i,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda i: canonical(post(port, concurrent_body(i))), range(4)))
    for i, reply in enumerate(replies):
        results[f"concurrent-{i}"] = reply
    return results


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve(binary, model, backend, bf16, directory, spec=0):
    port = free_port()
    log_path = Path(directory) / f"{backend}.log"
    log = open(log_path, "wb")
    command = [binary, "serve", f"--backend={backend}", f"--model={model}", "--host=127.0.0.1",
               f"--port={port}", "--ctx=4096", "--slots=4", f"--drafts={spec}", f"--db={directory}/{backend}.db"]
    if bf16:
        command.append("--bf16")
    env = os.environ.copy()
    env.setdefault("QWENNY_CACHE_MEM", "4")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    try:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{backend} server exited with {process.returncode}; see {log_path}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                    if response.read() == b"ok":
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.25)
        else:
            raise TimeoutError(f"{backend} server did not load in 600 seconds; see {log_path}")
        return process, port, log, log_path
    except Exception:
        process.kill()
        process.wait()
        log.close()
        raise


def stop(server):
    process, _, log, _ = server
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    log.close()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--bin", default="target/release/qwenny")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()
    binary = str(Path(args.bin).resolve())
    with tempfile.TemporaryDirectory(prefix="qwenny-backend-diff-") as directory:
        results = {}
        for backend in ("cuda", "mlx"):
            print(f"== {backend}", flush=True)
            server = serve(binary, args.model, backend, args.bf16, directory)
            try:
                results[backend] = run_suite(server[1])
            finally:
                stop(server)
            Path(directory, f"{backend}.json").write_text(json.dumps(results[backend], ensure_ascii=False, indent=2))

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
