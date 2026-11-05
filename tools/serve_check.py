#!/usr/bin/env python3
"""A real server end to end, as a client sees it: starts `qwenny serve` on a model, sends requests over HTTP and checks
the replies. Prints OK or FAIL per case and exits 1 if any failed. Standard library only.

  python3 -I tools/serve_check.py --model=Qwen/Qwen3.5-4B --bin=target/release/qwenny --slots=4 --ctx=32768 --bf16

Greedy requests are compared with each other: streamed against not, alone against concurrent, a continued conversation
against the same one from a cold prompt. The questions are ones any working model answers, so a wrong answer means the
output is wrong, not that the model is weak.
"""

import argparse
import concurrent.futures
import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Client:
    def __init__(self, port):
        self.base = f"http://127.0.0.1:{port}"

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return r.read().decode()

    def _post(self, body):
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        return urllib.request.Request(self.base + "/v1/chat/completions", data=data, headers=headers)

    def chat(self, **body):
        with urllib.request.urlopen(self._post(body), timeout=1800) as r:
            return json.load(r)

    def stream(self, **body):
        """The events of a streamed reply, reassembled as a reply's message and finish reason"""
        message, finish, usage = {"content": "", "reasoning_content": ""}, None, None
        with urllib.request.urlopen(self._post({**body, "stream": True, "stream_options": {"include_usage": True}}),
                                    timeout=1800) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                usage = event.get("usage") or usage
                for choice in event["choices"]:
                    delta = choice["delta"]
                    message["content"] += delta.get("content") or ""
                    message["reasoning_content"] += delta.get("reasoning_content") or ""
                    if delta.get("tool_calls"):
                        message["tool_calls"] = delta["tool_calls"]
                    finish = choice.get("finish_reason") or finish
        return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


def user(text):
    return {"role": "user", "content": text}


NO_THINK = {"enable_thinking": False}
GREEDY = {"temperature": 0, "top_p": 1.0, "presence_penalty": 0.0}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def parts(reply):
    choice = reply["choices"][0]
    message = choice["message"]
    return (message.get("content") or "").strip(), (message.get("reasoning_content") or "").strip(), choice["finish_reason"]


def expect(condition, what):
    if not condition:
        raise AssertionError(what)


def thinking_answers(c):
    text, reasoning, finish = parts(c.chat(messages=[user("What is 17 * 23? Give the final number.")], max_tokens=4096,
                                           seed=1))
    expect(finish == "stop", f"finish {finish}, reasoning ends {reasoning[-200:]!r}")
    expect(reasoning, "no reasoning")
    expect("391" in text, f"answer {text!r}")
    return f"391 after {len(reasoning)} characters of reasoning"


def thinking_off_answers(c):
    text, reasoning, finish = parts(c.chat(messages=[user("What is the capital of France? One word.")], max_tokens=64,
                                           chat_template_kwargs=NO_THINK, **GREEDY))
    expect(finish == "stop" and not reasoning, f"finish {finish}, reasoning {reasoning!r}")
    expect("Paris" in text, f"answer {text!r}")
    return repr(text)


def low_effort_answers(c):
    text, _, finish = parts(c.chat(messages=[user("What is 12 + 30? Give the final number.")], max_tokens=4096,
                                   reasoning_effort="low", seed=1))
    expect(finish == "stop" and "42" in text, f"finish {finish}, answer {text!r}")
    return repr(text[-60:])


def stream_matches(c):
    body = dict(messages=[user("Name three primary colors.")], max_tokens=256, **GREEDY)
    whole, streamed = c.chat(**body), c.stream(**body)
    expect(parts(whole) == parts(streamed), f"{parts(whole)} vs. {parts(streamed)}")
    expect(streamed["usage"] == whole["usage"], f"usage {streamed['usage']} vs. {whole['usage']}")
    return f"{whole['usage']['completion_tokens']} tokens"


def seed_repeats(c):
    body = dict(messages=[user("Write one sentence about the sea.")], max_tokens=256, seed=7)
    first, second, other = c.chat(**body), c.chat(**body), c.chat(**{**body, "seed": 8})
    expect(parts(first) == parts(second), "the same seed gave different replies")
    expect(parts(first) != parts(other), "another seed gave the same reply")
    return "the same seed repeats, another differs"


def length_limit(c):
    reply = c.chat(messages=[user("Count from 1 to 100.")], max_tokens=5, chat_template_kwargs=NO_THINK, **GREEDY)
    expect(reply["choices"][0]["finish_reason"] == "length", reply["choices"][0])
    expect(reply["usage"]["completion_tokens"] == 5, reply["usage"])
    return "5 tokens, finish length"


def stop_string(c):
    text, _, finish = parts(c.chat(messages=[user("Count from 1 to 10, separated by commas.")], max_tokens=128, stop=["5"],
                                   chat_template_kwargs=NO_THINK, **GREEDY))
    expect(finish == "stop" and "4" in text and "5" not in text, f"finish {finish}, {text!r}")
    return repr(text)


def tool_call(c):
    messages = [user("What is the weather in Paris right now? Use the tool.")]
    reply = c.chat(messages=messages, tools=[WEATHER], max_tokens=4096, seed=1)
    choice = reply["choices"][0]
    calls = choice["message"].get("tool_calls")
    expect(choice["finish_reason"] == "tool_calls" and calls, choice)
    expect(calls[0]["function"]["name"] == "get_weather", calls)
    arguments = json.loads(calls[0]["function"]["arguments"])
    expect("Paris" in arguments["city"], arguments)
    # the tool's answer goes back and the model uses it
    messages += [{"role": "assistant", "content": choice["message"].get("content") or "", "tool_calls": calls},
                 {"role": "tool", "content": "Sunny, 31 degrees."}]
    text, _, finish = parts(c.chat(messages=messages, tools=[WEATHER], max_tokens=4096, seed=1))
    expect(finish == "stop" and "31" in text, f"finish {finish}, {text!r}")
    return f"{arguments}, then {text[:60]!r}"


def conversation_continues(c):
    first = [user("My favourite number is 7349. Say OK.")]
    reply, _, _ = parts(c.chat(messages=first, max_tokens=64, chat_template_kwargs=NO_THINK, **GREEDY))
    turn = first + [{"role": "assistant", "content": reply}, user("What is my favourite number? Only the number.")]
    continued = parts(c.chat(messages=turn, max_tokens=64, chat_template_kwargs=NO_THINK, **GREEDY))
    expect("7349" in continued[0], continued)
    # a cold copy: another conversation in between takes the slot, then the same request again
    c.chat(messages=[user("Say something else entirely.")], max_tokens=32, chat_template_kwargs=NO_THINK, **GREEDY)
    again = parts(c.chat(messages=turn, max_tokens=64, chat_template_kwargs=NO_THINK, **GREEDY))
    expect(continued == again, f"{continued} vs. {again}")
    return repr(continued[0])


def concurrent_matches(c, n):
    questions = [f"What is {i + 2} + {i + 3}? Only the number." for i in range(n)]
    ask = lambda q: parts(c.chat(messages=[user(q)], max_tokens=32, chat_template_kwargs=NO_THINK, **GREEDY))
    alone = [ask(q) for q in questions]
    with concurrent.futures.ThreadPoolExecutor(n) as pool:
        together = list(pool.map(ask, questions))
    expect(alone == together, f"{alone} vs. {together}")
    for i, (text, _, _) in enumerate(alone):
        expect(str(2 * i + 5) in text, f"{questions[i]} {text!r}")
    return f"{n} requests at once, the same replies as one at a time"


def long_prompt(c):
    # about 12,000 tokens: several prompt chunks, within the default context of this tool
    filler = "\n".join(f"Record {i}: the shipment left warehouse {i % 17} on schedule." for i in range(300))
    secret = "The vault code is 58213."
    question = f"{filler}\n{secret}\n{filler}\n\nWhat is the vault code? Only the number."
    reply = c.chat(messages=[user(question)], max_tokens=32, chat_template_kwargs=NO_THINK, **GREEDY)
    text, _, finish = parts(reply)
    expect(finish == "stop" and "58213" in text, f"finish {finish}, {text!r}")
    return f"{reply['usage']['prompt_tokens']} prompt tokens"


def too_long_is_refused(c):
    try:
        c.chat(messages=[user("word " * 400000)], max_tokens=8)
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}"
    raise AssertionError("a prompt over the context was answered")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", default="target/release/qwenny")
    p.add_argument("--model", required=True)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--bf16", action="store_true")
    a = p.parse_args()

    failed = 0
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [a.bin, "serve", f"--model={a.model}", f"--port={port}", "--host=127.0.0.1", f"--slots={a.slots}",
               f"--ctx={a.ctx}", "--ui", f"--db={Path(tmp) / 'auth.db'}"] + (["--bf16"] if a.bf16 else [])
        log = open(Path(tmp) / "server.log", "w+")
        server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            c = Client(port)
            for _ in range(900):
                try:
                    if c.get("/health").strip() == "ok":
                        break
                except OSError:
                    pass
                if server.poll() is not None:
                    raise RuntimeError(f"the server exited: {server.returncode}")
                time.sleep(1)
            else:
                raise RuntimeError("the server did not come up")
            c.get("/")  # the web page
            model = json.loads(c.get("/v1/models"))["data"][0]["id"]
            print(f"OK    server up, model {model!r}", flush=True)
            cases = [thinking_answers, thinking_off_answers, low_effort_answers, stream_matches, seed_repeats, length_limit,
                     stop_string, tool_call, conversation_continues, lambda c: concurrent_matches(c, a.slots), long_prompt,
                     too_long_is_refused]
            for case in cases:
                name = getattr(case, "__name__", "?")
                name = "concurrent_matches" if name == "<lambda>" else name
                start = time.time()
                try:
                    print(f"OK    {name}: {case(c)} ({time.time() - start:.1f} s)", flush=True)
                except Exception as e:  # a case's failure is reported; the others still run
                    failed += 1
                    print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:600]}", flush=True)
            expect(server.poll() is None, "the server died")
        except (RuntimeError, AssertionError, OSError) as e:
            failed += 1
            log.seek(0)
            print(f"FAIL  {e}\n{log.read()[-2000:]}", flush=True)
        finally:
            server.terminate()
            server.wait()
    print(f"{failed} failed" if failed else "all cases OK")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
