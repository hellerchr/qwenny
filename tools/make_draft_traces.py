#!/usr/bin/env python3
"""A model's own replies, for training a drafter on what it really writes (tools/train_dflash2.py): sends prompts to a
running `qwenny serve` at its default sampling and appends one JSON line per reply. Rerun to continue: done prompts are
skipped. Standard library only.

  python3 -I tools/make_draft_traces.py <server url> <out.jsonl> <count> [--lcb=f.jsonl] [--tickets=f.jsonl] [--rows=f.jsonl]

Prompts, shuffled with a fixed seed: reasoning and code tasks over topics, German smart-home requests with tools, and
what the options add: LiveCodeBench problems (--lcb), German support tickets (--tickets), any rows with "messages"
(--rows, repeatable). A line: {"id", "messages", "tools", "kwargs", "reasoning", "content", "finish", "tokens"}.
"""

import argparse
import concurrent.futures
import json
import random
import threading
import urllib.request
from pathlib import Path

TOPICS = """hash maps; binary search; TCP congestion control; garbage collection; SQL indexes; the printing press;
photosynthesis; compound interest; the French Revolution; vaccines; black holes; plate tectonics; Rust ownership;
Python decorators; regular expressions; dynamic programming; graph traversal; public key cryptography; TLS handshakes;
SQL injection; cross-site scripting; buffer overflows; race conditions; memory leaks; REST API design; database
transactions; load balancers; caching strategies; bread baking; heat pumps; solar panels; bicycle maintenance; the water
cycle; supply and demand; inflation; probability puzzles; prime numbers; matrix multiplication; Bayes' theorem; sorting
algorithms; Linux processes; Docker containers; Kubernetes pods; git rebasing; unit testing; code review; home networking;
DNS; Zigbee; electric cars; sleep; coffee; the Roman Empire; volcanoes; orbital mechanics""".replace("\n", " ").split("; ")

TASKS = ["Explain {t} in detail, with a concrete example.", "What are the most common mistakes people make with {t}, and how do they avoid them?",
         "Write a short tutorial on {t} for a beginner.", "Compare two different approaches to {t} and recommend one.",
         "Write a Python program that demonstrates or simulates {t}, with comments.", "Write a Rust function related to {t}, with unit tests.",
         "Pose a hard quantitative problem about {t}, then solve it step by step.", "Erkläre {t} verständlich auf Deutsch.",
         "Write a JSON document that summarizes the key facts about {t}.", "Review this claim critically: everyone misunderstands {t}."]

AREAS = ["Wohnzimmer", "Küche", "Bad", "Schlafzimmer", "Flur", "Büro", "Kinderzimmer", "Keller", "Garten"]
DEVICES = ["Deckenlicht", "Stehlampe", "Spot", "Rollladen", "Heizung", "Lüftung", "Fernseher", "Steckdose"]
HOME_SYSTEM = ("Du bist der Assistent dieses Zuhauses. Antworte auf Deutsch, freundlich und knapp. Räume: " + ", ".join(AREAS)
               + ". Geräte: " + ", ".join(DEVICES) + ". Listen: Supermarkt, dm. Rufe für Aktionen die passende Funktion auf.")
HOME_REQUESTS = ["Mach {d} im {a} an.", "Schalte {d} im {a} aus.", "Stell das {a} auf {n} Grad.", "Setz {i} auf die Einkaufsliste.",
                 "Wie warm ist es im {a}?", "Ist {d} im {a} noch an?", "Mach alles im {a} aus und dann {d} an.", "Wie lange kocht {i}?",
                 "Erzähl mir etwas Interessantes über {i}.", "Dimme {d} im {a} auf {n} Prozent."]
ITEMS = ["Milch", "Kaffee", "Reis", "Nudeln", "Kartoffeln", "Eier", "Tomaten", "Spinat"]
HOME_TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object", "properties": p}}}
              for n, d, p in [("HassTurnOn", "Schaltet ein Gerät ein.", {"name": {"type": "string"}, "area": {"type": "string"}}),
                              ("HassTurnOff", "Schaltet ein Gerät aus.", {"name": {"type": "string"}, "area": {"type": "string"}}),
                              ("HassLightSet", "Dimmt ein Licht.", {"name": {"type": "string"}, "brightness": {"type": "number"}}),
                              ("HassClimateSetTemperature", "Stellt die Temperatur ein.", {"area": {"type": "string"}, "temperature": {"type": "number"}}),
                              ("HassGetState", "Liest den Zustand.", {"name": {"type": "string"}, "area": {"type": "string"}}),
                              ("todo_add", "Fügt einen Listeneintrag hinzu.", {"list": {"type": "string"}, "item": {"type": "string"}})]]
LCB = ("You will be given a question (problem specification) and will generate a correct Python program that matches the "
       "specification and passes all tests.\n\nQuestion: {q}\n\nRead the inputs from stdin, solve the problem and write the "
       "answer to stdout. Enclose your code within delimiters as follows.\n```python\n# YOUR CODE HERE\n```\n")


def user(text):
    return [{"role": "user", "content": text}]


def prompts(a):
    rng = random.Random(1)
    out = [(f"topic-{i}-{j}", user(task.format(t=t)), None, {}) for i, t in enumerate(TOPICS) for j, task in enumerate(TASKS)]
    for i in range(600):
        text = rng.choice(HOME_REQUESTS).format(d=rng.choice(DEVICES), a=rng.choice(AREAS), n=rng.randint(15, 80), i=rng.choice(ITEMS))
        # the assistant's setting (quick replies) and the reasoning kind in the ratio 3 : 1
        kwargs = {"enable_thinking": False} if i % 4 else {}
        out.append((f"home-{i}", [{"role": "system", "content": HOME_SYSTEM}] + user(text), HOME_TOOLS, kwargs))
    if a.lcb:
        for line in open(a.lcb):
            r = json.loads(line)
            out.append((f"lcb-{r['question_id']}", user(LCB.format(q=r["question_content"])), None, {}))
    if a.tickets:
        for i, line in enumerate(open(a.tickets)):
            messages = [m for m in json.loads(line)["messages"] if m["role"] != "assistant"]
            out.append((f"ticket-{i}", messages, None, {}))
    for path in a.rows:
        for i, line in enumerate(open(path)):
            r = json.loads(line)
            if r.get("messages"):
                out.append((f"{Path(path).stem}-{i}", [m for m in r["messages"] if m["role"] != "assistant"], None, {}))
    rng.shuffle(out)
    return out


def ask(url, item, seed, max_tokens):
    pid, messages, tools, kwargs = item
    body = {"messages": messages, "max_tokens": max_tokens, "seed": seed, "chat_template_kwargs": kwargs}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        reply = json.load(r)
    choice = reply["choices"][0]
    m = choice["message"]
    return {"id": pid, "messages": messages, "tools": tools, "kwargs": kwargs, "reasoning": m.get("reasoning_content") or "",
            "content": m.get("content") or "", "tool_calls": m.get("tool_calls"), "finish": choice["finish_reason"],
            "tokens": reply["usage"]["completion_tokens"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("out")
    p.add_argument("count", type=int, help="prompts in all, done ones included")
    p.add_argument("--lcb")
    p.add_argument("--tickets")
    p.add_argument("--rows", action="append", default=[])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    a = p.parse_args()

    done = {json.loads(line)["id"] for line in open(a.out)} if Path(a.out).exists() else set()
    work = [x for x in prompts(a)[:a.count] if x[0] not in done]
    print(f"{len(done)} done, {len(work)} to go", flush=True)
    lock, total = threading.Lock(), [0, 0]
    with open(a.out, "a") as out, concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        jobs = [pool.submit(ask, a.url, x, i, a.max_tokens) for i, x in enumerate(work)]
        for job in concurrent.futures.as_completed(jobs):
            try:
                row = job.result()
            except OSError as e:  # one failed request does not end the run; a rerun takes it again
                print(f"failed: {e}", flush=True)
                continue
            with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                total[0] += 1
                total[1] += row["tokens"]
                if total[0] % 50 == 0:
                    print(f"{total[0]} replies, {total[1]} tokens", flush=True)
    print(f"done: {total[0]} replies, {total[1]} tokens", flush=True)


if __name__ == "__main__":
    main()
