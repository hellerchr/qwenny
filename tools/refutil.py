"""What the make_*.py reference scripts share."""
import json
import sys

import torch


def cli(doc):
    """(positional arguments, {option: value}) of a command line: `--key=value`, or `--key` alone as ''; --help prints
    `doc` (the script's docstring)"""
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        print(doc)
        sys.exit(0)
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in argv if a.startswith("--"))
    return [a for a in argv if not a.startswith("--")], opts


def prompts(tests):
    """The reference prompts of make_reference.py and make_reference_lora.py from the Fjordhaus dev rows: a short
    question, a ticket (system + user, ~400 tokens) and several tickets (~2,000 tokens, more than one prompt chunk)"""
    rows = [json.loads(l) for l in open(tests)]
    ticket = [m for m in rows[0]["messages"] if m["role"] != "assistant"]
    many = "\n\n".join(m["content"] for r in rows[:12] for m in r["messages"] if m["role"] == "user")
    return [
        [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
        ticket,
        [{"role": "user", "content": "Summarize these customer tickets in one line each.\n\n" + many}],
    ]


def positions(n):
    """The positions of a prompt of n tokens whose full logits go into a reference (the last one always)"""
    return sorted({*range(0, n, 7 if n < 1000 else 31), n - 1})


def chunked(f, rows, n=256):
    """f over `rows` n at a time, as float32 on the CPU, concatenated (all the logits at once do not fit)"""
    return torch.cat([f(rows[c:c + n]).float().cpu() for c in range(0, len(rows), n)])
