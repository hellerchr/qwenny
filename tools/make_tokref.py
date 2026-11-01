"""Prompts of varied conversations rendered and tokenized by Transformers, for `qwenny chatcheck`.
Usage: ~/.cache/qwenny/venv/bin/python -I tools/make_tokref.py [model dir or HF id] [out.json]
(defaults: Qwen/Qwen3.5-4B, tools/tokref-qwen.json)
Per case: name, messages (OpenAI format, tool call arguments as objects: Transformers' template needs them so), tools,
kwargs (chat_template_kwargs), text (apply_chat_template, tokenize=False, add_generation_prompt=True) and ids
(apply_chat_template, tokenize=True)."""
import json, os, sys
from transformers import AutoTokenizer

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3.5-4B"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokref-qwen.json")
tok = AutoTokenizer.from_pretrained(model)

U = lambda c: {"role": "user", "content": c}
A = lambda c, **kw: {"role": "assistant", "content": c, **kw}
S = lambda c: {"role": "system", "content": c}
T = lambda c: {"role": "tool", "content": c}
call = lambda name, **args: {"type": "function", "function": {"name": name, "arguments": args}}

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city (°C, wind in km/h).",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name, e.g. \"München\""},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
WRITE = {"type": "function", "function": {
    "name": "write_file", "description": "Write a text file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}, "overwrite": {"type": "boolean"},
        "mode": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}},
        "meta": {"type": "object"}}, "required": ["path", "content"]}}}

CODE = "def add(a, b):\n    return a + b\n\n\nprint(add(1, 2))\n"
cases = [
    ("single user, thinking by default", [U("What is the capital of France? Answer in one sentence.")], None, {}),
    ("single user, thinking off", [U("What is the capital of France? Answer in one sentence.")], None, {"enable_thinking": False}),
    ("system + user, thinking on", [S("You are a terse assistant."), U("Hi!")], None, {"enable_thinking": True}),
    ("system + user, thinking off, padded whitespace",
     [S("  Answer in German.\n\n"), U("\n  Wie spät ist es?  \n")], None, {"enable_thinking": False}),
    ("multi-turn, earlier reasoning is dropped",
     [S("Be brief."), U("2+2?"), A("4", reasoning_content="Simple sum."), U("And times 3?")], None, {}),
    ("multi-turn, thinking off, reasoning inline in an earlier reply",
     [U("Name a prime."), A("<think>\nAny will do.\n</think>\n\n7"), U("Another one."), A("11"), U("One more.")],
     None, {"enable_thinking": False}),
    ("code and unicode", [U("Explain:\n```python\n" + CODE + "```\nUnd bitte auf Deutsch: Grüße ☃ \U0001F600 你好")], None, {}),
    ("content as a list of text parts", [U([{"type": "text", "text": "Part one."}, {"type": "text", "text": " Part two."}])], None, {}),
    ("tools + system + user", [S("You answer weather questions."), U("Wie ist das Wetter in München?")], [WEATHER, WRITE], {}),
    ("tools + user, thinking off", [U("Write hello to /tmp/a.txt")], [WRITE], {"enable_thinking": False}),
    ("tool call and response, reasoning after the last query",
     [U("Weather in Paris?"),
      A("", reasoning_content="I should call the weather tool.", tool_calls=[call("get_weather", city="Paris", unit="celsius")]),
      T('{"temp": 14, "sky": "cloudy"}')], [WEATHER, WRITE], {}),
    ("two tool calls, two responses, text before the calls",
     [S("Use tools."), U("Weather in Paris and Rome?"),
      A("Let me check both.", reasoning_content="Two calls.",
        tool_calls=[call("get_weather", city="Paris"), call("get_weather", city="Rome", unit="fahrenheit")]),
      T("14 C, cloudy"), T("21 C, sunny")], [WEATHER], {"enable_thinking": False}),
    ("tool call with every argument type, answer, next user turn",
     [U("Save my notes."),
      A("", tool_calls=[call("write_file", path="/tmp/n.txt", content="line 1\nline 2\n\nline 4", overwrite=True, mode=420,
                             tags=["a", "b ä"], meta={"k": [1, 2.5, None], "s": "x"}, extra=None, ratio=1.0)]),
      T("written 22 bytes"), A("Saved.", reasoning_content="Done."), U("Thanks, and delete it.")], [WRITE], {}),
    ("tool calls with function key absent and arguments empty",
     [U("Ping"), A("pong?", tool_calls=[{"function": {"name": "ping", "arguments": {}}}]), T("ok")], None, {"enable_thinking": False}),
]

res = []
for name, messages, tools, kwargs in cases:
    kw = dict(tools=tools, **kwargs) if tools else dict(kwargs)
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, **kw)["input_ids"]
    assert ids == tok.encode(text, add_special_tokens=False), name
    res.append({"name": name, "messages": messages, "tools": tools, "kwargs": kwargs, "text": text, "ids": ids})
    print(f"{len(ids):5d} tokens  {name}")
json.dump({"model": model, "cases": res}, open(out, "w"), ensure_ascii=False, indent=1)
print("wrote", out)
