"""What the tools that talk to a `qwenny serve` share: a free port, a server to start and stop, and its chat API.
Standard library only."""
import contextlib
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
    """A server's HTTP API: `chat` and `stream` take a request's JSON fields as keyword arguments and give a reply in
    the shape of a non-streaming one: {"choices": [{"message": ..., "finish_reason": ...}], "usage": ...}"""

    def __init__(self, port=None, url=None):
        self.base = url.rstrip("/") if url else f"http://127.0.0.1:{port}"

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return r.read().decode()

    def _open(self, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            e.msg += ": " + e.read().decode(errors="replace")[:500]
            raise

    def chat(self, **body):
        with self._open(body) as r:
            return json.load(r)

    def events(self, **body):
        """The parsed server-sent events of a streamed reply, with the usage of the last"""
        with self._open({**body, "stream": True, "stream_options": {"include_usage": True}}) as r:
            for line in r:
                line = line.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    yield json.loads(line[6:])

    def stream(self, **body):
        """A streamed reply reassembled: the message (content, reasoning_content, tool_calls), finish reason, usage"""
        message, finish, usage, tools = {"content": "", "reasoning_content": ""}, None, None, {}
        for event in self.events(**body):
            usage = event.get("usage") or usage
            for choice in event["choices"]:
                delta = choice["delta"]
                message["content"] += delta.get("content") or ""
                message["reasoning_content"] += delta.get("reasoning_content") or ""
                for part in delta.get("tool_calls") or []:
                    tool = tools.setdefault(part.get("index", 0), {"id": "", "type": "function",
                                                                   "function": {"name": "", "arguments": ""}})
                    tool["id"] += part.get("id") or ""
                    for key in ("name", "arguments"):
                        tool["function"][key] += (part.get("function") or {}).get(key) or ""
                finish = choice.get("finish_reason") or finish
        if tools:
            message["tool_calls"] = [tools[i] for i in sorted(tools)]
        return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


@contextlib.contextmanager
def server(binary, model, *args, timeout=900):
    """`qwenny serve` on a free local port with a password database of its own; yields a Client once /health says ok.
    The server's log ends the error when it does not come up."""
    port = free_port()
    with tempfile.TemporaryDirectory(prefix="qwenny-tool-") as tmp:
        cmd = [binary, "serve", f"--model={model}", f"--port={port}", "--host=127.0.0.1", f"--db={Path(tmp) / 'auth.db'}",
               *args]
        with open(Path(tmp) / "server.log", "w+") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                client, deadline = Client(port), time.monotonic() + timeout
                while True:
                    try:
                        if client.get("/health").strip() == "ok":
                            break
                    except OSError:
                        pass
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        what = f"exited with {process.returncode}" if process.poll() is not None else "did not load in time"
                        raise RuntimeError(f"the server {what}:\n{log.read()[-2000:]}")
                    time.sleep(0.5)
                yield client
                if process.poll() is not None:
                    raise RuntimeError("the server died")
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
