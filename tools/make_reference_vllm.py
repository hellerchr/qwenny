"""An oracle reference from a running vLLM server (its own kernels, e.g. Marlin for int4 checkpoints), for `qwenny check`.
Usage: python make_reference_vllm.py <reference.safetensors> <vLLM base URL> <model name> <out.safetensors> [--top=20]
Takes the prompts (ids_i) of an existing reference and asks vLLM for the top-k log-probabilities at every position
(prompt_logprobs) and 32 greedy tokens. Per prompt i: ids_i, topk_ids_i and topk_lp_i [positions][k] (the k most
likely next tokens after each position and their log-probabilities, rank order), argmax_i, gen_i.
vLLM returns log-probabilities, not logits: `qwenny check` compares its own log-softmax on those tokens."""
import json, os, sys, urllib.request
import torch
from safetensors.torch import load_file, save_file

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
ref, url, name, out = args[:4]
top = int(opts.get("top", 20))
r = load_file(ref)
# a server with key auth (vLLM's --api-key): the key from VLLM_API_KEY
auth = {"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]} if os.environ.get("VLLM_API_KEY") else {}
tensors = {}
i = 0
while f"ids_{i}" in r:
    ids = r[f"ids_{i}"].tolist()
    body = {"model": name, "prompt": ids, "max_tokens": 32, "temperature": 0, "logprobs": top, "prompt_logprobs": top,
            "return_tokens_as_token_ids": True, "repetition_penalty": 1.0}
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"} | auth)
    res = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]
    # position p predicts token p + 1: prompt_logprobs[p + 1]; the last position: the first generated token's
    rows = []
    for p in range(len(ids)):
        if p + 1 < len(ids):
            d = {int(k): v["logprob"] for k, v in res["prompt_logprobs"][p + 1].items() if v["rank"] <= top}
        else:
            d = {int(k.removeprefix("token_id:")): v for k, v in res["logprobs"]["top_logprobs"][0].items()}
        rows.append(sorted(d.items(), key=lambda kv: -kv[1])[:top])
    gen = [int(t.removeprefix("token_id:")) for t in res["logprobs"]["tokens"]]
    tensors |= {f"ids_{i}": torch.tensor(ids, dtype=torch.int32),
                f"topk_ids_{i}": torch.tensor([[t for t, _ in row] for row in rows], dtype=torch.int32),
                f"topk_lp_{i}": torch.tensor([[lp for _, lp in row] for row in rows], dtype=torch.float32),
                f"argmax_{i}": torch.tensor([row[0][0] for row in rows], dtype=torch.int32),
                f"gen_{i}": torch.tensor(gen, dtype=torch.int32)}
    print(f"prompt {i}: {len(ids)} tokens, greedy {res['text'][:60]!r}", flush=True)
    i += 1
save_file(tensors, out)
