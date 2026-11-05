"""Reference outputs of Qwen3.5's MTP layer (multi-token prediction, the draft model of speculative decoding) in
float32, built as vLLM's Qwen3_5MultiTokenPredictor: x = fc([norm_e(embed(t_{q+1})); norm_h(h_q)]), one full-attention
decoder layer over the MTP's own positions, then norm and the shared output layer: logits for t_{q+2}. h_q is the
target's final hidden state (after its final norm) at position q; the MTP token at q has position q.
Usage: python make_reference_mtp.py <model dir or HF id> <fjordhaus dev.jsonl> <out.safetensors>
Per prompt i (a ticket with its reply, teacher-forced): ids_i, target_i (the target's argmax at every position),
mtp_i (the MTP's argmax at positions 0 .. T-2), pos_i and logits_i (MTP logits at some positions), chain_i (the second
draft: the MTP again on its own output state at q and its first draft, at position q + 1, for q in pos_i)."""
import copy, json
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from huggingface_hub import snapshot_download
from pathlib import Path
from refutil import chunked, cli

model_dir, tests, out = cli(__doc__)[0]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="auto",
                                             max_memory={0: "9GiB", "cpu": "24GiB"})
model.eval()
lm = model.model.language_model if hasattr(model.model, "language_model") else model.model
cfg = copy.deepcopy(lm.config)
cfg.layer_types = ["full_attention"]

# the MTP weights (Transformers does not load them)
path = Path(model_dir) if Path(model_dir).is_dir() else Path(snapshot_download(model_dir))
w = {}
for f in path.glob("*.safetensors"):
    with safe_open(f, "pt") as sf:
        w |= {k: sf.get_tensor(k).float() for k in sf.keys() if k.startswith("mtp.")}
layer = qm.Qwen3_5DecoderLayer(cfg, 0)
layer.load_state_dict({k[len("mtp.layers.0."):]: v for k, v in w.items() if k.startswith("mtp.layers.0.")})
norm = lambda name: (lambda m: (m.load_state_dict({"weight": w[f"mtp.{name}.weight"]}), m)[1])(
    qm.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps))
pre_e, pre_h, fnorm = norm("pre_fc_norm_embedding"), norm("pre_fc_norm_hidden"), norm("norm")
fc = torch.nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
fc.load_state_dict({"weight": w["mtp.fc.weight"]})
for m in (layer, pre_e, pre_h, fnorm, fc):
    m.eval()
head, embed = model.get_output_embeddings(), lm.embed_tokens

# the rotary tables the target used, by position
rope = {}
lm.rotary_emb.register_forward_hook(lambda m, i, o: rope.__setitem__("cs", (o[0].cpu(), o[1].cpu())))

def mtp(x_ids, h, p0):
    """MTP over tokens x_ids [n] with target states h [n][H] at positions p0 .. p0 + n - 1 (causal over these)."""
    n = len(x_ids)
    e = embed(x_ids.to(embed.weight.device)).float().cpu()
    x = fc(torch.cat([pre_e(e), pre_h(h)], -1))[None]
    cos, sin = (t[:, p0:p0 + n] for t in rope["cs"])
    mask = torch.full((n, n), float("-inf")).triu(1)[None, None]
    y = layer(x, position_embeddings=(cos, sin), attention_mask=mask)
    y = y[0] if isinstance(y, tuple) else y
    hs = fnorm(y[0])
    return hs, chunked(lambda x: head(x.to(head.weight.device)), hs)

rows = [json.loads(l) for l in open(tests)]
tensors = {}
for i, r in enumerate(rows[:2]):
    ids = tok.apply_chat_template(r["messages"], enable_thinking=False, return_tensors="pt",
                                  return_dict=True)["input_ids"]
    with torch.no_grad():
        h = lm(input_ids=ids.to(model.device)).last_hidden_state[0].float().cpu()  # after the final norm
        target = chunked(lambda x: head(x.to(head.weight.device)), h).argmax(-1)
        T = ids.shape[1]
        hs, logits = mtp(ids[0, 1:], h[:-1], 0)
        draft = logits.argmax(-1)
        pos = sorted({*range(0, T - 1, 11), T - 2})
        # the chain: the MTP on its own state at q and its draft, at position q + 1 (one token, attending to the
        # MTP's entries 0 .. q, recomputed here as a prefix plus the new token)
        chain = []
        for q in pos[:24]:
            x = torch.cat([ids[0, 1:q + 2], draft[q:q + 1]])
            hh = torch.cat([h[:q + 1], hs[q:q + 1]])
            chain.append(mtp(x, hh, 0)[1][-1].argmax())
    acc = (draft[:-1] == target[1:T - 1]).float().mean().item()
    print(f"prompt {i}: {T} tokens, draft 1 agrees with the target's next argmax at {acc:.1%}")
    tensors |= {f"ids_{i}": ids[0].int(), f"target_{i}": target.int(), f"mtp_{i}": draft.int(),
                f"pos_{i}": torch.tensor(pos, dtype=torch.int32), f"logits_{i}": logits[pos].contiguous(),
                f"chain_{i}": torch.stack(chain).int()}
from safetensors.torch import save_file
save_file(tensors, out)
print("wrote", out)
