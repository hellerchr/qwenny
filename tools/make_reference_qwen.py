"""Reference outputs of Qwen3.5 from Transformers in float32, for the engine's correctness checks.
Usage: python make_reference_qwen.py <model dir or HF id> <fjordhaus dev.jsonl> <out.safetensors>
Per prompt i: ids_i (prompt tokens), pos_i (positions with full logits), logits_i (logits at pos_i), argmax_i (argmax
at every position), gen_i (32 greedy tokens after the prompt).
Per prompt i and layer L in LAYERS: xn_i_L (input_layernorm output, the mixer's input, [T][H]) and mix_i_L (the
mixer's output before the residual add, [T][H]) for `qwenny layercheck`.
Prompt 0: a short question; 1: a Fjordhaus ticket (system + user, ~400 tokens); 2: several tickets (~2,000 tokens,
more than one 512-token prompt chunk)."""
import json, sys
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

LAYERS = [0, 3, 30, 31]  # linear, full, linear, full attention

model_dir, tests, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(model_dir)
# fp32 does not fit the 16 GB GPU: the rest of the layers run on the CPU
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, device_map="auto",
                                             max_memory={0: "9GiB", "cpu": "24GiB"})
model.eval()
layers = next(m for n, m in model.named_modules() if n.endswith("layers") and isinstance(m, torch.nn.ModuleList))
print(type(model).__name__, len(layers), "layers")

captured = {}
def hooks(L):
    layer = layers[L]
    mixer = layer.linear_attn if hasattr(layer, "linear_attn") else layer.self_attn
    layer.input_layernorm.register_forward_hook(lambda m, i, o: captured.__setitem__(f"xn_{L}", o[0].float().cpu()))
    mixer.register_forward_hook(lambda m, i, o: captured.__setitem__(
        f"mix_{L}", (o[0] if isinstance(o, tuple) else o)[0].float().cpu()))
for L in LAYERS:
    hooks(L)

rows = [json.loads(l) for l in open(tests)]
ticket = [m for m in rows[0]["messages"] if m["role"] != "assistant"]
many = "\n\n".join(m["content"] for r in rows[:12] for m in r["messages"] if m["role"] == "user")
prompts = [
    [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    ticket,
    [{"role": "user", "content": "Summarize these customer tickets in one line each.\n\n" + many}],
]
tensors = {}
dev = next(model.parameters()).device
for i, msgs in enumerate(prompts):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                                  return_dict=True)["input_ids"].to(dev)
    captured.clear()
    with torch.no_grad():
        # the final hidden states, then the logits 256 rows at a time (all of them at once do not fit)
        hidden = model.model(input_ids=ids).last_hidden_state[0]  # after the final norm
        layer_out = dict(captured)
        head = model.get_output_embeddings()
        chunks = [head(hidden[c:c + 256].to(head.weight.device)).float().cpu() for c in range(0, hidden.shape[0], 256)]
        logits = torch.cat(chunks)
        gen = model.generate(ids, max_new_tokens=32, do_sample=False, repetition_penalty=1.0)[0, ids.shape[1]:]
    n = ids.shape[1]
    pos = sorted({*range(0, n, 7 if n < 1000 else 31), n - 1})
    tensors |= {f"ids_{i}": ids[0].int().cpu(), f"pos_{i}": torch.tensor(pos, dtype=torch.int32),
                f"logits_{i}": logits[pos].contiguous(), f"argmax_{i}": logits.argmax(-1).int(),
                f"gen_{i}": gen.int().cpu()}
    for k, v in layer_out.items():
        tensors[k.replace("_", f"_{i}_", 1)] = v.contiguous()
    print(f"prompt {i}: {n} tokens, greedy: {tok.decode(gen)!r}", flush=True)
save_file(tensors, out)
