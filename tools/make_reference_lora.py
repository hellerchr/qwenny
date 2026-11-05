"""Reference logits of Qwen3.5 with a LoRA adapter from Transformers + PEFT in float32, for `qwenny check --adapter`
and `qwenny adaptercheck` (the adapter served unmerged, per request).
Usage: python make_reference_lora.py <model dir or HF id> <fjordhaus dev.jsonl> <out.safetensors> <adapter dir>
                                     [--weights=<compressed-tensors checkpoint>]
The adapter: rank 16, alpha 32, every linear layer of the language model (as `qwenny train`), A as PEFT initializes
it and B random (PEFT's B is zero), seed 0; written to <adapter dir> in PEFT's format (f32, the checkpoint's tensor
names, as `qwenny train` saves) when it is not there, else loaded from it (one adapter for several models).
--weights: the weights of a compressed-tensors int4 / int8 checkpoint, dequantized exactly into the model
(make_train_reference.py), e.g. RedHatAI/Qwen3.5-4B-quantized.w4a16 into Qwen/Qwen3.5-4B's architecture.
Per prompt i (those of make_reference.py): ids_i, pos_i, and with the adapter logits_i (at pos_i), argmax_i (every
position), gen_i (32 greedy tokens); without it the same as base_logits_i, base_argmax_i, base_gen_i."""
import json, os
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from make_train_reference import TARGETS, dequantized
from refutil import chunked, cli, positions, prompts

args, opts = cli(__doc__)
model_dir, tests, out, adapter = args
weights = opts.get("weights")
tok = AutoTokenizer.from_pretrained(model_dir)
# in host memory first (weights and the adapter are set there: offloaded parameters are not real tensors), then
# dispatched (fp32 does not fit the 16 GB GPU: the rest of the layers stay on the host and run on the GPU in turn)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
model.eval()
# the text model's names "model.layers.N" are the checkpoint's "model.language_model.layers.N"
ck = lambda n: n.replace("model.layers.", "model.language_model.layers.").replace(
    "model.embed_tokens", "model.language_model.embed_tokens").replace("model.norm.", "model.language_model.norm.")
if weights:
    params = dict(model.named_parameters())
    by_ck = {}
    for n in params:
        by_ck.setdefault(ck(n), []).append(n)
    done = set()
    def dest(name):
        done.update(by_ck.get(name, []))
        return [params[n].data for n in by_ck.get(name, [])]
    with torch.no_grad():
        dequantized(weights, dest)
    missing = sorted(set(params) - done - {"lm_head.weight"})
    assert not missing, f"not in {weights}: {missing[:5]}"
    print(f"weights from {weights} ({len(done)} tensors, dequantized)")

torch.manual_seed(0)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                         task_type="CAUSAL_LM"))
model.eval()
lora = {n: p for n, p in model.named_parameters() if "lora_" in n}
# PEFT's "base_model.model.model.layers.3.mlp.up_proj.lora_A.default.weight" in the file as the checkpoint names it
fname = lambda n: n.replace(".default", "").replace("base_model.model.model.layers.", "base_model.model.model.language_model.layers.")
path = os.path.join(adapter, "adapter_model.safetensors")
with torch.no_grad():
    if os.path.exists(path):
        saved = load_file(path)
        assert len(saved) == len(lora), f"{path}: {len(saved)} tensors, the model has {len(lora)}"
        for n, p in lora.items():
            p.copy_(saved[fname(n)])
        print(f"adapter from {adapter}")
    else:
        for n, p in lora.items():
            if "lora_B" in n:  # zero after PEFT's init: random, of the size a trained adapter has
                p.normal_(0, 0.01)
        os.makedirs(adapter, exist_ok=True)
        save_file({fname(n): p.detach().float().cpu().contiguous() for n, p in lora.items()}, path)
        cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": model_dir, "r": 16,
               "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": sorted(TARGETS), "bias": "none",
               "fan_in_fan_out": False, "inference_mode": True, "use_dora": False, "use_rslora": False,
               "modules_to_save": None, "rank_pattern": {}, "alpha_pattern": {}}
        with open(os.path.join(adapter, "adapter_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"adapter written to {adapter} ({len(lora)} tensors)")

from accelerate import dispatch_model, infer_auto_device_map
inner = model.base_model.model  # the causal LM, its layers wrapped with the LoRA
dmap = infer_auto_device_map(inner, max_memory={0: "9GiB", "cpu": "40GiB"},
                             no_split_module_classes=[type(inner.model.layers[0]).__name__])
dispatch_model(inner, dmap)

tensors = {}
dev = "cuda"


def run(ids):
    """logits at every position (the head 256 rows at a time) and 32 greedy tokens"""
    with torch.no_grad():
        hidden = inner.model(input_ids=ids).last_hidden_state[0]
        head = inner.get_output_embeddings()
        logits = chunked(lambda x: head(x.to(head.weight.device)), hidden)
        gen = model.generate(input_ids=ids, max_new_tokens=32, do_sample=False, repetition_penalty=1.0)[0, ids.shape[1]:]
    return logits, gen.int().cpu()


for i, msgs in enumerate(prompts(tests)):
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                                  return_dict=True)["input_ids"].to(dev)
    n = ids.shape[1]
    pos = positions(n)
    tensors |= {f"ids_{i}": ids[0].int().cpu(), f"pos_{i}": torch.tensor(pos, dtype=torch.int32)}
    for prefix, on in [("", True), ("base_", False)]:
        if on:
            logits, gen = run(ids)
        else:
            with model.disable_adapter():
                logits, gen = run(ids)
        tensors |= {f"{prefix}logits_{i}": logits[pos].contiguous(), f"{prefix}argmax_{i}": logits.argmax(-1).int(),
                    f"{prefix}gen_{i}": gen}
        print(f"prompt {i}: {n} tokens, {'adapter' if on else 'base'}, greedy: {tok.decode(gen)!r}", flush=True)
    d = (tensors[f"logits_{i}"] - tensors[f"base_logits_{i}"]).abs().mean().item()
    print(f"prompt {i}: the adapter changes the logits by {d:.3f} on average", flush=True)
save_file(tensors, out)
