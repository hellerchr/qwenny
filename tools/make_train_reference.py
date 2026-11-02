"""References for `qwenny datacheck` and `traincheck`: what Transformers + PEFT compute with Unsloth's loss masks.

  python make_train_reference.py data <model dir> <data dir> <out.safetensors>
      Token ids and loss masks of train.jsonl and dev.jsonl (chat template with enable_thinking=False; replies marked
      as Unsloth's train_on_responses_only: from each assistant header to the next user header). CPU only.
  python make_train_reference.py grads <model dir> <data dir> <out.safetensors> [docs] [layers] [--weights=<dir>]
      fp32 model + PEFT LoRA (r=16, alpha=32, dropout 0, every linear layer of the language model, like Unsloth)
      with A and B both random, on the first `docs` training examples (default 3): loss (mean over all reply
      tokens), NLL per token, the initial LoRA weights, their gradients, and the weights after one AdamW step (lr
      2e-4, grad clip 1.0). `layers`: only the model's first layers (a Qwen3.5 4B in fp32 does not fit 16 GB;
      `traincheck` reads the count from the reference). --weights=<dir>: a compressed-tensors checkpoint (int4 / int8, symmetric, grouped) whose
      weights are dequantized exactly into the model (<model dir> may be that checkpoint: its config is used without
      the quantization), the reference for training on it.
"""
import json, sys
import torch
from safetensors.torch import save_file

# every linear layer of the language model, as Unsloth's get_peft_model (AutoModelForCausalLM loads the text model
# alone, "model.layers.N"; the tensors are saved under the multimodal checkpoint's names, as Unsloth's adapters)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                "gate_proj", "up_proj", "down_proj"]


def encode(tok, messages):
    ids = tok.apply_chat_template(messages, tokenize=True, return_dict=True, enable_thinking=False)["input_ids"]
    # Unsloth's train_on_responses_only (as ft-kit trains Qwen): from after each assistant header to the next user header
    reply = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    user = tok.encode("<|im_start|>user\n", add_special_tokens=False)
    at = lambda seq, i: ids[i:i + len(seq)] == seq
    mask, on, i = [0] * len(ids), False, 0
    while i < len(ids):
        if at(reply, i):
            on, i = True, i + len(reply)
            continue
        if at(user, i):
            on = False
        mask[i] = int(on)
        i += 1
    return ids, mask


def load_rows(data, split):
    with open(f"{data}/{split}.jsonl") as f:
        return [json.loads(l)["messages"] for l in f if l.strip()]


def data_ref(model, data, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    t = {}
    for split in ["train", "dev"]:
        ids, mask, starts = [], [], [0]
        for msgs in load_rows(data, split):
            i, m = encode(tok, msgs)
            ids += i
            mask += m
            starts.append(len(ids))
        t[f"{split}_ids"] = torch.tensor(ids, dtype=torch.int32)
        t[f"{split}_mask"] = torch.tensor(mask, dtype=torch.int32)
        t[f"{split}_starts"] = torch.tensor(starts, dtype=torch.int32)
        print(f"{split}: {len(starts) - 1} examples, {len(ids)} tokens, {sum(mask)} in the loss")
    save_file(t, out)


def dequantized(qdir, dest):
    """Fill the tensors dest(name) (a list, empty: not wanted) from a compressed-tensors checkpoint, matrices
    dequantized: w = (q - 2^(bits-1)) * scale per group (symmetric pack-quantized), in chunks of rows (little memory)"""
    import glob
    from safetensors import safe_open
    for f in sorted(glob.glob(f"{qdir}/*.safetensors")):
        with safe_open(f, "pt") as st:
            for n in st.keys():
                if n.endswith(".weight_packed"):
                    base = n.removesuffix(".weight_packed")
                    targets = dest(base + ".weight")
                    if not targets:
                        continue
                    packed = st.get_tensor(n)
                    rows, cols = st.get_tensor(base + ".weight_shape").tolist()
                    bits = packed.shape[1] * 32 // cols
                    shifts = torch.arange(0, 32, bits, device="cuda")
                    scale = st.get_tensor(base + ".weight_scale").float()
                    for r0 in range(0, rows, 4096):
                        q = packed[r0:r0 + 4096].cuda().to(torch.int64) & 0xFFFFFFFF
                        v = ((q[:, :, None] >> shifts) & ((1 << bits) - 1)).reshape(q.shape[0], cols).float() - (1 << (bits - 1))
                        v.mul_(scale[r0:r0 + 4096].cuda().repeat_interleave(cols // scale.shape[1], dim=1))
                        for t in targets:
                            t.view(rows, cols)[r0:r0 + q.shape[0]].copy_(v)
                elif not n.endswith((".weight_scale", ".weight_shape", ".weight_zero_point")):
                    for t in dest(n):
                        t.copy_(st.get_tensor(n).reshape(t.shape))


def load_model(model, layers, weights=None):
    """The tokenizer and the fp32 text model on the GPU (its first `layers` layers; 0: all), the weights of `weights`
    (a compressed-tensors checkpoint) dequantized into it, or from `model`"""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    tc = config.text_config
    if layers:
        tc.num_hidden_layers = layers
        tc.layer_types = tc.layer_types[:layers]
    if weights:
        # the architecture with random weights, then the checkpoint's dequantized: the text model's names are
        # "model.layers.N" for "model.language_model.layers.N"
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        with torch.device("cuda"):
            m = AutoModelForCausalLM.from_config(tc, dtype=torch.float32,
                                                 attn_implementation="eager")
        params = dict(m.named_parameters())
        ck = lambda n: n.replace("model.layers.", "model.language_model.layers.").replace(
            "model.embed_tokens", "model.language_model.embed_tokens").replace("model.norm.", "model.language_model.norm.")
        by_ck = {}
        for n in params:
            by_ck.setdefault(ck(n), []).append(n)
        tied = getattr(tc, "tie_word_embeddings", False)
        if tied and "lm_head.weight" in params:  # named_parameters lists a tied output layer once, as the embedding
            by_ck["model.language_model.embed_tokens.weight"].append("lm_head.weight")
        done = set()
        def dest(name):
            done.update(by_ck.get(name, []))
            return [params[n].data for n in by_ck.get(name, [])]
        with torch.no_grad():
            dequantized(weights, dest)
        missing = sorted(set(params) - done)
        assert not missing, f"not in {weights}: {missing[:5]}"
        print(f"weights from {weights} ({len(done)} tensors, dequantized)")
    else:
        m = AutoModelForCausalLM.from_pretrained(model, config=config, dtype=torch.float32, device_map="cuda",
                                                 trust_remote_code=True, attn_implementation="eager")
    return tok, m


def grads_ref(model, data, out, docs, layers, weights=None):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    tok, m = load_model(model, layers, weights)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                     task_type="CAUSAL_LM"))
    params = {n: p for n, p in m.named_parameters() if p.requires_grad}
    with torch.no_grad():  # B is zero after PEFT's init: give it the size a trained adapter has
        for n, p in params.items():
            if "lora_B" in n:
                p.normal_(0, 0.01)
    t = {f"init.{n}": p.detach().cpu().clone() for n, p in params.items()}

    encoded = [encode(tok, msgs) for msgs in load_rows(data, "train")[:docs]]
    total = sum(sum(mk[1:]) for _, mk in encoded)  # a token is predicted from the one before it
    loss_sum = 0.0
    for d, (ids, mask) in enumerate(encoded):
        x = torch.tensor([ids], device="cuda")
        logits = m(input_ids=x).logits[0, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none")
        keep = torch.tensor(mask[1:], device="cuda", dtype=torch.float32)
        loss = (nll * keep).sum() / total
        loss.backward()
        loss_sum += loss.item()
        t[f"ids_{d}"] = torch.tensor(ids, dtype=torch.int32)
        t[f"mask_{d}"] = torch.tensor(mask, dtype=torch.int32)
        t[f"nll_{d}"] = nll.detach().cpu()
    t["loss"] = torch.tensor([loss_sum])
    t["layers"] = torch.tensor([layers], dtype=torch.int32)
    for n, p in params.items():
        t[f"grad.{n}"] = p.grad.detach().cpu().clone()
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
    t["grad_norm"] = torch.tensor([norm.item()])
    opt = torch.optim.AdamW(list(params.values()), lr=2e-4, weight_decay=0.0)
    opt.step()
    for n, p in params.items():
        t[f"step1.{n}"] = p.detach().cpu().clone()
    t = {k.replace("model.model.layers.", "model.model.language_model.layers."): v for k, v in t.items()}
    print(f"{docs} examples, {total} loss tokens: loss {loss_sum:.6f}, grad norm {norm.item():.4f}, "
          f"{len(params)} LoRA tensors")
    save_file(t, out)


if __name__ == "__main__":
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--"))
    sys.argv = [a for a in sys.argv if not a.startswith("--")]
    mode, model, data, out = sys.argv[1:5]
    if mode == "data":
        data_ref(model, data, out)
    else:
        grads_ref(model, data, out, int(sys.argv[5]) if len(sys.argv) > 5 else 3, int(sys.argv[6]) if len(sys.argv) > 6 else 0,
                  opts.get("weights"))
