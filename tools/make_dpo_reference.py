"""Reference for `qwenny dpocheck`: what PyTorch + PEFT compute for preference training (RPO: DPO + NLL of the chosen
response, Pang et al. 2024), cross-checked against TRL's DPOTrainer.

  python make_dpo_reference.py <model dir> <data dir> <out.safetensors> [pairs] [layers] [--weights=<dir>]
                               [--beta=0.1] [--nll=1.0] [--no-trl]

Pairs from the first rows of <data dir>/train.jsonl (rows {"messages": [system, user, assistant]}): pair i has row
i's prompt, row i's reply as chosen and row i+1's as rejected; pair 1 has a multi-turn prompt (row 1's conversation and
a new user turn; its earlier reply is context, not loss), pair 2 renders with thinking (reasoning_content,
chat_template_kwargs {"enable_thinking": true}). The rows go into the reference's metadata ("pairs"), so dpocheck
encodes them itself and compares its token ids and masks.

Each response is `prompt + [response]` with the chat template, in the loss from after its "<|im_start|>assistant\\n"
to the end (as SFT masks a reply, Unsloth's train_on_responses_only). fp32 model (its first `layers` layers, default
4; int4 weights of --weights dequantized exactly, as make_train_reference.py does) + PEFT LoRA r=16 alpha=32 on every
linear layer: a reference adapter ("ref.": PEFT's A, B random) and the policy (ref. plus noise, "init."). Saved: the
reference's and the policy's summed log-probabilities of every response, the mean RPO loss over the pairs and its
parts, the LoRA gradients, and the weights after one AdamW step (lr 2e-4, clip 1.0). TRL: DPOTrainer with
loss_type ["sigmoid", "sft"], loss_weights [1, alpha] (TRL's RPO) and a "ref" copy of the starting adapter computes
the loss and gradients of each pair again; its loss is saved ("trl_loss") and both differences are printed.
"""
import json, sys
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from make_train_reference import TARGETS, load_model

HEADER = "<|im_start|>assistant\n"


def make_pairs(data, n):
    rows = [json.loads(l)["messages"] for l in open(f"{data}/train.jsonl") if l.strip()][:n + 2]
    pairs = []
    for i in range(n):
        prompt, chosen, rejected = rows[i][:-1], rows[i][-1], rows[i + 1][-1]
        pair = {"messages": prompt, "chosen": {"role": "assistant", "content": chosen["content"]},
                "rejected": {"role": "assistant", "content": rejected["content"]}}
        if i == 1:  # multi-turn: the earlier reply is context
            pair["messages"] = rows[i] + [{"role": "user", "content": "Bitte kürzer, in zwei Sätzen."}]
            pair["chosen"]["content"] = chosen["content"].split("\n\n")[0]
        if i == 2:  # thinking
            pair["chat_template_kwargs"] = {"enable_thinking": True}
            pair["chosen"]["reasoning_content"] = "Die Auszüge im Kontext beantworten die Frage; ich halte mich an sie."
            pair["rejected"]["reasoning_content"] = "Ich antworte aus dem Gedächtnis."
        pairs.append(pair)
    return pairs


def encode(tok, pair, key):
    kw = {"enable_thinking": False, **pair.get("chat_template_kwargs", {})}
    ids = tok.apply_chat_template(pair["messages"] + [pair[key]], tokenize=True, return_dict=True, **kw)["input_ids"]
    head = tok.encode(HEADER, add_special_tokens=False)
    k = max(i for i in range(len(ids)) if ids[i:i + len(head)] == head) + len(head)
    return ids, [0] * k + [1] * (len(ids) - k), k


def logp(m, ids, mask):
    """summed log-probability of the masked tokens, and their count"""
    x = torch.tensor([ids], device="cuda")
    nll = F.cross_entropy(m(input_ids=x).logits[0, :-1].float(), x[0, 1:], reduction="none")
    keep = torch.tensor(mask[1:], device="cuda", dtype=torch.float32)
    return -(nll * keep).sum(), int(keep.sum().item())


def main(model, data, out, n, layers, weights, beta, alpha, trl):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(0)
    tok, m = load_model(model, layers, weights)
    m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM"))
    params = {n_: p for n_, p in m.named_parameters() if p.requires_grad}
    with torch.no_grad():  # the reference: PEFT's A, a random B; the policy: that plus noise
        for n_, p in params.items():
            if "lora_B" in n_:
                p.normal_(0, 0.01)
        ref = {n_: p.detach().clone() for n_, p in params.items()}
        pol = {n_: p + 0.5 * p.std() * torch.randn_like(p) for n_, p in ref.items()}
    set_params = lambda v: [params[n_].data.copy_(v[n_]) for n_ in params]

    pairs = make_pairs(data, n)
    enc = [(encode(tok, p, "chosen"), encode(tok, p, "rejected")) for p in pairs]
    t = {}
    for i, ((ci, cm, ck), (ri, rm, rk)) in enumerate(enc):
        assert ci[:ck] == ri[:rk], f"pair {i}: the prompts' tokens differ"
        t[f"ids_c_{i}"], t[f"mask_c_{i}"] = torch.tensor(ci, dtype=torch.int32), torch.tensor(cm, dtype=torch.int32)
        t[f"ids_r_{i}"], t[f"mask_r_{i}"] = torch.tensor(ri, dtype=torch.int32), torch.tensor(rm, dtype=torch.int32)

    set_params(ref)
    with torch.no_grad():
        rf = [(logp(m, ci, cm)[0].item(), logp(m, ri, rm)[0].item()) for (ci, cm, _), (ri, rm, _) in enc]
    set_params(pol)
    lp, loss_sum, dpo_sum, nll_sum, margin_sum = [], 0.0, 0.0, 0.0, 0.0
    for ((ci, cm, _), (ri, rm, _)), (rc, rr) in zip(enc, rf):
        (lc, nc), (lr_, _) = logp(m, ci, cm), logp(m, ri, rm)
        z = beta * ((lc - rc) - (lr_ - rr))
        dpo, nll = -F.logsigmoid(z), -lc / nc
        loss = dpo + alpha * nll
        (loss / len(enc)).backward()
        lp.append((lc.item(), lr_.item()))
        loss_sum, dpo_sum, nll_sum, margin_sum = loss_sum + loss.item(), dpo_sum + dpo.item(), nll_sum + nll.item(), margin_sum + z.item()
    k = len(enc)
    grads = {n_: p.grad.detach().clone() for n_, p in params.items()}
    print(f"{k} pairs, beta {beta}, alpha {alpha}: loss {loss_sum / k:.6f} (DPO {dpo_sum / k:.6f}, NLL {nll_sum / k:.6f}), "
          f"margin {margin_sum / k:+.6f}")
    for i, ((c, r), (a, b)) in enumerate(zip(lp, rf)):
        print(f"  pair {i}: logp chosen {c:.3f} (ref {a:.3f}), rejected {r:.3f} (ref {b:.3f}), "
              f"{sum(enc[i][0][1])} / {sum(enc[i][1][1])} response tokens")

    if trl:
        t["trl_loss"] = torch.tensor([trl_check(m, tok, params, ref, pol, enc, beta, alpha, grads)])
        for p in params.values():
            p.grad = None
        set_params(pol)
        for n_, p in params.items():
            p.grad = grads[n_].clone()
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
    opt = torch.optim.AdamW(list(params.values()), lr=2e-4, weight_decay=0.0)
    opt.step()
    for n_ in params:
        t[f"ref.{n_}"], t[f"init.{n_}"] = ref[n_].cpu(), pol[n_].cpu()
        t[f"grad.{n_}"], t[f"step1.{n_}"] = grads[n_].cpu(), params[n_].detach().cpu().clone()
    t["ref_logp"] = torch.tensor([x for c in rf for x in c])
    t["logp"] = torch.tensor([x for c in lp for x in c])
    for key, v in [("loss", loss_sum / k), ("dpo", dpo_sum / k), ("nll", nll_sum / k), ("margin", margin_sum / k),
                   ("beta", beta), ("alpha", alpha), ("grad_norm", norm.item())]:
        t[key] = torch.tensor([v], dtype=torch.float32)
    t["layers"] = torch.tensor([layers], dtype=torch.int32)
    t = {k_.replace("model.model.layers.", "model.model.language_model.layers."): v.contiguous() for k_, v in t.items()}
    save_file(t, out, metadata={"pairs": "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pairs)})
    print(f"grad norm {norm.item():.5f}, {len(params)} LoRA tensors -> {out}")


def trl_check(m, tok, params, ref, pol, enc, beta, alpha, grads):
    """TRL's DPOTrainer (its RPO: loss_type sigmoid + sft) from the reference adapter, which it copies to a frozen
    "ref" adapter, then the policy's loss and gradients of each pair (batch of one, the mean over pairs)"""
    import tempfile
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    for n_ in params:
        params[n_].data.copy_(ref[n_])
    dummy = Dataset.from_list([{"prompt": [{"role": "user", "content": "x"}], "chosen": [{"role": "assistant", "content": "a"}],
                                "rejected": [{"role": "assistant", "content": "b"}]}])
    args = DPOConfig(output_dir=tempfile.mkdtemp(), beta=beta, loss_type=["sigmoid", "sft"], loss_weights=[1.0, alpha],
                     bf16=False, fp16=False, per_device_train_batch_size=1, report_to="none", gradient_checkpointing=False,
                     precompute_ref_log_probs=False)
    tr = DPOTrainer(model=m, args=args, train_dataset=dummy, processing_class=tok)
    for n_ in params:
        params[n_].data.copy_(pol[n_])
        params[n_].grad = None
    total = 0.0
    for (ci, _, ck), (ri, _, rk) in enc:
        batch = tr.data_collator([{"prompt_ids": ci[:ck], "chosen_ids": ci[ck:], "rejected_ids": ri[rk:]}])
        batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in batch.items()}
        loss = tr._compute_loss(m, batch, False)
        loss = loss[0] if isinstance(loss, tuple) else loss
        (loss / len(enc)).backward()
        total += loss.item()
    worst = max(((params[n_].grad - g).norm() / g.norm()).item() for n_, g in grads.items())
    print(f"TRL DPOTrainer: loss {total / len(enc):.6f}; gradients: worst relative difference to ours {worst:.2e}")
    # TRL's frozen "ref" adapter stays in the model; drop it so it does not change what follows
    m.delete_adapter("ref")
    return total / len(enc)


if __name__ == "__main__":
    opts = dict((a[2:].split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    model, data, out = argv[:3]
    main(model, data, out, int(argv[3]) if len(argv) > 3 else 3, int(argv[4]) if len(argv) > 4 else 4, opts.get("weights"),
         float(opts.get("beta", 0.1)), float(opts.get("nll", 1.0)), "no-trl" not in opts)
