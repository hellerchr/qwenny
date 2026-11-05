"""Train a DFlash2 drafter (speculative decoding) for a Granite target on one GPU.

The drafter, as z-lab's DFlash2DraftModel and vLLM speculators' trainer define it: the target's residual stream after a few
layers goes through fc and a norm into each drafter layer's keys and values (the context, a sliding window of tokens before
the anchor). A block [anchor, mask x (block_size - 1)] goes through attention + MLP layers wrapped in grouped dynamic
convolutions, then the target's frozen output layer; each row predicts its own position. A selector reranks each row's top
candidates by its predecessor: S(a, b) = U(b) + <A(a) * H(h), B(b)>.

Trained on the target's own replies (tools/make_draft_traces.py). The target stays frozen, at int8 so that it and the drafter
fit 24 GB; Adam's moments are bf16 for the same reason, with beta2 0.99 (0.999's step is below bf16's resolution).
Loss per speculators: unary 0.1 CE + 0.9 TV against the target's distribution with weights exp(-(position - 1) / gamma), plus
the selector's CE on rows whose target token is among the candidates ("strict"). TV is what sampled acceptance measures.

Usage: python -I tools/train_dflash2.py <model dir or HF id> <traces.jsonl> <out dir> [options]
Prints EAL, the expected tokens per verification step, teacher-forced on held-out traces: greedy along the selector's path, and
under the target's sampling at temperature 1 with fixed drafts (accepted with p(d)) or with drafts drawn from the unary (1 - TV).
Writes <out dir>/config.json and model.safetensors in the layout of z-lab's checkpoints. Adapted from z-lab/dflash (MIT) and
vllm-project/speculators (Apache-2.0)."""
import argparse, json, math, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


class Int8Linear(nn.Module):
    """A frozen matrix at int8 with a scale per row and 128 columns, as the engine stores dense weights"""
    def __init__(self, w):
        super().__init__()
        rows, cols = w.shape
        w = w.float()
        s = w.view(rows, cols // 128, 128).abs().amax(-1) / 127
        s = s.clamp_min(1e-12)
        q = (w.view(rows, cols // 128, 128) / s[..., None]).round().to(torch.int8)
        self.register_buffer("q", q)
        self.register_buffer("s", s.to(torch.bfloat16))

    def forward(self, x):
        rows = self.q.shape[0]
        w = (self.q.to(torch.bfloat16) * self.s[..., None]).view(rows, -1)
        return x.to(torch.bfloat16) @ w.T


def int8_target(model):
    """The target's linear layers (and output layer) at int8; returns the model"""
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    for name in names:
        parent, _, leaf = name.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        setattr(owner, leaf, Int8Linear(getattr(owner, leaf).weight.data))
        torch.cuda.empty_cache()
    return model


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def linear(i, o, std=0.02):
    m = nn.Linear(i, o, bias=False)
    nn.init.normal_(m.weight, std=std)
    return m


class DynConv(nn.Module):
    """Two-tap grouped convolution inside a block; identity at first. x [a][block][H]"""
    def __init__(self, h, kernel=2, group=16):
        super().__init__()
        self.h, self.kernel, self.group, self.groups = h, kernel, group, h // group
        self.base_kernel = nn.Parameter(torch.zeros(2, kernel, h))
        self.base_kernel.data[:, 0] = 1
        self.kernel_projection = nn.Linear(h, 2 * kernel * self.groups, bias=False)
        nn.init.zeros_(self.kernel_projection.weight)

    def apply_kernel(self, x, delta, base):
        a, b, _ = x.shape
        blocks = x.view(a, b, self.groups, self.group)
        out = (base[0].view(self.groups, self.group) + delta[:, :, 0, :, None]) * blocks
        for tap in range(1, self.kernel):
            coef = base[tap].view(self.groups, self.group) + delta[:, tap:, tap, :, None]
            out = torch.cat([out[:, :tap], out[:, tap:] + coef * blocks[:, :-tap]], 1)
        return out.view(a, b, self.h)

    def prepare(self, x):
        k = self.kernel_projection(x).view(*x.shape[:2], 2, self.kernel, self.groups)
        return self.apply_kernel(x, k[:, :, 0], self.base_kernel[0]), k[:, :, 1]

    def finish(self, x, delta):
        return self.apply_kernel(x, delta, self.base_kernel[1])


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


class Attention(nn.Module):
    def __init__(self, h, heads, kv, dim):
        super().__init__()
        self.heads, self.kv, self.dim, self.group = heads, kv, dim, heads // kv
        self.q_proj, self.k_proj, self.v_proj = linear(h, heads * dim), linear(h, kv * dim), linear(h, kv * dim)
        self.o_proj = linear(heads * dim, h)
        self.q_norm, self.k_norm = RMSNorm(dim), RMSNorm(dim)
        self.scale = dim ** -0.5

    def context(self, ctx, cos, sin):
        """Keys and values of the context [T][H]: [T][kv][dim]"""
        t = ctx.shape[0]
        k = self.k_norm(self.k_proj(ctx).view(t, self.kv, self.dim))
        return rotate(k, cos[:, None], sin[:, None]), self.v_proj(ctx).view(t, self.kv, self.dim)

    def forward(self, x, cos, sin, k_ctx, v_ctx, visible):
        """x [a][b][H]; RoPE tables [a][b][dim]; the context visible to each anchor: visible [a][T]"""
        a, b, _ = x.shape
        c, s = cos[:, :, None], sin[:, :, None]
        q = rotate(self.q_norm(self.q_proj(x).view(a, b, self.heads, self.dim)), c, s)
        k = rotate(self.k_norm(self.k_proj(x).view(a, b, self.kv, self.dim)), c, s)
        v = self.v_proj(x).view(a, b, self.kv, self.dim)
        q = q.view(a, b, self.kv, self.group, self.dim) * self.scale
        far = torch.einsum("abgrd,tgd->abgrt", q, k_ctx).masked_fill(~visible[:, None, None, None], -math.inf)
        near = torch.einsum("abgrd,acgd->abgrc", q, k)
        p = torch.cat([far, near], -1).softmax(-1)
        t = k_ctx.shape[0]
        out = torch.einsum("abgrt,tgd->abgrd", p[..., :t], v_ctx) + torch.einsum("abgrc,acgd->abgrd", p[..., t:], v)
        return self.o_proj(out.flatten(2))


class Mlp(nn.Module):
    def __init__(self, h, ff):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = linear(h, ff), linear(h, ff), linear(ff, h)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, h, ff, heads, kv, dim):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(h), RMSNorm(h)
        self.self_attn, self.mlp = Attention(h, heads, kv, dim), Mlp(h, ff)
        self.attention_conv, self.mlp_conv = DynConv(h), DynConv(h)

    def forward(self, x, cos, sin, ctx, ctx_cos, ctx_sin, visible):
        k_ctx, v_ctx = self.self_attn.context(ctx, ctx_cos, ctx_sin)
        y, kernel = self.attention_conv.prepare(self.input_layernorm(x))
        x = x + self.attention_conv.finish(self.self_attn(y, cos, sin, k_ctx, v_ctx, visible), kernel)
        y, kernel = self.mlp_conv.prepare(self.post_attention_layernorm(x))
        return x + self.mlp_conv.finish(self.mlp(y), kernel)


class Drafter(nn.Module):
    def __init__(self, cfg, a):
        super().__init__()
        h = cfg.hidden_size
        self.h, self.block, self.top_k, self.mask = h, a.block_size, a.top_k, a.mask_token_id
        self.fc = linear(len(a.taps) * h, h)
        self.hidden_norm, self.norm = RMSNorm(h), RMSNorm(h)
        self.layers = nn.ModuleList(Layer(h, a.ff, 32, 8, 128) for _ in range(a.layers))
        v = cfg.vocab_size
        self.predecessor_codebook = nn.Parameter(torch.empty(v, a.rank).normal_(std=0.02))
        self.successor_codebook = nn.Parameter(torch.empty(v, a.rank).normal_(std=0.02))
        self.hidden_projection = linear(h, a.rank)

    def state(self):
        w = {k: v for k, v in self.state_dict().items() if k not in ("predecessor_codebook", "successor_codebook", "hidden_projection.weight")}
        w |= {"candidate_selector.predecessor_codebook": self.predecessor_codebook,
              "candidate_selector.successor_codebook": self.successor_codebook,
              "candidate_selector.hidden_projection.weight": self.hidden_projection.weight}
        return {k: v.detach().to(torch.bfloat16).cpu().contiguous() for k, v in w.items()}

    def forward(self, embed, taps, ids, doc_start, anchors, table, window, use_checkpoint):
        """The block's output states [a][b][H]; doc_start [T]: where each token's document starts"""
        device = ids.device
        T, A = ids.shape[0], anchors.shape[0]
        ctx = self.hidden_norm(self.fc(taps.float()))
        pos = torch.arange(T, device=device) - doc_start
        cos, sin = table(max(int(pos.max()) + self.block + 1, 1), device)
        block_pos = pos[anchors][:, None] + torch.arange(self.block, device=device)
        # the context of an anchor: its document's tokens before it, in the window
        rows = torch.arange(T, device=device)[None]
        visible = (rows < anchors[:, None]) & (rows >= doc_start[anchors][:, None]) & (rows >= anchors[:, None] - window)
        tokens = torch.full((A, self.block), self.mask, device=device)
        tokens[:, 0] = ids[anchors]
        x = embed(tokens).float()
        args = (cos[block_pos], sin[block_pos], ctx, cos[pos], sin[pos], visible)
        for layer in self.layers:
            x = checkpoint(layer, x, *args, use_reentrant=False) if use_checkpoint and self.training else layer(x, *args)
        return self.norm(x)

    def select(self, unary, hidden, predecessors, candidates):
        """unary [n][V], hidden [n][H], predecessors [n], candidates [n][k]: the candidates' scores"""
        context = self.predecessor_codebook[predecessors] * self.hidden_projection(hidden)
        return unary.gather(-1, candidates) + (context[:, None] * self.successor_codebook[candidates]).sum(-1)


def rope_table(theta, dim):
    cache = {}
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    def table(n, device):
        if n not in cache or cache[n][0].device != device:
            f = torch.outer(torch.arange(n, device=device).float(), inv.to(device))
            e = torch.cat([f, f], -1)
            cache[n] = (e.cos(), e.sin())
        return cache[n]
    return table


def tokens_of(tok, row, max_len):
    """Prompt + reply ids and where the reply starts, as the target wrote it after its generation prompt"""
    kw = dict(row["kwargs"], tools=row["tools"]) if row["tools"] else dict(row["kwargs"])
    prompt = tok.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False, **kw)
    text = row["content"]
    if row["reasoning"]:
        text = row["reasoning"] + "\n</think>\n\n" + text
    for call in row.get("tool_calls") or []:
        f = call["function"]
        params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in json.loads(f["arguments"]).items())
        text += f"\n<tool_call>\n<function={f['name']}>\n{params}</function>\n</tool_call>"
    head = tok.encode(prompt, add_special_tokens=False)
    return (head + tok.encode(text + "<|im_end|>", add_special_tokens=False))[:max_len], len(head)


def packed(data, length, rng):
    """Documents packed to about `length` tokens: ids, the start of each token's document, positions allowed as anchors"""
    ids, starts, anchors = [], [], []
    while len(ids) < length:
        doc, reply = rng.choice(data)
        at = len(ids)
        ids += doc
        starts += [at] * len(doc)
        anchors += range(at + max(reply - 1, 0), at + len(doc) - 8)
    return ids, starts, anchors


class Adam:
    """AdamW with moments at bf16 (half the memory); beta2 is 0.99 because of that"""
    def __init__(self, params, lr, wd, betas=(0.9, 0.99), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr, self.wd, self.betas, self.eps, self.t = lr, wd, betas, eps, 0
        self.m = [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.params]
        self.v = [torch.zeros_like(p, dtype=torch.bfloat16) for p in self.params]

    @torch.no_grad()
    def step(self, scale):
        self.t += 1
        b1, b2 = self.betas
        lr = self.lr * scale
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.lerp_(g.to(m.dtype), 1 - b1)
            v.lerp_(g.pow(2).to(v.dtype), 1 - b2)
            hat_m = m.float() / (1 - b1 ** self.t)
            hat_v = v.float() / (1 - b2 ** self.t)
            if self.wd and p.ndim > 1:
                p.mul_(1 - lr * self.wd)
            p.add_(hat_m / (hat_v.sqrt() + self.eps), alpha=-lr)
            p.grad = None


class Stats:
    def __init__(self, block):
        self.block, self.loss, self.n, self.steps = block, 0.0, 0, 0
        self.tv, self.top, self.recall, self.sel, self.count = [torch.zeros(block) for _ in range(5)]
        self.eal, self.eal_u, self.eal_p, self.eal_tv, self.blocks = 0.0, 0.0, 0.0, 0.0, 0

    def line(self):
        c = self.count.clamp_min(1)
        pos = " ".join(f"{x:.2f}" for x in (self.sel / c)[1:].tolist())
        b = max(self.blocks, 1)
        return (f"EAL greedy {self.eal / b:.3f} (unary {self.eal_u / b:.3f}), sampled {self.eal_p / b:.3f} (fixed draft, p(d)) "
                f"{self.eal_tv / b:.3f} (unary drawn, 1-TV) | selector by position {pos} "
                f"| 1-TV {(self.tv / c)[1:].mean():.3f} recall {(self.recall / c)[1:].mean():.3f} | loss {self.loss / max(self.steps, 1):.3f}")


def compute(model, target, head, embed, batch, a, stats, table):
    ids, starts, anchors = batch["ids"], batch["starts"], batch["anchors"]
    block = a.block_size
    hidden = model(embed, batch["taps"], ids, starts, anchors, table, a.window, a.checkpoint)
    rows = hidden[:, 1:].reshape(-1, hidden.shape[-1])
    at = (anchors[:, None] + torch.arange(block - 1, device=ids.device)).reshape(-1)  # the target's row before each token
    with torch.no_grad():
        p = head(batch["last"][at]).float().softmax(-1)
    unary = head(rows).float()
    log_q = unary.log_softmax(-1)
    weights = torch.exp(-torch.arange(block - 1, device=ids.device).float() / a.gamma).repeat(anchors.shape[0])
    argmax = p.argmax(-1)
    ce = F.nll_loss(log_q, argmax, reduction="none")
    tv = 0.5 * (p - log_q.exp()).abs().sum(-1)
    unary_loss = (weights * (a.ce * ce + (1 - a.ce) * tv)).sum() / weights.sum()
    # the selector, teacher-forced on the true previous tokens
    tokens = ids[(anchors[:, None] + torch.arange(block, device=ids.device))]
    predecessors = tokens[:, :-1].reshape(-1)
    with torch.no_grad():
        candidates = unary.topk(a.top_k, -1).indices
        hit = candidates == argmax[:, None]
        contains = hit.any(-1)
        gold = hit.float().argmax(-1)
    scores = model.select(unary, rows, predecessors, candidates)
    sel_ce = F.cross_entropy(scores, gold, reduction="none") * contains
    loss = unary_loss + (weights * sel_ce).sum() / weights.sum()
    with torch.no_grad():
        chosen = candidates.gather(-1, scores.argmax(-1, keepdim=True))[:, 0]
        correct = (chosen == argmax).view(-1, block - 1)
        unary_ok = (unary.argmax(-1) == argmax).view(-1, block - 1)
        for i in range(block - 1):
            stats.sel[i + 1] += correct[:, i].sum().item()
            stats.tv[i + 1] += (1 - tv).view(-1, block - 1)[:, i].sum().item()
            stats.recall[i + 1] += contains.view(-1, block - 1)[:, i].sum().item()
            stats.count[i + 1] += correct.shape[0]
        stats.eal += (correct.float().cumprod(1).sum(1) + 1).sum().item()
        stats.eal_u += (unary_ok.float().cumprod(1).sum(1) + 1).sum().item()
        taken = p.gather(-1, chosen[:, None])[:, 0].view(-1, block - 1)
        stats.eal_p += (taken.cumprod(1).sum(1) + 1).sum().item()
        stats.eal_tv += ((1 - tv).view(-1, block - 1).cumprod(1).sum(1) + 1).sum().item()
        stats.blocks += correct.shape[0]
        stats.loss += loss.item()
        stats.steps += 1
    return loss


def target_batch(target, taps_at, data, a, rng, device):
    ids, starts, anchors = packed(data, a.seq_len, rng)
    ids = torch.tensor(ids[:a.seq_len + 4096], device=device)
    starts = torch.tensor(starts[:ids.shape[0]], device=device)
    anchors = torch.tensor([x for x in anchors if x < ids.shape[0] - a.block_size], device=device)
    saved = []
    hooks = [target.model.layers[i].register_forward_hook(lambda m, i, o: saved.append((o[0] if isinstance(o, tuple) else o).detach())) for i in taps_at]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        # documents are packed by chunks of their own: a forward per document, not one over all of them
        last, parts = [], []
        for s in sorted(set(starts.tolist())):
            piece = ids[starts == s]
            saved.clear()
            last.append(target.model(input_ids=piece[None], use_cache=False).last_hidden_state[0].to(torch.bfloat16))
            parts.append(torch.cat([x[0] for x in saved], -1).to(torch.bfloat16))
    for h in hooks:
        h.remove()
    return {"ids": ids, "starts": starts, "anchors": anchors, "last": torch.cat(last), "taps": torch.cat(parts)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("traces")
    p.add_argument("out")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--anchors", type=int, default=384, help="blocks per drafter step")
    p.add_argument("--inner", type=int, default=4, help="drafter steps per target forward")
    p.add_argument("--max-doc", type=int, default=4096)
    p.add_argument("--heldout", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--layers", type=int, default=5)
    p.add_argument("--ff", type=int, default=8192, help="MLP width, a multiple of 256: less than the target's fits 24 GB and drafts faster")
    p.add_argument("--taps", type=int, nargs="+", default=[1, 10, 19, 28, 37])
    p.add_argument("--mask-token-id", type=int, default=100264, help="an unused token: <|unused_11|>")
    p.add_argument("--rank", type=int, default=256)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--ce", type=float, default=0.1)
    p.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--init")
    a = p.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = AutoConfig.from_pretrained(a.model)
    target = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cpu").eval().requires_grad_(False)
    int8_target(target)
    target = target.cuda()
    head, embed = target.lm_head, target.model.embed_tokens
    model = Drafter(cfg, a).cuda()
    if a.init:
        w = load_file(a.init)
        own = {k.replace("candidate_selector.", ""): v.float() for k, v in w.items()}
        print(model.load_state_dict(own, strict=False), flush=True)
    print(f"drafter {sum(x.numel() for x in model.parameters()) / 1e6:.0f}M parameters; {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    rows = [json.loads(line) for line in open(a.traces)]
    random.Random(0).shuffle(rows)
    data = [tokens_of(tok, r, a.max_doc) for r in rows]
    data = [d for d in data if len(d[0]) - d[1] > a.block_size + 1]
    held, train = data[:a.heldout], data[a.heldout:]
    print(f"{len(train)} traces ({sum(len(d[0]) for d in train)} tokens), {len(held)} held out", flush=True)
    theta = cfg.rope_parameters["rope_theta"]
    table = rope_table(theta, 128)
    config = {"architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "attention_bias": False, "dtype": "bfloat16", "head_dim": 128,
              "hidden_act": "silu", "hidden_size": cfg.hidden_size, "intermediate_size": a.ff,
              "num_attention_heads": 32, "num_hidden_layers": a.layers, "num_key_value_heads": 8, "num_target_layers": cfg.num_hidden_layers,
              "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": theta, "rope_type": "default"}, "sliding_window": a.window,
              "use_sliding_window": True, "layer_types": ["sliding_attention"] * a.layers, "vocab_size": cfg.vocab_size,
              "dflash_config": {"block_size": a.block_size, "conv_group_size": 16, "conv_kernel_size": 2, "mask_token_id": a.mask_token_id,
                                "selector_rank": a.rank, "selector_top_k": a.top_k, "target_layer_ids": a.taps}}
    json.dump(config, open(out / "config.json", "w"), indent=1)

    def evaluate():
        model.eval()
        stats, er = Stats(a.block_size), random.Random(1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(8):
                b = target_batch(target, a.taps, held, a, er, "cuda")
                for _ in range(2):
                    pick = b["anchors"][torch.randperm(len(b["anchors"]), device="cuda")[:a.anchors]]
                    compute(model, target, head, embed, b | {"anchors": pick}, a, stats, table)
        model.train()
        return stats

    print("start:", evaluate().line(), flush=True)
    opt = Adam(model.parameters(), a.lr, a.wd)
    stats, t0, seen, best, batch = Stats(a.block_size), time.time(), 0, 0.0, None
    for step in range(1, a.steps + 1):
        if batch is None or (step - 1) % a.inner == 0:
            batch = target_batch(target, a.taps, train, a, rng, "cuda")
            seen += batch["ids"].shape[0]
            order = torch.randperm(len(batch["anchors"]), device="cuda")
        slice_at = ((step - 1) % a.inner) * a.anchors
        pick = order[slice_at:slice_at + a.anchors]
        if len(pick) < 8:
            batch = None
            continue
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = compute(model, target, head, embed, batch | {"anchors": batch["anchors"][pick]}, a, stats, table)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        progress = step / a.steps
        opt.step(min(1.0, step / 200) * (1 - progress))
        if step % 50 == 0:
            rate = stats.blocks / (time.time() - t0)
            print(f"step {step} ({rate:.0f} blocks/s, {seen} target tokens, {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB): {stats.line()}", flush=True)
            stats, t0 = Stats(a.block_size), time.time()
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate()
            print(f"held out, step {step}: {ev.line()}", flush=True)
            state = model.state()
            save_file(state, str(out / "last.safetensors"))
            score = ev.eal / max(ev.blocks, 1)
            if score > best:
                best = score
                save_file(state, str(out / "model.safetensors"))
                print(f"best so far: {best:.3f}", flush=True)


if __name__ == "__main__":
    main()
