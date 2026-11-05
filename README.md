# qwenny

An inference server and LoRA trainer for Qwen3.5 / Qwen3.8 models on a single NVIDIA GPU. The host code is Rust,
the model runs in CUDA kernels compiled with `nvcc`, and the server speaks the OpenAI chat completions API. It is
built for running the 27B model with a 250k-token context on one 24 GB GPU.

- Output follows the model: speculative drafts are verified against the full model, sampling with drafts keeps the
  model's distribution, and checks compare the results with PyTorch / Transformers / PEFT references.
- int4 weights (symmetric or with zero points) run on Marlin kernels; the KV cache is int8; decode steps replay as
  CUDA graphs; the checkpoint's MTP layer drafts four tokens per step by default.
- One binary: `serve` (API and an optional web page), `train` (LoRA), `merge`, `bench` and the checks.

## Performance

Swift 1.5 (Qwen3.8-27B, int4) on one RTX 3090 at 350 W, default settings (a context of 253,952 tokens), one stream:

| Workload | Speed |
|---|---|
| Code (a Java Spring Boot CRUD app), thinking off | 188–202 tok/s |
| 8 mixed prompts (prose, math, code, JSON), thinking on (`tools/decode_bench.py`) | ~110 tok/s |
| Prefill, 2,048-token chunks | ~1,860 tok/s |
| Prefill, a 32k-token context | ~1,660 tok/s |

Decode speed depends on the text: the more predictable it is (code, structured output), the more drafts are
accepted per step. Tokens per second are counted after the first token.

## Models

`model_type` qwen3_5 (three of four layers Gated DeltaNet, every fourth gated attention): Qwen3.5-4B, Qwen3.5-9B,
Qwen3.8-27B. Weights in bf16, or pre-quantized as compressed-tensors `pack-quantized` int4 / int8 (llm-compressor,
AutoRound, GPTQModel), symmetric or with zero points. A model without an MTP layer decodes without drafts, except
n-gram chains on deterministic requests.

`QWENNY_SPEC_STATS=1` logs tokens per step and acceptance by draft position per request.

## Requirements

- An NVIDIA GPU with compute capability 8.6 or newer; 24 GB for the 27B
- The CUDA toolkit (`nvcc`) and a Rust toolchain (edition 2024)
- Python with Transformers and PEFT, only to make the check references

## Quick start

```bash
cargo build --release
target/release/qwenny serve --model=Qwen/Qwen3.5-4B --ui     # API and web page on http://localhost:18023/

curl -s localhost:18023/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 200}'
```

Every command takes `--model=<directory or Hugging Face id>` (default `Qwen/Qwen3.5-4B`, downloaded once to the
Hugging Face cache; `HF_HOME`, `HF_TOKEN` and `HF_HUB_OFFLINE` apply), `--adapter=[<name>=]<dir>` (a PEFT LoRA,
served unmerged: see Server) and `--bf16` (16-bit weights instead of int8; pre-quantized checkpoints keep their
format).

```bash
qwenny train data/                        # LoRA on data/train.jsonl (+ dev.jsonl), rows {"messages": [...]}
qwenny serve --adapter=runs/data-1/adapter  # "model": "data-1" with the adapter, "qwenny" without
qwenny merge runs/data-1/adapter merged/  # write the merged model (bf16 checkpoints)
qwenny bench --pos=32000                  # decode, verify steps and prefill after a 32k context
qwenny bench --adapter=<dir>              # the same for base requests with the adapter loaded, then adapter requests
```

## MLX on Apple Silicon

Build without CUDA's default feature on macOS:

```bash
cargo build --release --no-default-features
target/release/qwenny serve --backend=mlx --model=<model-directory> --ctx=8192 --slots=1 --ui
```

MLX caches int8 attention keys/values with f32 scales, f32 DeltaNet state, and the MTP's own KV. The adaptive
vocabulary uses the existing int4 output weights and saves ids/scores under `~/.cache/qwenny/draft-heads/` (or
`$XDG_CACHE_HOME/qwenny/draft-heads/`). It grows on demand from the checkpoint's list to at most 65,536 rows.
Cross-request saved sessions/prefix reuse, adapters and training are not implemented on MLX yet. `--ctx` limits
context but KV allocation grows on demand; long-context performance must be measured separately.

## Server

`qwenny serve` options: `--host=0.0.0.0 --port=18023 --name=qwenny`, `--ctx=<tokens>` (default: the model's native
context, or the longest one session can hold on the GPU if that is less), `--slots=16` (concurrent sessions at most),
`--drafts=<n>` (speculative drafts per step, 0 turns drafting off; default: the draft layer's own, 7 for n-gram chains),
`--ui` (the web page), `--decisions=<presets.json>[,...]`,
`--db=<file>` (where the password and keys are kept; default `~/.local/share/qwenny/qwenny.db`),
`--adapter=[<name>=]<dir>` (a LoRA served beside the base model under `<name>`; default the directory's name, or its
parent's for `qwenny train`'s `runs/<data>-N/adapter`).
`QWENNY_CACHE_MEM=<GiB>` (environment, every command) caps the device memory of the slots' caches; by default they
take what is free after loading, less 512 MiB for CUDA graphs and the driver (a GPU shared with other work: set it).

**API**: `POST /v1/chat/completions` (streaming or not), `GET /v1/models`, `GET /health`. Request fields: `model`,
`messages`,
`tools`, `chat_template_kwargs`, `enable_thinking`, `reasoning_effort` (or `reasoning.effort`), `max_tokens`,
`temperature`, `top_p`, `top_k` (0: the whole vocabulary), `repetition_penalty`, `presence_penalty`, `seed`, `stop`, `stream`,
`stream_options.include_usage`, `ignore_eos`.

- `model`: with an adapter loaded, its name runs the request (chat or decision) with the adapter and `--name` (or no
  `model`) without it; any other name is an error. `/v1/models` lists both. Without an adapter the field is ignored.
- Thinking follows the model's chat template (on unless `enable_thinking` is false). `reasoning_effort`: `none` and
  `minimal` turn it off, `high` maps to the template's `xhigh`. The thinking is returned in `reasoning_content`.
- Tool calls use Qwen's XML format; parameter values are typed by the request's tool schema.
- Defaults when a request leaves them out (from the model card): thinking on: temperature 1.0, top_p 0.95, top_k 20;
  thinking off: 0.7, 0.8, 20; presence_penalty 1.5 in both.
- Sessions: a finished request's state stays cached; the next turn continues from there instead of reading the
  conversation again, and a new conversation that starts with the same system prompt (and tools) as a cached one
  copies that part instead of reading it. Concurrent requests run in parallel and share GPU memory by what they
  hold, so any one of them can use the full context. When the GPU is full, idle sessions move to host memory (and
  come back when continued), and a running request may pause until there is room; when host memory is full, the
  least recently used session is dropped and read again when continued. Requests do not fail for lack of memory.

**Web page** (`--ui`): Chat (streaming, thinking toggle, tok/s per reply; chats are kept in the browser), Decide (the
decisions endpoint) and Access. HTML and SVG blocks in a reply open in a sandboxed side panel.

**Access**: by default the server is open, the web page and the API alike. The page's Access tab shows the connection
details to copy (base URL, model, API key, examples), creates and revokes API keys, can require a key for the API
(`Authorization: Bearer <key>`), and can set a password for the page. If you forget the password, run
`qwenny password` on the server.

**Decisions**: `POST /v1/decisions` scores options instead of generating text: one prefill, then a softmax over the
labels' tokens at the answer position.

```bash
curl -s localhost:18023/v1/decisions -H 'Content-Type: application/json' -d '{"question": "What does the customer want?",
  "options": ["Refund", "Exchange", "Information"], "messages": [{"role": "user", "content": "..."}]}'
# {"choice": "Exchange", "index": 1, "probs": [...], "margin": ..., "label_mass": ...}
```

Fields: `messages`, `question`, `options` (strings, `{"key", "description"}` objects, or a map), `preset` (from
`--decisions`), `type` (`choice`; `score`: 2 to 10 ordered levels; `noul`: yes / no), `temperature`, `threshold` and
`min_margin` (adds `decision`: `auto` or `escalate`), `questions` (several at once), `think` (think first, optionally
with a token budget). `GET /v1/decisions/presets` lists the presets; `examples/` has some.

## How it works

**Engine.** Weights are int8 (groups of 128 with an f32 scale), f16 (`--bf16`), or the checkpoint's int4 in Marlin's
layout, dequantized to f16 as `(q - z) * scale`. Activations are f32, the KV cache int8 with an f32 scale per position
and head. Matrix products use tensor cores (`mma.m16n8k16`, f16 inputs, f32 accumulation). A step processes up to
2,048 tokens from any mix of requests (prompt chunks, decode tokens, drafts); steps of up to 32 tokens are replayed as
CUDA graphs (the least recently used go when the device memory the caches leave runs short; a step without room for
its graph launches its kernels directly). Sampling (repetition and presence penalty, temperature, top-k, top-p) runs on
the GPU. Prompts are built from the model's `chat_template.jinja` and `tokenizer.json`.

**LoRA adapters** are served unmerged, so one server answers the base model and the adapter side by side, in the same
steps: the base matrices stay as stored (int4, int8, 16-bit) and every adapted product adds alpha / r · B (A x) for
the tokens of requests that use the adapter. The modules that read the same input share one stacked A (a 16-bit
tensor-core product of the step); a kernel adds B's part to the product's partial sums, and for gate / up it adds it
before SwiGLU (the product then runs without its fused SwiGLU epilogue). Base requests get exactly the base model's
values, and steps without adapter requests skip the adapter's kernels. A and B are f16 on the GPU (exact for a bf16
adapter, 11 bits for an f32 one, as the base's 16-bit weights; f32 accumulation); rank 16 on all linear layers of the
27B takes ~230 MB, and the plain [gate; up] product of a 2,048-token step ~150 MB more. A session's cached state
belongs to the model it was computed with: a request for the other model reads its prompt again. The draft layer
(MTP) and the output layer stay unadapted: drafts are verified by the adapted model, so only acceptance can change.
Supported: ranks 8, 16, 32, 64 (PEFT, no rsLoRA / DoRA / rank or alpha patterns) on the language model's linear
layers (an adapter of the MTP or output layer is refused).

**Speculative decoding.** The checkpoint's MTP layer drafts four tokens per step by default (`--drafts=N`
for more on MLX). The target model checks all of
them in one step.

- Draft vocabulary: the drafts are scored over a subset of the output layer only (its rows for up to 65,536 tokens),
  which makes a draft step much cheaper. The subset starts from the checkpoint's list (`mtp_draft_vocab_ids.pt`) and
  adapts while the server runs: it counts the tokens it reads and writes (older counts decay), gives frequent tokens
  without a row one, and replaces rows that have not been used for a while. The selection is saved under
  `~/.cache/qwenny/draft-heads/` and loaded on the next start. No weights are trained; only which rows are included
  changes, so tokens you use often (code identifiers, domain terms, other languages) become draftable. MLX now
  uses the same 16-row growth/replacement and 65,536-row cap when the target output layer is stored in int4 and
  the checkpoint provides draft vocabulary ids. Its rows are gathered from the existing target weights, with no
  requantization; sampled draft distributions already drawn remain valid after a head update.
- Sampled replies draw their drafts from the draft layer's distribution q under the request's sampling settings;
  the target accepts a draft d with probability min(1, p(d) / q(d)) and otherwise samples from max(0, p − q)
  normalized (speculative sampling), so the output distribution is the model's. Greedy replies draft the argmax.
- On greedy requests, a repeated sequence of at least 8 tokens in the prompt or reply supplies drafts directly
  (n-gram chains).

**Training.** LoRA on all linear layers of the language model, as Unsloth's `get_peft_model`, with Unsloth's
`train_on_responses_only` loss mask. Output: `adapter/` (PEFT format), `train_log.jsonl` and `summary.json` under
`runs/<data>-N`. Rows render without thinking; a row's `"chat_template_kwargs"` (e.g. `{"enable_thinking": true,
"reasoning_effort": "low"}` with the assistant message's `reasoning_content`) renders it as a request with those
settings would be. The dev loss is measured every half epoch; the adapter with the lowest dev loss is kept, and training
stops after two measurements without improvement (without `dev.jsonl`, 5 % of `train.jsonl` is held out). Options:
`--epochs=3`, `--batch=8`, `--lr=2e-4`, `--rank=16`, `--alpha=32`, `--dropout=0.05`, `--max-len=4096`,
`--precision=f16` (or `bf16`, `f32`), `--init-adapter=<dir>` (start from a PEFT adapter, with its rank and alpha,
instead of a fresh LoRA). Each layer's backward pass recomputes its forward pass; frozen weights stay in their stored
format and are dequantized per product.

**Preference training** (DPO with an NLL term, RPO: Pang et al. 2024, "Iterative Reasoning Preference
Optimization"). Data whose rows carry `"chosen"` and `"rejected"` switches `qwenny train` to it (the first row of
`train.jsonl` decides; every row must then be a pair):

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "chosen": {"role": "assistant", "content": "...", "reasoning_content": "..."},
 "rejected": {"role": "assistant", "content": "..."}, "chat_template_kwargs": {"enable_thinking": true}}
```

Each response renders as `messages + [response]` (with the row's `chat_template_kwargs`), and only it is in the loss,
masked as SFT masks a reply (the prompt's earlier assistant turns are context). The reference policy is the starting
point, frozen: `--init-adapter` (e.g. the SFT / RFT adapter) or, without it, the base model; its log-probabilities of
every response are computed once before training. Per pair, with summed response log-probabilities `lp` (policy) and
`rf` (reference), margin `m = β((lp_c − rf_c) − (lp_r − rf_r))` and loss `−log σ(m) + α · NLL_c` (NLL_c: the chosen
response's mean per-token NLL), averaged over the step's pairs. Options: `--beta=0.1`, `--nll=1.0` (α; 0 is plain
DPO), and other defaults than SFT's: `--lr=5e-6` (alignment-handbook's QLoRA DPO recipe for Zephyr) and
`--dropout=0` (TRL's `disable_dropout`: policy and reference are compared without dropout); `--batch` counts pairs.
The log has per step the loss, its DPO and NLL parts, the mean margin, the share of pairs with a positive margin
(`accuracy`) and both rewards; the dev pairs (dev.jsonl or 5 % held out) are measured the same way every half epoch
(and at the start, where the margin is 0), and the adapter of the lowest dev loss is kept. A pair's two responses
share a micro-batch when they fit (4,096 tokens); a longer pair is split, the later half's log-probability then comes
from a forward pass before the backward of the first.

## Checks

`tools/checks.sh` runs each check against a reference with a pass criterion: `tools/checks.sh changed` (what the
uncommitted changes need), `all`, or one area (`qwen-infer`, `chat`, `qwen-train`, `qwen-dpo`, `qwen-train-27b`).

| Check | Compared with |
|---|---|
| `check`, `layercheck`, `longcheck` | Transformers fp32 logits and layer outputs |
| `check --adapter`, `adaptercheck` | Transformers + PEFT fp32 logits with a LoRA; adapter and base requests in the same steps vs. each in steps of one model |
| `speccheck` | the same model without drafts (one token per step) |
| `samplecheck` | a host reference sampler, draw for draw, including sampled drafts |
| `cachecheck`, `swapcheck` | the same sequence without saving or swapping |
| `chatcheck`, `tokcheck` | Transformers' chat template and tokenizer |
| `traincheck`, `datacheck` | PEFT's LoRA gradients and AdamW step; Unsloth's loss masks |
| `dpocheck` | preference training: RPO loss, log-probabilities and LoRA gradients of PyTorch + PEFT (cross-checked with TRL's `DPOTrainer`), the pairs' tokens and masks |

The references are made by `tools/make_reference.py`, `make_reference_mtp.py`, `make_reference_lora.py`,
`make_train_reference.py`, `make_dpo_reference.py` and `make_tokref.py`; `make_reference_vllm.py` takes top-k log-probabilities from a vLLM server for int4 checkpoints.
`tools/decode_bench.py` (one stream) and `tools/serve_bench.py` (1 to 16 streams) measure a running server;
`tools/serve_check.py` starts one and checks its replies over HTTP (thinking, streaming, tools, continued conversations, concurrent requests).

## Layout

```
src/main.rs                     command line
src/server.rs, src/sessions.rs  API server, scheduling, speculative decoding; slots and host swap
src/auth.rs                     the page's password and the API keys (SQLite)
src/chat.rs, src/decide.rs      chat template and tool calls; decisions
src/sample.rs                   sampling parameters, host reference sampler
src/checkpoint.rs, src/hub.rs   weight loading and formats; Hugging Face downloads
src/cuda/                       device code: products, steps, graphs (core.rs), the model (qwen/), training runtime
src/training.rs, src/lora.rs    LoRA training, loading and merging (served: src/cuda/qwen/lora.rs, kernels/lora.cuh)
kernels/                        CUDA kernels: products (gemm, Marlin), attention, DeltaNet, sampling, training
tools/                          checks, reference scripts, benchmarks
docs/exl3-plan.md               plan: EXL3 checkpoints (higher quality at ~4 bits)
```

## Acknowledgements

The int4 kernels are based on [Marlin](https://github.com/IST-DASLab/marlin) (IST Austria, Apache License 2.0),
extended with zero points and an argmax epilogue. Swift 1.5's int4 checkpoint and draft vocabulary were prepared with
[HyperQwen](https://github.com/syv-ai/HyperQwen).

## License

No license has been chosen yet. `kernels/marlin.cuh` keeps Marlin's Apache 2.0 notice.

## Linting

`scripts/lint.sh` is the CI gate: `cargo fmt --all --check`, then `cargo clippy --all-targets --all-features -- -D warnings`.
The lint levels live in the `[lints]` tables of `Cargo.toml` and the format in `rustfmt.toml`. `cargo machete` finds unused dependencies.
