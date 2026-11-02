# qwenny: notes for agents

What qwenny is and how it is used (engine, options, API, training): `README.md`. This file is how to work on it: the
rules, where things are and which commands to run. It is the only agent file; `CLAUDE.md` just includes it.

Several agent sessions may work in this repository at the same time.

## The goal: speed without giving up exactness

Most work here is making qwenny faster (decode tok/s, prompt tok/s, batched decode, speculative acceptance, training
step time), with the output staying what the model computes. No cheap compromises:

- No lossy shortcuts to win speed: the KV cache stays int8 (an f32 scale per position and head; no 4-bit or lower),
  no lower-precision accumulation, no approximated softmax / activations / sampling, no dropping or truncating work
  the reference does, no draft acceptance that changes the output distribution. Speed comes from better kernels,
  scheduling, fusion, memory layout, CUDA graphs, batching and drafts that are verified exactly.
- Weight formats are the checkpoint's own (int4 / int8 / bf16) or the documented int8 default; a new lossy format is
  the user's decision, not an optimization.
- Clear levers only: a change that adds complexity stays when it brings a measured gain of more than 3.5 % end to end
  (the user's threshold); otherwise the simpler version stays. Measure before and after with the same command, model,
  context and power limit. The checks apply either way.

## Main only

All work happens on `main`: commit small, checked steps there directly. The working tree is shared: stage only your
own files (`git add <files>`), and leave another session's uncommitted changes to it (ask it).

## Checks

The engine's logic is proven by checks against references from the originals (PyTorch + PEFT, Transformers and
Unsloth's loss masks). `tools/checks.sh` runs them, each with a pass criterion, and decides which ones a change needs.

- Before committing a change to `src/`, `kernels/`, `build.rs` or `tools/make_*`: run `tools/checks.sh changed`; it
  ends with `all checks OK`, or the commit waits. When it prints a `NOTE` about the 27B, also run
  `tools/checks.sh qwen-train-27b` on the 3090.
- A missing reference is a FAIL that prints the command that makes it; `--make` makes it. References live outside the
  repository (`~/.cache/qwen-engine/ref` for the 4B, `~/qref` for the 27B); `make_ref` in the script is their only
  recipe list.
- A check whose result changes on purpose (a new weight format, a changed reference): update its limit or reference in
  `tools/checks.sh` in the same commit and give the reason in the commit message. Loosening a limit so a speed change
  passes needs the user's agreement.
- A new model size or code path gets its check in `tools/checks.sh` (an `area` entry and its files in `areas_of`) in
  the commit that adds it.
- `out of memory` FAILs mean a busy GPU, not a bug: free the GPU and run again.

```bash
cargo build --release                       # target/release/qwenny; kernels compile to PTX for compute_86
cargo test --release                        # unit tests, CPU, seconds (checks.sh runs these first)
tools/checks.sh changed                     # the checks the uncommitted changes need
tools/checks.sh all                         # qwen-infer, chat, qwen-train, qwen-dpo: everything on the 4B
tools/checks.sh qwen-train-27b              # the 27B, on the 3090 only (from the dev box: tools/vm.sh tools/checks.sh qwen-train-27b)
cargo build --release --features mlx        # also the MLX backend (--backend=mlx; MLX's CUDA build, long the first time)
tools/checks.sh mlx                         # the MLX backend (builds with --features mlx)
```

- Areas: `qwen-infer` (logits vs. Transformers for int8 and 16-bit; with a LoRA vs. PEFT for int4, int8 and 16-bit,
  and adaptercheck: adapter and base requests in the same steps; layercheck of layers 0 and 3, speccheck with MTP,
  samplecheck with sampled drafts, longcheck, cachecheck, swapcheck, mtpcheck as INFO), `chat` (template and tokenizer
  vs. Transformers), `qwen-train` (LoRA gradients vs. PEFT for bf16 and int4, Unsloth's masks), `qwen-dpo`
  (preference training: RPO loss and LoRA gradients vs. PyTorch + PEFT and TRL, pairs split and packed), `qwen-train-27b`,
  `mlx` (the MLX backend: logits vs. Transformers, and vs. CUDA's on the same prompts within a bound (`check
  --write` makes CUDA's into a reference), speccheck for DeltaNet rollback; `tools/backend_diff.py`, whether CUDA and
  MLX servers reply byte for byte alike, as INFO: the backends round differently, so sampled replies can split at a
  near-tie). `areas_of` in the script maps changed files to areas.
- Correctness is tested on the 4B: everything but `qwen-train-27b` runs on `Qwen/Qwen3.5-4B` (bf16, int8 and
  16-bit) and `RedHatAI/Qwen3.5-4B-quantized.w4a16` (int4), from the Hugging Face cache.
- `QWENNY_BIN` and `QWENNY_PY` override the binary and the Python.
- The single checks, for a quick look while working: `qwenny check <ref>`, `layercheck <ref> --layer=N`,
  `speccheck`, `samplecheck`, `longcheck`, `cachecheck`, `swapcheck`, `mtpcheck <ref>`, `adaptercheck <ref>
  --adapter=<dir>`, `tokcheck`, `chatcheck`, `traincheck`, `datacheck`, `dpocheck` (all take `--model=`; `src/main.rs` lists their
  options; `check` and `bench` take `--adapter=` too).

## Measure speed

```bash
qwenny bench --model=<m>                    # decode, round trip, with logits, 1+2 / 1+4 draft steps, decode x4..x16, prompt
qwenny bench --model=<m> --pos=32000        # the same after a 32k context
qwenny bench --model=<m> --adapter=<dir>    # base requests with a LoRA loaded, then the LoRA's requests
python3 tools/decode_bench.py <url>         # a running server, one stream: 8 seeded prompts, tok/s after the first token
python3 tools/decode_bench.py <url> '{"temperature": 0}'   # the same greedy
python3 tools/serve_bench.py                # a running server with 1, 4, 8 and 16 concurrent replies
```

Bench the 4B int4 for quick iteration, the 27B (Swift 1.5) for the numbers that count, with nothing else on the GPU.
A speed number from the 3090 states its power limit (`nvidia-smi -q -d POWER`, currently 350 W) and whether
gpu-unlock is active (`sudo journalctl -k -b | grep gpu_unlock`, currently active).

- `qwenny bench` measures the engine's steps; what a change does to served decode (acceptance, drafts, host work)
  shows only end to end: `tools/decode_bench.py` against a **fresh server per run** (the draft head learns while it
  serves), the old and the new binary alternating, at least two runs each. Seeded runs repeat within about 1 %; a
  change of the products' summation order changes sampled texts, so compare geomeans, not single prompts.
- Profiling: `nsys profile -t cuda --cuda-graph-trace=graph` for the timeline and idle gaps; `--cuda-graph-trace=node`
  for per-kernel times only (it inflates graph launches to milliseconds, so its gaps are not real). Nsight Compute
  cannot replay the 27B (its memory does not fit twice); profile a kernel on the 4B, or time it in nsys.
- Rooflines on the 3090: about 860 GB/s attainable read bandwidth, 149 TFLOPS f16 tensor cores with f32 accumulation
  (gpu-unlock gives the full rate). A 27B verify step is about 84 % weight reads; decode attention at long context
  is at the bandwidth roofline already.

## Machines and GPUs

| | local dev box | the 3090 VM |
|---|---|---|
| GPU | RTX 5080, 16 GB | RTX 3090, 24 GB, 350 W limit, gpu-unlock active |
| host | | `ai`, `ubuntu@192.168.31.153` (tailscale 100.85.81.25) |
| repo | `~/Projects/qwenny` | `/home/ubuntu/qwenny` (a git clone of `github.com:hellerchr/qwenny`) |
| Python (Transformers, PEFT) | `~/.cache/qwenny/venv` | `/home/ubuntu/ft/.venv` (torch 2.14, transformers 5.20, peft 0.21) |
| references | `~/.cache/qwen-engine/ref` (4B) | `~/.cache/qwen-engine/ref` (4B), `~/qref` (27B) |
| Fjordhaus data (checks) | `~/.cache/qwenny/fjordhaus-data/data` | `~/.cache/qwenny/fjordhaus-data/data` |

`hostname` tells which one you are on.

- **The 3090 is free to use for this project**: the user allows stopping any task running on it, the `qwenny`
  service included. Look at `nvidia-smi`, stop what holds the GPU, and start the service again when done
  (`sudo systemctl start qwenny`) unless the user says otherwise. Announce a reboot of the VM to every session.
- The 5080 is shared with other sessions: before a job that takes more than a few GB or minutes, check `nvidia-smi`
  and agree with the session using the GPU. A server or bench takes all free device memory for its slots' caches when
  it starts: cap it there with `QWENNY_CACHE_MEM=<GiB>`.
- **On the VM, run as `ubuntu`, not root.** An agent's shell there may be root (`HOME=/root`), which has no cargo, no
  Hugging Face cache, no references and no venv, and a root build leaves root-owned files in `target/`:

  ```bash
  sudo -u ubuntu -H bash -lc 'cd ~/qwenny && export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH && <command>'
  ```

  Commits are the user's (Christian Heller, as in `git log`): the VM's clone has that identity in its `.git/config`.
- `tools/vm.sh <command>` (from the dev box) rsyncs the sources into the VM's `~/qwenny` with `--delete` (excluding
  `.git` and `target`) and runs the command there. That overwrites uncommitted edits made on the VM, and its builds
  replace `target/release/qwenny`, the binary the service runs (the running process keeps the old one until a
  restart).

## Models (VM)

- **Early testing: the 4B int4**, `/opt/qwen38/models/Qwen3.5-4B-w4a16` (RedHatAI w4a16, compressed-tensors
  `pack-quantized`, symmetric, with an MTP layer; 5.2 GB on disk, roughly 3–4 GB on the GPU plus the KV cache).
  Try changes on it first, and use the 27B only once they work on the 4B. The bf16 4B is
  `/opt/qwen38/models/Qwen3.5-4B` (8.8 GB; also `Qwen/Qwen3.5-4B` in the Hugging Face cache). Example:
  `target/release/qwenny serve --model=/opt/qwen38/models/Qwen3.5-4B-w4a16 --port=18025 --ctx=8192 --slots=1`.
  It does not fit next to the Swift 1.5 service: stop the service while testing.
- `~/models/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen-INT4`: Swift 1.5, the model the service serves (int4 AWQ body,
  GPTQ output layer, MTP head in `model-mtp-bf16.safetensors`, draft vocabulary `mtp_draft_vocab_ids.pt`).
- `~/models/Swift-1.5-Qwen3.8-27B-EXL3-4.00bpw-H5`: Swift 1.5 as EXL3 (4.00 bits, 5-bit output layer, MTP layer;
  `BigBojangles/Swift-1.5-Qwen3.8-27b-EXL3-4.00bpw-H5`, branch `SC_4.00bpw_H5`), for the EXL3 port.
- `/opt/qwen38/models/`: `Qwen3.8-27B-W4A16-AutoRound` (the 27B training check's model, `Q27` in checks.sh),
  `Qwen3.8-27B-W4A16-AutoRound-fast`, `Qwen3.5-4B`, `Qwen3.5-4B-w4a16`, `adapters/`.
- Hugging Face cache (`/home/ubuntu/.cache/huggingface/hub`): `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`,
  `RedHatAI/Qwen3.5-{4B,9B}-quantized.w4a16`, `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`.
- Learned draft-vocabulary rows: `~/.cache/qwenny/draft-heads/`.

## The qwenny service (VM)

`/etc/systemd/system/qwenny.service` runs Swift 1.5 from this repo's `target/release/qwenny` as user `ubuntu`:

```
qwenny serve --model=/home/ubuntu/models/Swift-1.5-Qwen3.8-27B-W4A16-HyperQwen-INT4 \
  --host=0.0.0.0 --port=18024 --name=swift-1.5 --ui
```

`Restart=always`. It holds nearly all of the 3090's memory (~23.8 of 24 GB), so checks, benches and the 4B test model
need it stopped; start it again afterwards.

```bash
systemctl status qwenny                     # state
journalctl -u qwenny -f                     # logs
sudo systemctl stop qwenny                  # free the GPU
sudo systemctl start qwenny                 # back up; restart picks up a new build
curl -s localhost:18024/health              # "ok" when loaded
```

The page is at `http://192.168.31.153:18024/`. A restart reloads the 27B and takes a while; wait for `/health`.

## Source map

- `src/main.rs`: the CLI (serve, train, merge, bench and the checks) and its options.
- `src/engine.rs`, `src/cuda/core.rs`, `src/cuda/qwen/mod.rs`: the model, one step of up to 2,048 tokens, CUDA graphs.
- `src/cuda/qwen/{attn,deltanet,draft}.rs` + `kernels/qwen_{attn,deltanet}.cuh`: gated attention, Gated DeltaNet,
  the MTP's draft head.
- `kernels/gemm.cuh` (int8 / f16 tensor-core products), `kernels/marlin.cuh` (all int4, symmetric or with zero
  points; argmax epilogue), `kernels/sampling.cuh` (the sampler, sampled drafts).
- `src/server.rs`, `src/sessions.rs` (slots, saved states, host swap), `src/chat.rs` (template, tool calls),
  `src/decide.rs`, `src/sample.rs`, `src/ui.html`.
- `src/training.rs`, `src/lora.rs`, `src/cuda/learn.rs`, `src/cuda/qwen/train.rs`, `kernels/qwen_train.cu`: LoRA;
  `src/preference.rs`: preference training (RPO) on it.
- `src/cuda/qwen/lora.rs`, `kernels/lora.cuh`: a LoRA served unmerged, per request (`Engine::set_adapter` per slot;
  sessions keep the base model's and the adapter's states apart).
- `src/checkpoint.rs`, `src/hub.rs`: loading weights (formats, quantization), Hugging Face downloads.
- `src/checks.rs`: the checks and `bench`.
- `src/mlx/`: the MLX backend (`--backend=mlx`; Metal on macOS, MLX's CUDA backend on Linux with `--features mlx`,
  through `vendor/mlx-sys`): the model as MLX operations (`weights.rs`: layers in compiled stages; `model.rs`: caches,
  rollback; `delta.rs`: the delta rule per token and chunked), inference only (n-gram drafts; no MTP drafts, adapters,
  training or saved sessions). It is for Apple hardware: optimize what helps any MLX device (fewer operations,
  compiled stages, less memory traffic); its speed on the 5080 through MLX's CUDA backend says little about Metal.
