# EXL3 in qwenny: plan

Goal: run Swift 1.5 (Qwen3.8-27B) from an EXL3 checkpoint at about 4 bits per weight, for higher quality than the
current AWQ int4 at the same memory, with the full context (about 250k tokens, int8 KV cache) on one RTX 3090.
Assumption: EXL3 at 4.0 bits is better than AWQ int4 for this model (to be confirmed by the quality check below).

## Why the bits per weight cannot grow

Device memory on the 3090 (24 GB) with one session of 250k tokens:

| | GiB |
|---|---|
| KV cache and DeltaNet state of the session | ~8.4 |
| step buffers, CUDA context, reserve | ~2 |
| output layer and draft layer | ~0.8 |
| left for the 64 body layers (~24.3B weights) | ~12 |

So about 4 bits per weight, scales included. Quality has to come from the quantization method, not from more bits.

## Checkpoint

`BigBojangles/Swift-1.5-Qwen3.8-27b-EXL3-4.00bpw-H5`, branch `SC_4.00bpw_H5` (16.4 GB on disk, embedding and vision
weights included), downloaded to `~/models/Swift-1.5-Qwen3.8-27B-EXL3-4.00bpw-H5` on the VM.

- Body: EXL3 4.00 bits per weight, codebook `mul1`; output layer 5 bits; MTP layer 4 bits (39 tensors); vision 6 bits
  (not loaded).
- Body on the device: about 11.3 GiB (the current int4 body: about 12.1 GiB).
- Alternative: `JRamirez-UAB/Swift-1.5-Qwen3.8-27B-EXL3-3.85bpw-24GB` (3.85 bits, 14.7 GB, MTP included, int8
  embedding).

## The format

Per quantized matrix (in features k, out features n):

| tensor | dtype, shape | meaning |
|---|---|---|
| `trellis` | int16 `[k/16, n/16, 16·K]` | 16×16 tiles, K bits per weight (K = 4 for the body, 5 for the output layer) |
| `suh` | f16 `[k]` | sign and scale vector of the input's Hadamard rotation |
| `svh` | f16 `[n]` | sign and scale vector of the output's Hadamard rotation |
| `mul1` | int32 scalar | codebook multiplier, 0x83DCD12D (2212286765) for every matrix here |

- Matrices are stored separately (q, k, v; gate, up; in_proj_qkv, in_proj_z), each with its own `suh`: matrices that
  share an input cannot be stacked into one product as in the Marlin path; each rotates its own copy of the input.
- `in_proj_a`, `in_proj_b` and the norms stay bf16. The embedding is int8 (`qweight`, `scales`).
- The product: y = H₁₂₈(svh ⊙ (W · H₁₂₈(suh ⊙ x))), H₁₂₈ a Hadamard transform over blocks of 128.
- Decoding a tile: its 256 × K bits are one circular bitstream; weight t of the tile has as its 16-bit state the
  window ending at bit (t + 1)·K. The `mul1` codebook makes the weight from the state: multiply by the multiplier
  (32 bits), sum the 4 bytes (dp4a) plus 0x6400, read as f16, times 1/147.7, plus −10.39 (ExLlamaV3:
  `quant/codebook.cuh`, `quant/exl3_dq.cuh`).
- Still to pin down against the reference: the order of a tile's 256 weights (row, column) and the word order of the
  int16 array.

## Steps

### 1. Reference and format

1. ExLlamaV3 (MIT license) in a scratch venv that reuses the checks venv's torch, as the reference runtime
   (`exllamav3/modules/quant/exl3.py`: `reconstruct`, `forward`).
2. qwenny's loader reads EXL3 checkpoints (`quant_method: exl3`): the tensors above per matrix, the int8 embedding,
   the 5-bit output layer, the MTP layer.
3. A CPU dequantizer in Rust, unit-tested bit for bit against ExLlamaV3's `reconstruct` on tensors of the
   checkpoint.

### 2. Kernels

1. The Hadamard transform over blocks of 128 with the sign vectors (from ExLlamaV3's `had_r_128`).
2. The decode product, up to 32 token rows (one stream with drafts, or several streams), from `exl3_gemv` and the
   small-batch `exl3_gemm`; with qwenny's conventions: CUDA graphs, the SwiGLU epilogue (gate and up), the argmax
   epilogue (output layer).
3. The prompt product (chunks of 2,048 tokens): first tile slices reconstructed to f16 and qwenny's f16 tensor-core
   product; the fused `exl3_gemm` if prefill needs it.

### 3. Engine

- A weight kind `Wt::Exl3` next to Marlin; AWQ and GPTQ checkpoints stay supported.
- The output layer on EXL3 with the argmax epilogue; the MTP layer on EXL3.
- The draft head: an EXL3 output row exists only together with its 128-wide Hadamard block, so its rows cannot be
  selected from the output layer. It is rebuilt at load: the 65,536 rows dequantized and stored as int8 (~335 MB
  instead of ~168 MB as int4); a learned row is written the same way. This affects only the drafts, not the output.

### 4. Checks and measurements

- Per matrix: qwenny's GPU product against ExLlamaV3's, with a limit on the error.
- The whole model: logits against ExLlamaV3 on a set of prompts (top-1 agreement, KL divergence).
- `speccheck`, `samplecheck`, `longcheck`, `cachecheck` on the EXL3 27B; the 250k context fits beside the weights.
- A 4B check model converted to EXL3 with ExLlamaV3's converter, so that `tools/checks.sh` covers the format on the
  4B (an `area` for it and its files in `areas_of`).
- Speed against AWQ: decode with 1 to 3 streams, verify steps, prefill (`qwenny bench`, `tools/decode_bench.py`).
- Quality against AWQ: KL divergence to a higher-precision reference, to confirm the assumption.

### 5. Service

`--model=` to the EXL3 directory; nothing else changes.

## Risks

- Speed: EXL3 reads about 6 % fewer bytes than int4 (4.0 against ~4.25 bits per weight), so single-stream decode
  should be about as fast or slightly faster; decoding the trellis costs more arithmetic per weight, so prefill and
  batched decode (several streams) may get slower. Measured right after steps 2.2 and 2.3.
- The tile layout and word order are unknown until checked against the reference.
- Effort: one to two weeks of focused work, mostly the kernels and their checks.

## Status (2026-11-01)

- Disk: 46 GB free (the DFlash2 drafters removed; `~/ft` reduced to its `.venv`, which the checks use; the Fjordhaus
  data moved to `~/.cache/qwenny/fjordhaus-data`).
- The checkpoint is downloading.
- The format above is mapped from the checkpoint's `quantization_config.json` and ExLlamaV3's source (commit
  `bd1f519`).
- Next: step 1.1 (the reference venv), then 1.3 (the dequantizer).
