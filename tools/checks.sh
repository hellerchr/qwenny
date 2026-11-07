#!/bin/bash
# The correctness checks against references from the originals (PyTorch + PEFT, Transformers, TRL / Unsloth's masks),
# with a pass criterion each, after `cargo test`: OK / FAIL per check, exit status 1 if any failed.
#
#   tools/checks.sh changed          the areas the uncommitted changes touch (git diff HEAD); see `areas_of`
#   tools/checks.sh <area>...        qwen-train qwen-dpo qwen-train-27b chat qwen-infer mlx, or all (all
#                                    but qwen-train-27b and mlx)
#   --make                           make missing references (else a missing one is a FAIL that prints its command)
#
# qwen-train-27b runs on the RTX 3090 VM only (the 27B needs 24 GB): tools/vm.sh tools/checks.sh qwen-train-27b, after
# asking whoever uses the 3090. Every other area runs on the local GPU. Python: QWENNY_PY (default the venv below).
set -uo pipefail
cd "$(dirname "$0")/.."
BIN=${QWENNY_BIN:-target/release/qwenny}
PY=${QWENNY_PY:-$HOME/.cache/qwenny/venv/bin/python}
[ -x "$PY" ] || PY=$HOME/ft/.venv/bin/python   # the VM's venv (Transformers, PEFT)
MAKE=0
FAILS=()

# --- references: where each lives and the command that makes it ---------------------------------------------------
QREF=$HOME/.cache/qwen-engine/ref      # Qwen3.5-4B
VREF=$HOME/qref                        # Qwen3.8-27B (VM)
FJ=$HOME/.cache/qwenny/fjordhaus-data/data   # Fjordhaus train.jsonl / dev.jsonl
Q27=/opt/qwen38/models/Qwen3.8-27B-W4A16-AutoRound
W4A16=RedHatAI/Qwen3.5-4B-quantized.w4a16
LORA=$QREF/lora-adapter               # the LoRA references' adapter (rank 16, every linear layer, random B)

snapshot() { "$PY" -c "from huggingface_hub import snapshot_download as s; print(s('$1', local_files_only=True))"; }

# make_ref <file>: the command that makes reference <file>
make_ref() {
    case $1 in
        $QREF/train-grads-4l.safetensors) echo "$PY tools/make_train_reference.py grads Qwen/Qwen3.5-4B $FJ $1 3 4" ;;
        $QREF/train-grads-4l-w4a16.safetensors)
            echo "$PY tools/make_train_reference.py grads \$(snapshot $W4A16) $FJ $1 3 4 --weights=\$(snapshot $W4A16)" ;;
        # preference training (RPO): 3 pairs of the Fjordhaus data, the first 4 layers, cross-checked with TRL
        $QREF/dpo-4l.safetensors) echo "$PY tools/make_dpo_reference.py Qwen/Qwen3.5-4B $FJ $1 3 4" ;;
        $QREF/dpo-4l-w4a16.safetensors)
            echo "$PY tools/make_dpo_reference.py \$(snapshot $W4A16) $FJ $1 3 4 --weights=\$(snapshot $W4A16)" ;;
        $QREF/data-ref.safetensors) echo "$PY tools/make_train_reference.py data Qwen/Qwen3.5-4B $FJ $1" ;;
        # fp32 is 33 GiB: the memory limits keep it from the dev box's swap (the rest runs from disk, on the GPU layer by layer)
        $QREF/reference.safetensors) echo "$PY tools/make_reference.py Qwen/Qwen3.5-4B $FJ/dev.jsonl $1" ;;
        $QREF/mtp-ref.safetensors) echo "$PY tools/make_reference_mtp.py Qwen/Qwen3.5-4B $FJ/dev.jsonl $1" ;;
        # the LoRA references share one random adapter ($LORA, written by the first one made): remake all together
        $QREF/lora-ref-w4a16.safetensors)
            echo "$PY tools/make_reference_lora.py Qwen/Qwen3.5-4B $FJ/dev.jsonl $1 $LORA --weights=\$(snapshot $W4A16)" ;;
        $QREF/lora-ref.safetensors) echo "$PY tools/make_reference_lora.py Qwen/Qwen3.5-4B $FJ/dev.jsonl $1 $LORA" ;;
        $VREF/train-grads-27b-4l.safetensors) echo "$PY tools/make_train_reference.py grads $Q27 $FJ $1 3 4 --weights=$Q27" ;;
        *) echo "no recipe for $1" ;;
    esac
}

# need <file>: the reference exists (made with --make), else a FAIL with its recipe
need() {
    [ -e "$1" ] && return 0
    local cmd; cmd=$(make_ref "$1")
    if [ $MAKE = 1 ]; then
        echo "making $1: $cmd"
        mkdir -p "$(dirname "$1")"
        eval "$cmd" && [ -e "$1" ] && return 0
    fi
    fail "missing reference $1 — make it with: $cmd   (or rerun with --make)"
    return 1
}

ok() { echo "OK    $*"; }
fail() { echo "FAIL  $*"; FAILS+=("$*"); }

# --- checks ---------------------------------------------------------------------------------------------------------

# run <name> <command...>: one check. It passes when the command exits 0 and its last line is the summary. Every
# `qwenny` check takes its limit as --limit=<x>, fails with an error when its measure is not below it, and ends in one
# summary line.
run() {
    local name=$1; shift
    local out; out=$("$@" 2>&1)
    if [ $? -eq 0 ]; then ok "$name: $(tail -1 <<<"$out")"
    else fail "$name: $(tail -3 <<<"$out" | tr '\n' ' ' | cut -c1-500)"; fi
}

# traincheck <name> <reference> <max worst relative error> <args...>: loss, LoRA gradients, AdamW vs. PEFT fp32
# (traincheck, dpocheck and datacheck print their numbers and do not judge them yet: their criteria are read here)
traincheck() {
    local name=$1 ref=$2 max=$3; shift 3
    need "$ref" || return
    local out; out=$($BIN traincheck "$ref" --packed "$@" 2>&1)
    local err cos; err=$(grep -oP 'gradients of \d+ tensors: worst relative error \K[0-9.e+-]+' <<<"$out")
    cos=$(grep -oP 'lowest cosine \K[0-9.]+' <<<"$out")
    if [ -n "$err" ] && awk -v e="$err" -v m="$max" 'BEGIN { exit !(e < m) }'; then
        ok "traincheck $name: worst gradient error $err (< $max), cosine $cos"
    else
        fail "traincheck $name: worst gradient error ${err:-?} (limit $max): $(grep -E '^(loss|Error)' <<<"$out" | head -2)"
    fi
}

# dpocheck <name> <reference> <max worst relative error> <args...>: preference training (RPO) vs. PyTorch + PEFT fp32:
# the pairs' token ids and masks identical, then loss, log-probabilities and LoRA gradients
dpocheck() {
    local name=$1 ref=$2 max=$3; shift 3
    need "$ref" || return
    local out; out=$($BIN dpocheck "$ref" "$@" 2>&1)
    local err cos loss; err=$(grep -oP 'gradients of \d+ tensors: worst relative error \K[0-9.e+-]+' <<<"$out")
    cos=$(grep -oP 'lowest cosine \K[0-9.]+' <<<"$out")
    loss=$(grep -oP '^loss .*\(diff \K[0-9.e+-]+' <<<"$out")
    if [ -n "$err" ] && grep -qP '^(\d+) responses: ids identical for \1, masks identical for \1$' <<<"$out" \
        && awk -v e="$err" -v m="$max" 'BEGIN { exit !(e < m) }'; then
        ok "dpocheck $name: tokens identical, loss diff $loss, worst gradient error $err (< $max), cosine $cos"
    else
        fail "dpocheck $name: worst gradient error ${err:-?} (limit $max): $(grep -E '^(loss|Error|[0-9]+ responses)' <<<"$out" | head -2 | tr '\n' ' ')"
    fi
}

# datacheck <name> <reference> <args...>: token ids and loss masks identical to TRL's / Unsloth's for every example
datacheck() {
    local name=$1 ref=$2; shift 2
    need "$ref" || return
    local out; out=$($BIN datacheck "$FJ" "$ref" "$@" 2>&1)
    if awk '/examples: ids identical/ { n = $2; a = $7; b = $11; sub(/,/, "", a); if (a != n || b != n) bad = 1; seen = 1 }
            END { exit !(seen && !bad) }' <<<"$out"; then
        ok "datacheck $name: $(grep -E '^(train|dev): ' <<<"$out" | tr '\n' ' ')"
    else
        fail "datacheck $name: $(grep -E '^(train|dev): |Error' <<<"$out" | tr '\n' ' ')"
    fi
}

# logits <name> <reference> <max mean |diff| per prompt> <args...>: logits vs. Transformers fp32 (qwenny check)
logits() {
    local name=$1 ref=$2 max=$3; shift 3
    need "$ref" || return
    run "check $name" $BIN check "$ref" "$@" --limit="$max"
}

# layercheck <layer> [reference] [model] [limit]: one layer's mixer vs. Transformers (max |diff|; default the Qwen 4B, 0.05)
layercheck() {
    local ref=${2:-$QREF/reference.safetensors} model=${3:-Qwen/Qwen3.5-4B} max=${4:-0.05}
    need "$ref" || return
    run "layercheck $1" $BIN layercheck "$ref" --model="$model" --layer="$1" --limit="$max"
}

# chatcheck <tokref> <args...>: chat template and tokenizer vs. Transformers, every case identical
chatcheck() { run "chatcheck $1" $BIN chatcheck "$1" "${@:2}"; }

# tokcheck [reference] [model]: prompt tokens identical to Transformers' (the reference's first two prompts; default the
# Qwen 4B)
tokcheck() {
    local ref=${1:-$QREF/reference.safetensors} model=${2:-Qwen/Qwen3.5-4B}
    need "$ref" || return
    run tokcheck $BIN tokcheck "$ref" "$FJ/dev.jsonl" --model="$model"
}

# speccheck <options>: drafts (the model's draft layer) give the tokens of one token per step
speccheck() { run "speccheck $*" $BIN speccheck "$@"; }

# backend_logits <name> <max mean |diff| per prompt> [chunk] <args...>: MLX's logits vs. CUDA's on the reference prompts
# (CUDA writes them as a reference: qwenny check --write); the two backends' kernels sum in different orders, so this is a
# bound, not identity
backend_logits() {
    local name=$1 max=$2; shift 2
    local ref=$QREF/reference.safetensors
    need "$ref" || return
    local chunk=()
    [[ ${1:-} =~ ^[0-9]+$ ]] && { chunk=("$1"); shift; }
    local tmp; tmp=$(mktemp -d)
    local out; out=$($BIN check "$ref" "$@" --backend=cuda --write="$tmp/cuda.safetensors" 2>&1)
    if [ -e "$tmp/cuda.safetensors" ]; then
        logits "MLX vs. CUDA $name" "$tmp/cuda.safetensors" "$max" "${chunk[@]}" "$@" --backend=mlx
    else
        fail "MLX vs. CUDA $name: the CUDA logits: $(grep Error <<<"$out" | head -1)"
    fi
    rm -rf "$tmp"
}

# backend_diff <args...>: whether fresh CUDA and MLX servers give byte-identical replies (tools/backend_diff.py); INFO,
# since rounding differences flip sampled tokens at near-ties
backend_diff() {
    local out; out=$(python3 tools/backend_diff.py --bin="$BIN" "$@" 2>&1)
    if grep -qE '^(OK|FAIL) ' <<<"$out"; then
        echo "INFO  backend_diff $*: $(grep -cE '^OK ' <<<"$out")/$(grep -cE '^(OK|FAIL) ' <<<"$out") replies byte-identical $(grep -E '^FAIL' <<<"$out" | cut -c7- | tr '\n' ' ')"
    else
        fail "backend_diff $*: did not run: $(tail -1 <<<"$out")"
    fi
}

# samplecheck <options>: the sampler draw for draw against its host reference (no files)
samplecheck() { run "samplecheck $*" $BIN samplecheck "$@"; }

# longcheck <limit> <options>: long contexts at every position, decode steps vs. prompt chunks: the logits of the last 64
# positions agree up to rounding (an argmax may flip where two logits nearly tie: the 4B has one at 1,500 tokens, 0.026
# apart)
longcheck() {
    local max=$1; shift
    run "longcheck $*" $BIN longcheck "$@" --limit="$max"
}

# adaptercheck <reference> <max mean |diff|> <args...>: a LoRA served unmerged and its base model in the same steps:
# every logit row identical to the steps with one model, and each within the limit of its reference (PEFT / base)
adaptercheck() {
    local ref=$1 max=$2; shift 2
    need "$ref" || return
    run "adaptercheck $*" $BIN adaptercheck "$ref" "$@" --limit="$max"
}

# serve <args...>: a real server answering over HTTP (tools/serve_check.py), every case OK
serve() { run "serve $*" python3 tools/serve_check.py --bin="$BIN" "$@"; }

# --- areas ----------------------------------------------------------------------------------------------------------

area() {
    echo "== $1"
    case $1 in
        qwen-train)
            traincheck "Qwen3.5-4B f32" "$QREF/train-grads-4l.safetensors" 1e-4 --model=Qwen/Qwen3.5-4B --precision=f32
            traincheck "Qwen3.5-4B f16" "$QREF/train-grads-4l.safetensors" 0.05 --model=Qwen/Qwen3.5-4B --precision=f16
            traincheck "Qwen3.5-4B int4 f32" "$QREF/train-grads-4l-w4a16.safetensors" 1e-4 --model=$W4A16 --precision=f32
            datacheck "Qwen (Unsloth's masks)" "$QREF/data-ref.safetensors" --model=Qwen/Qwen3.5-4B ;;
        qwen-dpo)
            # preference training: every pair split over two passes (the later half's forward first), and packed
            dpocheck "Qwen3.5-4B f32, pairs split" "$QREF/dpo-4l.safetensors" 1e-4 --model=Qwen/Qwen3.5-4B --precision=f32
            dpocheck "Qwen3.5-4B f32, packed" "$QREF/dpo-4l.safetensors" 1e-4 --model=Qwen/Qwen3.5-4B --precision=f32 --packed
            dpocheck "Qwen3.5-4B f16, packed" "$QREF/dpo-4l.safetensors" 0.05 --model=Qwen/Qwen3.5-4B --precision=f16 --packed
            dpocheck "Qwen3.5-4B int4 f32" "$QREF/dpo-4l-w4a16.safetensors" 1e-4 --model=$W4A16 --precision=f32 ;;
        qwen-train-27b)
            traincheck "Qwen3.8-27B int4 f32" "$VREF/train-grads-27b-4l.safetensors" 1e-4 --model=$Q27 --precision=f32
            traincheck "Qwen3.8-27B int4 f16" "$VREF/train-grads-27b-4l.safetensors" 0.05 --model=$Q27 --precision=f16 ;;
        chat)
            chatcheck tools/tokref-qwen.json --model=Qwen/Qwen3.5-4B
            tokcheck ;;
        qwen-infer)
            logits "Qwen3.5-4B int8" "$QREF/reference.safetensors" 0.12 --model=Qwen/Qwen3.5-4B
            logits "Qwen3.5-4B 16-bit" "$QREF/reference.safetensors" 0.05 --model=Qwen/Qwen3.5-4B --bf16
            # a LoRA served unmerged (the int4 matrices as stored), decode steps and prompt chunks of 64; on int8 and
            # 16-bit weights; mixed with base requests in the same steps. The adapter moves the logits by ~0.3-0.5 on
            # average, so these limits are an order of magnitude tighter than its effect
            logits "Qwen3.5-4B int4 + LoRA" "$QREF/lora-ref-w4a16.safetensors" 0.05 --model=$W4A16 --adapter=$LORA
            logits "Qwen3.5-4B int4 + LoRA, chunks of 64" "$QREF/lora-ref-w4a16.safetensors" 0.05 64 --model=$W4A16 --adapter=$LORA
            logits "Qwen3.5-4B int8 + LoRA" "$QREF/lora-ref.safetensors" 0.12 --model=Qwen/Qwen3.5-4B --adapter=$LORA
            logits "Qwen3.5-4B 16-bit + LoRA" "$QREF/lora-ref.safetensors" 0.05 --model=Qwen/Qwen3.5-4B --bf16 --adapter=$LORA
            adaptercheck "$QREF/lora-ref-w4a16.safetensors" 0.05 --model=$W4A16 --adapter=$LORA
            layercheck 0
            layercheck 3
            speccheck --model=Qwen/Qwen3.5-4B
            samplecheck --model=Qwen/Qwen3.5-4B
            longcheck 0.1 --bf16 --model=Qwen/Qwen3.5-4B
            run cachecheck $BIN cachecheck --model=Qwen/Qwen3.5-4B
            run swapcheck $BIN swapcheck --model=Qwen/Qwen3.5-4B --len=32768
            serve --model=Qwen/Qwen3.5-4B --slots=4 --ctx=32768 --bf16
            if need "$QREF/mtp-ref.safetensors"; then   # no pass criterion yet: its result, to compare by eye
                echo "INFO  mtpcheck: $($BIN mtpcheck "$QREF/mtp-ref.safetensors" --model=Qwen/Qwen3.5-4B 2>&1 | tail -1)"
            fi ;;
        mlx)
            # the MLX backend (built with --features mlx): the reference limits CUDA has, MLX's logits within about 1.6x the
            # distance to CUDA's measured on the 5080 (16-bit 0.016, int8 0.025, int4 0.021), the rollback, and (INFO)
            # whether servers reply byte for byte as CUDA's
            logits "MLX Qwen3.5-4B int8" "$QREF/reference.safetensors" 0.12 --model=Qwen/Qwen3.5-4B --backend=mlx
            logits "MLX Qwen3.5-4B 16-bit" "$QREF/reference.safetensors" 0.05 --model=Qwen/Qwen3.5-4B --bf16 --backend=mlx
            backend_logits "Qwen3.5-4B 16-bit" 0.025 --model=Qwen/Qwen3.5-4B --bf16
            backend_logits "Qwen3.5-4B int8" 0.04 --model=Qwen/Qwen3.5-4B
            backend_logits "Qwen3.5-4B int4" 0.035 --model=$W4A16
            backend_logits "Qwen3.5-4B int4, chunks of 64" 0.035 64 --model=$W4A16
            # DeltaNet rollback: restarts inside a step replay from the checkpoint
            speccheck --model=Qwen/Qwen3.5-4B --backend=mlx
            # MTP on MLX: its drafts and rollback, sampled drafts, then the Metal code's Rust side (it runs on a Mac only)
            speccheck --model=$W4A16 --backend=mlx
            samplecheck --model=$W4A16 --backend=mlx
            if RUSTFLAGS="--cfg has_metal" cargo check --features mlx --target-dir target/metal-check 2>&1 | grep -q '^error'; then
                fail "Metal code does not type-check"; else ok "Metal code type-checks"; fi
            # a real server: its caches' device memory (admission, the context) and replies over HTTP
            serve --model=$W4A16 --slots=4 --ctx=32768 --backend=mlx
            backend_diff --model=Qwen/Qwen3.5-4B --bf16
            backend_diff --model=Qwen/Qwen3.5-4B
            backend_diff --model=$W4A16 ;;
        *) fail "unknown area $1" ;;
    esac
}

# areas_of <file>: the areas a changed file can break (the map AGENTS.md describes)
areas_of() {
    case $1 in
        kernels/qwen_train.cu|kernels/train_common.cuh|src/cuda/qwen/train.rs|src/cuda/qwen/train/*|src/cuda/learn.rs\
        |src/training.rs|tools/make_train_reference.py) echo qwen-train qwen-dpo ;;
        src/preference.rs|tools/make_dpo_reference.py) echo qwen-dpo ;;
        src/checkpoint.rs|build.rs|src/engine.rs|src/hub.rs) echo qwen-train qwen-dpo qwen-infer mlx ;;
        src/cuda/qwen/mod.rs|src/lora.rs) echo qwen-train qwen-dpo qwen-infer ;;
        src/abliterate.rs) echo qwen-infer ;;
        src/mlx/*|vendor/mlx-sys/*|tools/backend_diff.py|Cargo.toml) echo mlx ;;
        src/draft_vocab.rs) echo qwen-infer mlx ;;
        src/main.rs|tools/checks.sh) echo qwen-train qwen-dpo chat qwen-infer mlx ;;
        src/chat.rs) echo chat qwen-train qwen-dpo ;;
        kernels/*|src/cuda/*) echo qwen-infer ;;
        src/sample.rs|src/sessions.rs|src/checks.rs|src/server.rs) echo qwen-infer mlx ;;
        tools/make_reference*.py|tools/make_tokref.py|tools/tokref-*.json) echo chat qwen-infer ;;
        tools/serve_check.py|tools/qwenny_client.py) echo qwen-infer mlx ;;
    esac
}

args=()
for a in "$@"; do
    case $a in
        -h|--help|help) sed -n 2,10p "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --make) MAKE=1 ;;
        all) args+=(qwen-train qwen-dpo chat qwen-infer) ;;
        changed)
            files=$(git diff --name-only HEAD; git ls-files --others --exclude-standard src kernels tools vendor)
            mapfile -t found < <(for f in $files; do areas_of "$f"; done | tr ' ' '\n' | grep . | sort -u)
            [ ${#found[@]} -eq 0 ] && echo "the changes touch no checked area" && exit 0
            echo "changed: $(echo $files | tr '\n' ' ')-> ${found[*]}"
            if grep -qE '^(kernels/qwen_train.cu|src/cuda/qwen/(train|mod).rs|src/checkpoint.rs)$' <<<"$files"; then
                echo "NOTE  these files reach the 27B too: also run tools/vm.sh tools/checks.sh qwen-train-27b (on the 3090)"
            fi
            args+=("${found[@]}") ;;
        *) args+=("$a") ;;
    esac
done
[ ${#args[@]} -eq 0 ] && { sed -n 2,10p "$0" | sed 's/^# \{0,1\}//'; exit 2; }

# the MLX backend needs its cargo feature (MLX's CUDA build, long the first time); the binary keeps the CUDA backend
features=(); [[ " ${args[*]} " == *" mlx "* ]] && features=(--features mlx)
cargo build --release "${features[@]}" 2>&1 | grep -E '^error' -A5 && { echo "FAIL  the build"; exit 1; }
# the unit tests (CPU, seconds) with every run; a test whose model is not in the Hugging Face cache prints SKIPPED and
# returns early: that is a FAIL, not a pass
output=$(cargo test --release "${features[@]}" -- --nocapture 2>&1)
tests=$(grep -E '^test result' <<<"$output")
skipped=$(grep -oE 'SKIPPED: .*' <<<"$output" | sort -u)
if [ -n "$skipped" ]; then fail "cargo test: tests checked nothing: $(tr '\n' ' ' <<<"$skipped")"
elif [ -n "$tests" ] && ! grep -qv '^test result: ok' <<<"$tests"; then ok "cargo test: $(grep -oP "\d+(?= passed)" <<<"$tests" | awk "{ s += \$1 } END { print s }") passed"
else fail "cargo test: ${tests:-does not build}"; fi
for a in "${args[@]}"; do area "$a"; done
echo
if [ ${#FAILS[@]} -eq 0 ]; then echo "all checks OK"; else echo "${#FAILS[@]} FAILED"; exit 1; fi
