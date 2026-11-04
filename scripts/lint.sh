#!/usr/bin/env bash
# CI-grade lint gate: formatting, then clippy with every warning an error. Exits nonzero on any issue.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
