#!/usr/bin/env bash
# Lint gate: formatting, then clippy with every warning an error. Exits nonzero on any issue.
set -euo pipefail
case "${1:-}" in -h|--help|help) sed -n 2p "$0" | sed 's/^# //'; exit 0 ;; esac
cd "$(dirname "$0")/.."
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
