#!/usr/bin/env bash
# Lint gate: formatting, clippy with every warning an error, then the command line's documentation. Nonzero on any issue.
set -euo pipefail
case "${1:-}" in -h|--help|help) sed -n 2p "$0" | sed 's/^# //'; exit 0 ;; esac
cd "$(dirname "$0")/.."
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
# every command, argument and option described, the help within 120 columns, every option read (src/cli.rs)
cargo test --release --all-features -- cli:: abliterate::tests::the_help_states_the_pass_bar
