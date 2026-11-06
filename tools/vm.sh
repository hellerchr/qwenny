#!/bin/bash
# Sync the sources to the VM and run a command there in ~/qwenny (cargo and nvcc on PATH): tools/vm.sh <command...>
set -e
case "${1:-}" in ""|-h|--help|help) sed -n 2p "$0" | sed 's/^# //'; exit 0 ;; esac
VM=${VM:-ubuntu@192.168.31.153}
rsync -a --delete --exclude target --exclude .git --exclude .claude --exclude __pycache__ "$(dirname "$0")/../" "$VM:qwenny/"
ssh "$VM" "cd ~/qwenny && export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH && $*"
