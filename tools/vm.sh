#!/bin/bash
# Sync the sources to the VM and run a command there in ~/qwenny (cargo and nvcc on PATH).
set -e
VM=${VM:-ubuntu@192.168.31.153}
rsync -a --delete --exclude target --exclude .git "$(dirname "$0")/../" "$VM:qwenny/"
ssh "$VM" "cd ~/qwenny && export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH && $*"
