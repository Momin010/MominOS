#!/bin/bash
# VM startup-script: auto-resume SFT training on every boot.
# Installed as the instance's `startup-script` metadata so that after any
# reboot (host maintenance, manual start, crash-restart) training comes back
# up on its own in a detached tmux session, resuming from the newest checkpoint.
export HOME=/root

# Don't double-launch if training is already up.
if pgrep -f sft_train.py >/dev/null 2>&1; then exit 0; fi
if tmux has-session -t sft 2>/dev/null; then exit 0; fi

# Launch the resumable trainer detached so it outlives this boot script.
tmux new-session -d -s sft 'bash /root/MominOS/training/run_sft.sh'
