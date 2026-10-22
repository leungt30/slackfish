#!/bin/bash
# Keep a copy of a training run's last.pt every 20k steps (last.pt is overwritten every --save-every
# steps), so a run can later be resumed from an earlier point: before a bad stretch, or before a WSD
# decay to train it longer. Run it next to training:  scripts/snapshot-checkpoints.sh Models/Weights/NAME &
# Copies last.pt to snapshot_step<N>.pt when step N of epoch 1 is a multiple of 20,000 (from epoch 2
# on: snapshot_e<E>_step<N>.pt), and to snapshot_stepend.pt / snapshot_e<E>_stepend.pt at the end of
# an epoch; exits after the last epoch.
out=$1; seen=""; mtime=""
every=${SNAPSHOT_EVERY:-20000}; poll=${SNAPSHOT_POLL:-30}   # (overridable for testing)
while true; do
  m=$(stat -c %Y "$out/last.pt" 2>/dev/null)
  if [ -n "$m" ] && [ "$m" != "$mtime" ]; then
    mtime=$m
    # epoch, step within it ("end" once the epoch is done), and the run's number of epochs
    read -r epoch step epochs < <(~/.local/bin/uv run python -c "import torch; c = torch.load('$out/last.pt', weights_only=False, map_location='cpu'); print(c['epoch'], c['step'] or 'end', c['args']['epochs'])" 2>/dev/null)
    if [ -n "$step" ] && [ "$epoch:$step" != "$seen" ]; then
      seen=$epoch:$step
      name=$([ "$epoch" = 1 ] && echo snapshot_step$step || echo snapshot_e${epoch}_step$step)
      if [ "$step" = end ] || [ $((step % every)) -eq 0 ]; then cp "$out/last.pt" "$out/$name.pt"; echo "snapshot $name"; fi
      [ "$step" = end ] && [ "$epoch" -ge "$epochs" ] && exit 0
    fi
  fi
  sleep $poll
done
