#!/usr/bin/env bash
# Free host RAM before resuming a full-weight 12B run, and hand it back after.
#
# Resuming costs ~28 GB more than a cold start: Trainer._load_optimizer_and_
# scheduler loads optimizer.pt with map_location="cpu", so the whole state sits
# in host RAM while it is copied into GTT. On Strix Halo the GPU shares the
# 124 GB pool, so the inference stack that systemd restarts at boot (~21 GB)
# is the difference between resuming and being OOM-killed.
#
#   sudo scripts/prepare_train_memory.sh stop
#   sudo scripts/prepare_train_memory.sh restore [--drop-swap]
set -Eeuo pipefail

SERVICES=(llama-server-llm.service llama-server-embed.service opensearch.service)
SWAPFILE=/mnt/data/swapfile
SWAPSIZE=${SWAPSIZE:-48G}

[[ $EUID -eq 0 ]] || { echo "Run with sudo." >&2; exit 1; }

ACTION=${1:-stop}
OPTION=${2:-}

report() {
  echo
  free -h
  swapon --show || true
}

case "$ACTION" in
  stop)
    for unit in "${SERVICES[@]}"; do
      if systemctl is-active --quiet "$unit"; then
        echo "stopping $unit"
        systemctl stop "$unit"
      else
        echo "already stopped: $unit"
      fi
    done

    # The swapfile only absorbs the transient CPU-side optimizer copy; once
    # training is stepping, nothing pages.
    if swapon --show=NAME --noheadings | grep -qx "$SWAPFILE"; then
      echo "swap already active: $SWAPFILE"
    else
      [[ -f "$SWAPFILE" ]] || {
        echo "creating $SWAPFILE ($SWAPSIZE)"
        fallocate -l "$SWAPSIZE" "$SWAPFILE"
        chmod 600 "$SWAPFILE"
        mkswap "$SWAPFILE" >/dev/null
      }
      swapon "$SWAPFILE"
      echo "swap enabled: $SWAPFILE"
    fi

    report
    echo
    echo "Now resume as your normal user:  ./run_p4v1.sh train"
    ;;

  restore)
    for unit in "${SERVICES[@]}"; do
      echo "starting $unit"
      systemctl start "$unit"
    done

    if [[ "$OPTION" == --drop-swap ]]; then
      if swapon --show=NAME --noheadings | grep -qx "$SWAPFILE"; then
        swapoff "$SWAPFILE"
      fi
      rm -f "$SWAPFILE"
      echo "swap removed: $SWAPFILE"
    fi

    report
    ;;

  *)
    echo "Usage: $0 [stop | restore [--drop-swap]]" >&2
    exit 2
    ;;
esac
