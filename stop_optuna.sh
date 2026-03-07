#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# stop_optuna.sh — Gracefully stop all Optuna workers launched by
#                  run_optuna_parallel.sh
#
# Usage:
#   bash stop_optuna.sh          # SIGTERM (graceful)
#   bash stop_optuna.sh --force  # SIGKILL (immediate)
# ---------------------------------------------------------------------------
set -euo pipefail

PID_FILE="logs/optuna_workers.pid"
SIGNAL="TERM"

if [[ "${1:-}" == "--force" ]]; then
    SIGNAL="KILL"
fi

if [[ ! -f "$PID_FILE" ]]; then
    echo "INFO: No PID file found at $PID_FILE — nothing to stop."
    exit 0
fi

STOPPED=0
MISSING=0

while IFS= read -r PID; do
    [[ -z "$PID" ]] && continue
    if kill -0 "$PID" 2>/dev/null; then
        kill -"$SIGNAL" "$PID" 2>/dev/null && echo "Sent SIG${SIGNAL} to PID $PID" || true
        STOPPED=$(( STOPPED + 1 ))
    else
        echo "PID $PID not running (already finished?)"
        MISSING=$(( MISSING + 1 ))
    fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "Done. Stopped: $STOPPED  Already-gone: $MISSING"
