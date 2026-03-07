#!/usr/bin/env bash
set -euo pipefail

# Load credentials from .env if present
[ -f .env ] && set -a && source .env && set +a

mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/run_${TIMESTAMP}.log"

source .venv/bin/activate
nohup python main.py > "$LOG_FILE" 2>&1 &

echo "Started (PID $!). Logs: $LOG_FILE"
