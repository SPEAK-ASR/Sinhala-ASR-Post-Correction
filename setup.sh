#!/usr/bin/env bash
set -euo pipefail

# One-time environment setup. Safe to re-run.

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Create .env from template if it doesn't exist yet
[ -f .env ] || cp .env.example .env

echo ""
echo "Setup complete."
echo "Edit .env with your credentials before running start.sh"
