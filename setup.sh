#!/usr/bin/env bash
set -euo pipefail

# One-time environment setup. Safe to re-run.

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# ─── Collect credentials and write .env ───────────────────────────────────────
echo ""
echo "─── API Token Setup ────────────────────────────────────────────────────────"

# HuggingFace token
if [[ -f .env ]] && grep -q "^HF_TOKEN=hf_" .env 2>/dev/null; then
    echo "[setup] HuggingFace token already set in .env — skipping."
else
    echo ""
    echo "Enter your HuggingFace token (starts with 'hf_')."
    echo "  → Get one at: https://huggingface.co/settings/tokens"
    echo "  → Press Enter to skip (needed for private datasets / push to Hub)."
    read -r -p "HuggingFace token: " HF_TOKEN_INPUT
fi

# Weights & Biases API key
if [[ -f .env ]] && grep -q "^WANDB_API_KEY=[^y]" .env 2>/dev/null; then
    echo "[setup] W&B API key already set in .env — skipping."
else
    echo ""
    echo "Enter your Weights & Biases API key."
    echo "  → Get one at: https://wandb.ai/authorize"
    echo "  → Press Enter to skip (only needed if use_wandb=True in config/training.py)."
    read -r -p "W&B API key: " WANDB_KEY_INPUT
fi

# Write .env
HF_TOKEN_VAL="${HF_TOKEN_INPUT:-hf_your_token_here}"
WANDB_VAL="${WANDB_KEY_INPUT:-your_wandb_key_here}"

cat > .env <<EOF
# HuggingFace token (needed for private datasets / higher rate limits / push to Hub)
HF_TOKEN=${HF_TOKEN_VAL}

# Weights & Biases API key (only needed if use_wandb=True in config/training.py)
WANDB_API_KEY=${WANDB_VAL}
EOF

echo ""
echo "[setup] .env written."
if [[ "${HF_TOKEN_VAL}" == "hf_your_token_here" ]]; then
    echo "[setup] WARNING: HuggingFace token not set. Edit .env before running start.sh"
fi
if [[ "${WANDB_VAL}" == "your_wandb_key_here" ]]; then
    echo "[setup] NOTE: W&B key not set. Set it in .env if you enable use_wandb=True."
fi

echo ""
echo "Setup complete. Run:  bash start.sh"
