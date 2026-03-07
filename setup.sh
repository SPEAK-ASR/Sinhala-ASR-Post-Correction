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

# Read existing values from .env as defaults (so Enter keeps them)
_existing_hf=""
_existing_wandb=""
if [[ -f .env ]]; then
    _existing_hf=$(grep "^HF_TOKEN=" .env | cut -d= -f2- | tr -d '[:space:]')
    _existing_wandb=$(grep "^WANDB_API_KEY=" .env | cut -d= -f2- | tr -d '[:space:]')
fi
# Treat placeholder strings as empty
[[ "$_existing_hf"    == "hf_your_token_here"    ]] && _existing_hf=""
[[ "$_existing_wandb" == "your_wandb_key_here"   ]] && _existing_wandb=""

# HuggingFace token
echo ""
echo "Enter your HuggingFace token (starts with 'hf_')."
echo "  → Get one at: https://huggingface.co/settings/tokens"
if [[ -n "$_existing_hf" ]]; then
    echo "  → Current value: ${_existing_hf:0:10}…  (press Enter to keep it)"
else
    echo "  → Press Enter to skip (needed for private datasets / push to Hub)."
fi
read -r -p "HuggingFace token: " HF_TOKEN_INPUT

# Weights & Biases API key
echo ""
echo "Enter your Weights & Biases API key."
echo "  → Get one at: https://wandb.ai/authorize"
if [[ -n "$_existing_wandb" ]]; then
    echo "  → Current value: ${_existing_wandb:0:10}…  (press Enter to keep it)"
else
    echo "  → Press Enter to skip (only needed if use_wandb=True in config/training.py)."
fi
read -r -p "W&B API key: " WANDB_KEY_INPUT

# Resolve final values: user input > existing > placeholder
HF_TOKEN_VAL="${HF_TOKEN_INPUT:-${_existing_hf:-hf_your_token_here}}"
WANDB_VAL="${WANDB_KEY_INPUT:-${_existing_wandb:-your_wandb_key_here}}"

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
