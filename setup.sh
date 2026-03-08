#!/usr/bin/env bash
set -euo pipefail

# One-time environment setup. Safe to re-run.

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

# ---------------------------------------------------------------------------
# Install PyTorch with the correct compute backend.
#
# Plain `pip install torch` on Linux resolves to the CPU-only wheel, which
# makes torch.cuda.is_available() return False even on GPU machines.
# We detect CUDA / ROCm here and pull the matching wheel explicitly.
# requirements.txt still carries `torch>=2.2.0` as a marker; because torch
# is already installed before that file is processed, pip just validates the
# version constraint and skips reinstalling it.
# ---------------------------------------------------------------------------
install_torch_gpu() {
    # ── ROCm (AMD) ──────────────────────────────────────────────────────────
    if command -v rocm-smi &>/dev/null; then
        # Determine ROCm major.minor (e.g. "6.0" → "rocm6.0")
        ROCM_VER=$(rocm-smi --version 2>/dev/null \
            | grep -oP 'ROCm version:\s*\K[0-9]+\.[0-9]+' \
            | head -1 || true)
        if [[ -n "$ROCM_VER" ]]; then
            # Strip patch digit so "6.2.3" → "6.2"
            ROCM_SHORT="${ROCM_VER%.*}"
            ROCM_TAG="rocm${ROCM_SHORT}"
            echo "[setup] ROCm ${ROCM_VER} detected – installing torch for ${ROCM_TAG}"
            pip install torch --index-url "https://download.pytorch.org/whl/${ROCM_TAG}"
            return 0
        fi
    fi

    # ── CUDA (NVIDIA) ────────────────────────────────────────────────────────
    if command -v nvidia-smi &>/dev/null; then
        # nvidia-smi reports the *maximum* CUDA version the driver supports.
        CUDA_DRIVER_VER=$(nvidia-smi --query-gpu=driver_version \
            --format=csv,noheader 2>/dev/null | head -1 | grep -oP '[0-9]+\.[0-9]+' || true)
        # Map driver version → CUDA toolkit version (rough heuristic)
        # Driver ≥ 525  → CUDA 12.x available;  Driver < 525 → CUDA 11.8
        CUDA_MAJOR=$(echo "$CUDA_DRIVER_VER" | cut -d. -f1)
        if [[ -n "$CUDA_MAJOR" && "$CUDA_MAJOR" -ge 525 ]]; then
            # Prefer CUDA 12.1 wheel (broadest compat inside the 12.x family)
            CUDA_TAG="cu121"
        else
            CUDA_TAG="cu118"
        fi
        echo "[setup] NVIDIA driver ${CUDA_DRIVER_VER} detected – installing torch for ${CUDA_TAG}"
        pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
        return 0
    fi

    echo "[setup] No GPU detected – installing CPU-only torch (GPU training will not work)"
    pip install "torch>=2.2.0"
}

install_torch_gpu

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
# Quick GPU sanity check so a CPU-only install is visible immediately
python3 -c "
import torch
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    names = ', '.join(torch.cuda.get_device_name(i) for i in range(n))
    print(f'[setup] GPU OK: {n} device(s) – {names}')
else:
    print('[setup] WARNING: torch.cuda.is_available() = False')
    print('        Training will run on CPU (very slow for mBART).')
    print('        Re-run setup.sh on a machine with NVIDIA/AMD drivers.')
"

echo ""
echo "Setup complete. Run:  bash start.sh"
