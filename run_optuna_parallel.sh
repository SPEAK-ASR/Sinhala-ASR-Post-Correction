#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_optuna_parallel.sh — Launch parallel Optuna workers
#
# Works with a single GPU (multiple workers sharing the same device) or
# multiple GPUs (workers distributed round-robin across devices).
#
# Usage:
#   bash run_optuna_parallel.sh                          # auto workers, 50 trials
#   bash run_optuna_parallel.sh 100                      # 100 total trials, 1 worker/GPU
#   bash run_optuna_parallel.sh 100 --workers 3          # 3 concurrent workers on GPU 0
#   bash run_optuna_parallel.sh 100 --workers 4 --resume # resume study with 4 workers
#   bash run_optuna_parallel.sh 15 --resume              # resume, 1 worker per GPU
#
# Single GPU (e.g. MI300X with ~192 GB VRAM):
#   All workers receive CUDA_VISIBLE_DEVICES=0 / HIP_VISIBLE_DEVICES=0.
#   Optuna assigns trials automatically — no duplicate work.
#   Rule of thumb: workers = floor(VRAM_GB / peak_VRAM_per_trial_GB)
#   e.g.  192 GB / 40 GB ≈ 4 workers  →  bash run_optuna_parallel.sh 50 --workers 4
#
# Multi-GPU:
#   Workers are assigned round-robin: worker i → GPU (i % GPU_COUNT).
#   Default workers = GPU_COUNT (one per GPU).
#
# All workers share the same Optuna study via JournalFileStorage
# (optuna_results/journal.log), coordinating trials automatically.
#
# Workers run fully detached (survive terminal close).
# Stop all workers : bash stop_optuna.sh
# Monitor progress : tail -f logs/optuna_worker0_gpu0.log
# ---------------------------------------------------------------------------
set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
TOTAL_TRIALS=50
WORKERS=""        # --workers N  (overrides GPU auto-detection)
RESUME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            RESUME=true
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --*)
            echo "ERROR: Unknown option '$1'"
            exit 1
            ;;
        *)
            TOTAL_TRIALS="$1"
            shift
            ;;
    esac
done

# ---------------------------------------------------------------------------
# GPU detection — ROCm first, then CUDA, then assume 1
# ---------------------------------------------------------------------------
detect_gpu_count() {
    if command -v rocm-smi &>/dev/null; then
        local count
        count=$(rocm-smi --showuniqueid 2>/dev/null | grep -c 'GPU\[' || true)
        echo $(( count < 1 ? 1 : count ))
    elif command -v nvidia-smi &>/dev/null; then
        nvidia-smi -L 2>/dev/null | wc -l
    else
        echo 1
    fi
}

GPU_COUNT=$(detect_gpu_count)
GPU_COUNT=$(( GPU_COUNT < 1 ? 1 : GPU_COUNT ))   # guard against 0

# Default: one worker per GPU; user may override with --workers
TOTAL_WORKERS="${WORKERS:-$GPU_COUNT}"

if [[ "$TOTAL_WORKERS" -lt 1 ]]; then
    echo "ERROR: --workers must be >= 1"
    exit 1
fi

TRIALS_PER_WORKER=$(( (TOTAL_TRIALS + TOTAL_WORKERS - 1) / TOTAL_WORKERS ))  # ceil

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo " Optuna parallel launcher"
echo "   Total trials   : $TOTAL_TRIALS"
echo "   GPU(s) detected: $GPU_COUNT"
echo "   Total workers  : $TOTAL_WORKERS"
echo "   Trials/worker  : $TRIALS_PER_WORKER"
echo "   Resume mode    : $RESUME"
if [[ "$TOTAL_WORKERS" -gt "$GPU_COUNT" ]]; then
    WPG=$(( (TOTAL_WORKERS + GPU_COUNT - 1) / GPU_COUNT ))
    echo "   Workers/GPU    : ~$WPG  (sharing VRAM — ensure sufficient VRAM)"
fi
echo "============================================================"

# ---------------------------------------------------------------------------
# Prepare log and results directories
# ---------------------------------------------------------------------------
mkdir -p logs optuna_results

JOURNAL_FILE="optuna_results/journal.log"

if [[ "$RESUME" == true ]]; then
    echo "INFO: Resume mode — preserving existing journal ($JOURNAL_FILE)"
    if [[ ! -f "$JOURNAL_FILE" ]]; then
        echo "WARNING: No journal file found — starting fresh."
    fi
    rm -f logs/optuna_workers.pid
else
    if [[ -f "$JOURNAL_FILE" ]]; then
        echo "WARNING: $JOURNAL_FILE already exists."
        read -r -p "Delete it? This will lose your current progress. [y/N] " CONFIRM
        if [[ "${CONFIRM,,}" == "y" ]]; then
            echo "INFO: Deleting old journal and PID file."
            rm -f "$JOURNAL_FILE" logs/optuna_workers.pid
        else
            echo "INFO: Keeping existing journal. Use --resume to continue from it, or remove it manually."
            exit 0
        fi
    else
        echo "INFO: Fresh study — removing old PID file"
        rm -f logs/optuna_workers.pid
    fi
fi

# ---------------------------------------------------------------------------
# Build shared flags for tune_hyperparams.py
# ---------------------------------------------------------------------------
RESUME_FLAG=""
if [[ "$RESUME" == true ]]; then
    RESUME_FLAG="--resume"
fi

# ---------------------------------------------------------------------------
# Launch workers
# ---------------------------------------------------------------------------
for WORKER_IDX in $(seq 0 $(( TOTAL_WORKERS - 1 ))); do
    # Round-robin GPU assignment: worker i → GPU (i % GPU_COUNT)
    GPU_ID=$(( WORKER_IDX % GPU_COUNT ))
    LOG_FILE="logs/optuna_worker${WORKER_IDX}_gpu${GPU_ID}.log"

    echo "Launching worker $WORKER_IDX → GPU $GPU_ID  (log: $LOG_FILE)"

    # setsid creates a new process group so the worker is fully independent.
    # nohup keeps it alive after the terminal closes.
    # Set all three device-visibility vars for ROCm + CUDA compat.
    setsid nohup bash -c "
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        export HIP_VISIBLE_DEVICES=${GPU_ID}
        export ROCR_VISIBLE_DEVICES=${GPU_ID}
        exec python tune_hyperparams.py \
            --gpu-id ${GPU_ID} \
            --n-trials ${TRIALS_PER_WORKER} \
            --journal-file ${JOURNAL_FILE} \
            ${RESUME_FLAG}
    " >> "${LOG_FILE}" 2>&1 &

    echo $! >> logs/optuna_workers.pid
done

# Detach all launched jobs from this shell so closing the terminal won't
# send SIGHUP to the workers.
disown -a

echo "============================================================"
echo "All $TOTAL_WORKERS worker(s) launched in the background."
echo ""
echo "  PID file : logs/optuna_workers.pid"
echo "  Journal  : $JOURNAL_FILE"
echo "  Logs     : logs/optuna_worker<N>_gpu<ID>.log"
echo ""
echo "  Monitor  : tail -f logs/optuna_worker0_gpu0.log"
echo "  Status   : ps -p \$(paste -sd, logs/optuna_workers.pid)"
echo "  Stop all : bash stop_optuna.sh"
echo "============================================================"
