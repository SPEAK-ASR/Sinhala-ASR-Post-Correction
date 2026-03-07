#!/usr/bin/env bash
# tune.sh — GPU-aware launcher for Optuna HPO (Sinhala ASR Post-Correction)
#
# Detects available GPUs and spawns one optimisation process per GPU.
# All processes share a single JournalFileBackend storage so Optuna
# coordinates trial assignments automatically (no duplicate work).
#
# Usage
# -----
#   bash tune.sh [OPTIONS]
#
# Options
#   --resume            Load existing study from journal file (load_if_exists=True)
#   --workers N         Parallel trials per GPU process (default: 1)
#                       workers=1 recommended for mbart-large-50 (~11 GB VRAM / trial).
#                       Use workers>1 only on A100/H100-class GPUs with smaller models.
#   --n-jobs N          Override total n_jobs passed to study.optimize()
#                       Default: derived from N_PARAMS inside tune_hyperparams.py
#   --n-trials N        Trials to run PER GPU process (default: 50)
#   --study-name NAME   Optuna study name (default: sinhala_asr_hpo)
#   --journal-file PATH JournalStorage file path (default: optuna_results/journal.log)
#   --plot-dir PATH     Output directory for Plotly HTML plots
#                       (default: optuna_results/plots)
#
# Examples
#   bash tune.sh                                # auto-detect GPUs, 50 trials each
#   bash tune.sh --resume --workers 1           # resume across all GPUs
#   bash tune.sh --n-trials 100 --workers 1     # 100 trials per GPU
#   bash tune.sh --n-jobs 4 --n-trials 20       # force 4 n_jobs, 20 trials
#
# After the run, generate/refresh plots:
#   python tune_hyperparams.py --visualize

set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
WORKERS=1
N_TRIALS=50
N_JOBS_OVERRIDE=""          # empty = let tune_hyperparams.py derive from N_PARAMS
RESUME=false
STUDY_NAME="sinhala_asr_hpo"
JOURNAL_FILE="optuna_results/journal.log"
PLOT_DIR="optuna_results/plots"

# ─── Argument parsing ──────────────────────────────────────────────────────────
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
        --n-jobs)
            N_JOBS_OVERRIDE="$2"
            shift 2
            ;;
        --n-trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --study-name)
            STUDY_NAME="$2"
            shift 2
            ;;
        --journal-file)
            JOURNAL_FILE="$2"
            shift 2
            ;;
        --plot-dir)
            PLOT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,40p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "[tune.sh] ERROR: Unknown option: $1" >&2
            echo "          Run 'bash tune.sh --help' for usage." >&2
            exit 1
            ;;
    esac
done

# ─── GPU detection ─────────────────────────────────────────────────────────────
GPU_IDS=()

if command -v nvidia-smi &>/dev/null; then
    mapfile -t GPU_IDS < <(nvidia-smi --list-gpus | awk '{print NR-1}')
fi

N_GPUS="${#GPU_IDS[@]}"

if [[ "$N_GPUS" -eq 0 ]]; then
    echo "[tune.sh] WARNING: No NVIDIA GPUs detected — running on CPU."
    GPU_IDS=("cpu")
    N_GPUS=1
else
    echo "[tune.sh] Detected ${N_GPUS} GPU(s): ${GPU_IDS[*]}"
fi

# ─── Show configuration ───────────────────────────────────────────────────────
echo ""
echo "[tune.sh] ── Configuration ─────────────────────────────────────────────"
echo "[tune.sh]   Study name   : ${STUDY_NAME}"
echo "[tune.sh]   Journal file : ${JOURNAL_FILE}"
echo "[tune.sh]   Plot dir     : ${PLOT_DIR}"
echo "[tune.sh]   GPUs         : ${N_GPUS}  (${GPU_IDS[*]})"
echo "[tune.sh]   Workers/GPU  : ${WORKERS}"
echo "[tune.sh]   Trials/GPU   : ${N_TRIALS}"
echo "[tune.sh]   n-jobs ovrd  : ${N_JOBS_OVERRIDE:-'(auto from N_PARAMS)'}"
echo "[tune.sh]   Resume       : ${RESUME}"
echo "[tune.sh] ────────────────────────────────────────────────────────────────"
echo ""

# ─── Effective concurrency warning ────────────────────────────────────────────
TOTAL_WORKERS=$(( N_GPUS * WORKERS ))
echo "[tune.sh] Effective parallel trials: ${TOTAL_WORKERS} (${N_GPUS} GPU × ${WORKERS} worker)"
if [[ "$WORKERS" -gt 1 ]]; then
    echo "[tune.sh] WARNING: workers>1 uses Python threading on each GPU."
    echo "[tune.sh]          Only safe for models that fit multiple copies in VRAM."
fi
echo ""

# ─── Create output dirs ────────────────────────────────────────────────────────
mkdir -p "$(dirname "${JOURNAL_FILE}")" "${PLOT_DIR}" optuna_results/trials

# ─── Build common Python args ──────────────────────────────────────────────────
COMMON_ARGS=(
    --workers        "${WORKERS}"
    --n-trials       "${N_TRIALS}"
    --study-name     "${STUDY_NAME}"
    --journal-file   "${JOURNAL_FILE}"
    --plot-dir       "${PLOT_DIR}"
)
[[ "${RESUME}" == "true" ]]         && COMMON_ARGS+=(--resume)
[[ -n "${N_JOBS_OVERRIDE}" ]]       && COMMON_ARGS+=(--n-jobs "${N_JOBS_OVERRIDE}")

# ─── Launch one process per GPU ────────────────────────────────────────────────
declare -a PIDS=()
declare -a LOGS=()

for GPU_ID in "${GPU_IDS[@]}"; do
    LOG_FILE="optuna_results/gpu_${GPU_ID}.log"
    LOGS+=("${LOG_FILE}")

    if [[ "${GPU_ID}" == "cpu" ]]; then
        GPU_ENV="CUDA_VISIBLE_DEVICES="
    else
        GPU_ENV="CUDA_VISIBLE_DEVICES=${GPU_ID}"
    fi

    echo "[tune.sh] Launching GPU ${GPU_ID} → ${LOG_FILE}"

    env "${GPU_ENV}" python tune_hyperparams.py \
        --gpu-id "${GPU_ID}" \
        "${COMMON_ARGS[@]}" \
        2>&1 | tee "${LOG_FILE}" &

    PIDS+=($!)
done

echo ""
echo "[tune.sh] ${N_GPUS} worker process(es) running (PIDs: ${PIDS[*]})."
echo "[tune.sh] Streaming logs to:"
for LOG in "${LOGS[@]}"; do
    echo "[tune.sh]   ${LOG}"
done
echo ""

# ─── Wait for all processes ────────────────────────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    GPU_ID="${GPU_IDS[$i]}"
    if wait "${PID}"; then
        echo "[tune.sh] GPU ${GPU_ID} (PID ${PID}) finished successfully."
    else
        echo "[tune.sh] WARNING: GPU ${GPU_ID} (PID ${PID}) exited with non-zero status." >&2
        FAILED=$(( FAILED + 1 ))
    fi
done

# ─── Post-run summary ──────────────────────────────────────────────────────────
echo ""
echo "[tune.sh] ── Run complete ────────────────────────────────────────────────"
echo "[tune.sh]   Results  : optuna_results/"
echo "[tune.sh]   Plots    : ${PLOT_DIR}/"
echo "[tune.sh]   Journal  : ${JOURNAL_FILE}"
if [[ "${FAILED}" -gt 0 ]]; then
    echo "[tune.sh]   WARNING: ${FAILED} process(es) exited with errors. Check logs above."
fi
echo ""
echo "[tune.sh] To generate / refresh visualisations:"
echo "[tune.sh]   python tune_hyperparams.py --visualize \\"
echo "[tune.sh]     --journal-file ${JOURNAL_FILE} --plot-dir ${PLOT_DIR}"
echo ""
echo "[tune.sh] To view best parameters:"
echo "[tune.sh]   cat ${PLOT_DIR}/best_params.txt"

[[ "${FAILED}" -gt 0 ]] && exit 1
exit 0
