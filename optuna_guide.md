# Optuna HPO Guide — Sinhala ASR Post-Correction

Complete reference for running, resuming, and analysing hyperparameter optimisation
studies for the `facebook/mbart-large-50` Seq2Seq post-correction model.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Search Space (14 parameters)](#3-search-space-14-parameters)
4. [LoRA Rank & Alpha Deep Dive](#4-lora-rank--alpha-deep-dive)
5. [Optuna Design Decisions](#5-optuna-design-decisions)
6. [Quick Start Commands](#6-quick-start-commands)
7. [Multi-GPU Usage](#7-multi-gpu-usage)
8. [Resume & Fault Tolerance](#8-resume--fault-tolerance)
9. [Visualising Results](#9-visualising-results)
10. [Analysing Best Hyperparameters](#10-analysing-best-hyperparameters)
11. [Applying Best Params to Full Training](#11-applying-best-params-to-full-training)
12. [Troubleshooting](#12-troubleshooting)
13. [Command Reference (all flags)](#13-command-reference-all-flags)

---

## 1. Architecture Overview

| Component | Detail |
|-----------|--------|
| Model | `facebook/mbart-large-50` (~600M params) |
| PEFT | LoRA on `q_proj` + `v_proj` |
| Framework | HuggingFace `Seq2SeqTrainer` |
| HPO engine | Optuna — TPESampler + NopPruner |
| Storage | `JournalStorage(JournalFileBackend)` — crash-safe, multi-process |
| Objective | Minimise **WER** (Word Error Rate) after 1 training epoch |
| Scripts | `tune_hyperparams.py` (Python) · `tune.sh` (GPU launcher) |
| Outputs | `optuna_results/journal.log` · `optuna_results/plots/*.html` |

Each trial trains for **exactly 1 epoch** and reports its validation WER as the
objective. A 1-epoch proxy is fast enough to rank configurations reliably, and
using NopPruner (no within-trial pruning) keeps the evaluation signal clean.

---

## 2. Prerequisites

```bash
# Install all dependencies (includes optuna, peft, transformers, evaluate)
bash setup.sh

# Verify Optuna is available
python -c "import optuna; print(optuna.__version__)"

# Check GPU availability
nvidia-smi --list-gpus
```

Required packages (from `requirements.txt`):
- `optuna >= 3.5`
- `optuna-integration` (for Plotly visualisations)
- `transformers >= 4.40`
- `peft >= 0.10`
- `evaluate`
- `sacrebleu`, `jiwer`  (WER / CER / BLEU metrics)

---

## 3. Search Space (14 Parameters)

| # | Parameter | Type | Range / Choices | Rationale |
|---|-----------|------|-----------------|-----------|
| 1 | `learning_rate` | float (log) | [5e-6, 2e-4] | AdamW initial LR |
| 2 | `weight_decay` | float | [0.0, 0.15] | L2 regularisation |
| 3 | `adam_beta2` | float | [0.980, 0.9999] | Momentum for squared gradients |
| 4 | `adam_epsilon` | float (log) | [1e-9, 1e-6] | Numerical stability |
| 5 | `max_grad_norm` | float | [0.5, 5.0] | Gradient clipping threshold |
| 6 | `lr_scheduler_type` | categorical | linear, cosine, cosine_with_restarts | LR schedule shape |
| 7 | `warmup_ratio` | float | [0.01, 0.20] | Fraction of steps for LR warm-up |
| 8 | `per_device_train_batch_size` | categorical | 8, 16, 32 | Per-GPU batch size |
| 9 | `gradient_accumulation_steps` | categorical | 1, 2, 4 | Effective batch = batch × GAS |
| 10 | `neftune_noise_alpha` | categorical | None, 5.0, 10.0, 15.0 | NEFTune embedding noise |
| 11 | `label_smoothing_factor` | categorical | 0.0, 0.05, 0.1, 0.2 | Cross-entropy label smoothing |
| 12 | `lora_r` | categorical | 4, 6, 8, 12, 16, 24, 32 | LoRA matrix rank |
| 13 | `lora_alpha_ratio` | categorical | 0.5, 1.0, 2.0, 4.0 | LoRA scaling (alpha = ratio × r) |
| 14 | `lora_dropout` | float | [0.0, 0.2] | LoRA adapter dropout |

**Effective batch size** = `per_device_train_batch_size × gradient_accumulation_steps × n_gpus`

**Sampler**: TPESampler with `n_startup_trials = 2 × 14 = 28` random trials before
multivariate TPE activates.

---

## 4. LoRA Rank & Alpha Deep Dive

### Why not restrict to powers-of-2?

LoRA rank (`r`) controls the size of the two low-rank matrices injected into each
attention layer. Mathematically, any positive integer works. Powers-of-2 (4, 8, 16…)
are a *convenience convention*, not a hardware requirement. CUDA tensor cores align on
multiples of 8 for performance, but the weight-update computation with PEFT LoRA is
dominated by forward/backward passes through the base model, not the adapter matrices.

Including `r ∈ {4, 6, 8, 12, 16, 24, 32}` gives TPE finer granularity to find the
optimal capacity/efficiency trade-off.

### Why not hardcode `lora_alpha = 2 × lora_r`?

The effective LoRA scaling multiplier is `alpha / r`. Fixing `alpha = 2r` means the
scaling is always 2.0, regardless of `r`. But this is simply *one popular default*:

| `lora_alpha_ratio` | Effective scaling (alpha/r) | Interpretation |
|--------------------|----------------------------|----------------|
| 0.5 | 0.5 | Conservative — adapter weights have half impact |
| 1.0 | 1.0 | Identity scaling (original LoRA paper default) |
| 2.0 | 2.0 | "Double the LR" convention (HuggingFace default) |
| 4.0 | 4.0 | Aggressive — adapter weights dominate updates |

The best value depends on the base model, task difficulty, and rank. By tuning
`lora_alpha_ratio` independently from `lora_r`, TPE can discover combinations like
`r=8, ratio=4.0` (high scaling, low rank) vs `r=24, ratio=0.5` (high rank, low
scaling) that fixed conventions would never explore.

The actual integer alpha used in each trial is computed as:
```python
lora_alpha = max(1, round(lora_r * lora_alpha_ratio))
```

---

## 5. Optuna Design Decisions

### Sampler — `TPESampler(multivariate=True, constant_liar=True)`

- **TPE** is the recommended algorithm for mixed categorical/continuous spaces
  (see `optuna_docs/003_efficient_optimization_algorithms.py`).
- `multivariate=True` — models correlations between hyperparameters (e.g., the
  interaction between `learning_rate` and `lr_scheduler_type`). Critical for DL.
- `constant_liar=True` — when multiple workers are running in parallel, pending
  trials are treated as if they already returned the current best value. This
  prevents workers from all sampling the same region while waiting for results.
- `n_startup_trials=2×N_PARAMS=28` — ensures 28 random trials before TPE
  switches on; too few random samples makes the surrogate model unreliable.

### Pruner — `NopPruner`

Each trial runs exactly 1 epoch with no intermediate checkpoints.
Hyperband / MedianPruner require multiple intermediate values within a trial to
decide whether to prune. Since there is only one value (end-of-epoch WER), these
pruners would have nothing to act on. `NopPruner` keeps the API consistent.

If you increase `num_train_epochs` per trial (e.g. for a more accurate proxy), you
can switch to `HyperbandPruner(min_resource=1, max_resource=N, reduction_factor=3)`.

### Storage — `JournalStorage(JournalFileBackend)`

- Append-only file log: crash-safe, no database required.
- Fully multi-process safe — all GPU workers share one file.
- Resumable with `--resume` / `load_if_exists=True`.
- Human-readable JSON; easy to inspect with `cat optuna_results/journal.log`.

---

## 6. Quick Start Commands

### Single GPU, fresh study (50 trials)
```bash
python tune_hyperparams.py --gpu-id 0 --n-trials 50
```

### Single GPU, verbose logging
```bash
python tune_hyperparams.py --gpu-id 0 --n-trials 50 2>&1 | tee optuna_results/gpu_0.log
```

### Use `tune.sh` (auto-detects GPUs, recommended)
```bash
# Auto-detect all GPUs, 50 trials each, workers=1 (safe for mbart)
bash tune.sh

# Specify trial count
bash tune.sh --n-trials 100

# Run on CPU only (for testing / debugging)
python tune_hyperparams.py --gpu-id cpu --n-trials 3
```

### Test that everything works (2 quick trials)
```bash
python tune_hyperparams.py --gpu-id 0 --n-trials 2
```

---

## 7. Multi-GPU Usage

`tune.sh` detects all NVIDIA GPUs and spawns one Python process per GPU.
All processes share the same `journal.log`, so Optuna coordinates trial
assignments with no duplicate work.

```bash
# 4 GPUs, 50 trials each = 200 total trials
bash tune.sh --n-trials 50

# Verify GPU assignment
nvidia-smi --list-gpus
```

**workers=1 (default) is strongly recommended for mbart-large-50** — the model
requires ~11 GB VRAM per trial, leaving no room for concurrent trials on a 24 GB
GPU. Only increase workers on A100/H100 (80 GB) with a smaller model (e.g. mT5-small):

```bash
# Only on A100/H100 with mt5-small (~300 MB VRAM per trial)
bash tune.sh --workers 4 --n-trials 30
```

To completely override the `n_jobs` passed to `study.optimize()`:
```bash
bash tune.sh --n-jobs 2 --n-trials 50
```

### Manually launching per-GPU processes

If you prefer not to use `tune.sh`:
```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python tune_hyperparams.py \
    --gpu-id 0 --n-trials 50 --resume &

# GPU 1
CUDA_VISIBLE_DEVICES=1 python tune_hyperparams.py \
    --gpu-id 1 --n-trials 50 --resume &

wait
```

---

## 8. Resume & Fault Tolerance

The `JournalFileBackend` is append-only, so every completed trial is persisted
immediately. If a run is interrupted (OOM, cluster preemption, `Ctrl+C`):

```bash
# Resume from where it left off
bash tune.sh --resume --n-trials 50

# Or directly
python tune_hyperparams.py --gpu-id 0 --n-trials 50 --resume
```

With `--resume`, Optuna loads the existing study (`load_if_exists=True`) and
issues only the remaining `--n-trials` trials. Already-completed trials are
not re-run.

### Inspecting the journal
```bash
# Count completed trials
python -c "
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
storage = JournalStorage(JournalFileBackend('optuna_results/journal.log'))
study = optuna.load_study(study_name='sinhala_asr_hpo', storage=storage)
done = [t for t in study.trials if t.state.name == 'COMPLETE']
print(f'Completed: {len(done)} / {len(study.trials)} trials')
print(f'Best WER : {study.best_value:.4f}  (trial #{study.best_trial.number})')
"
```

### Changing study name (start a new study with same storage file)
```bash
python tune_hyperparams.py --gpu-id 0 --n-trials 50 \
    --study-name sinhala_asr_hpo_v2
```

---

## 9. Visualising Results

Generate all Plotly HTML visualisations without running new trials:
```bash
python tune_hyperparams.py --visualize
```

With a custom journal / plot directory:
```bash
python tune_hyperparams.py --visualize \
    --journal-file optuna_results/journal.log \
    --plot-dir optuna_results/plots
```

### Generated plots

| Filename | What it shows |
|----------|---------------|
| `optimization_history.html` | Objective WER over trials — convergence curve |
| `param_importances.html` | Which parameters matter most (Fanova method) |
| `parallel_coordinate.html` | All 14 params across all trials — spot patterns |
| `contour.html` | 2D interaction maps between param pairs |
| `slice.html` | Marginal effect of each parameter individually |
| `intermediate_values.html` | Per-epoch metrics (useful if epochs > 1) |
| `edf.html` | Empirical distribution of objective values |
| `rank.html` | Trial rank vs parameter values |
| `timeline.html` | Trial scheduling and wall-clock timing |
| `duration_importances.html` | Which params affect trial *duration* most |
| `best_params.txt` | Plain text summary of best hyperparameters |

Open any `.html` file in a browser for an interactive Plotly chart.

---

## 10. Analysing Best Hyperparameters

### Quick best-params summary
```bash
cat optuna_results/plots/best_params.txt
```

### Python analysis
```python
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

storage = JournalStorage(JournalFileBackend("optuna_results/journal.log"))
study = optuna.load_study(study_name="sinhala_asr_hpo", storage=storage)

print(f"Best WER  : {study.best_value:.4f}")
print(f"Best trial: #{study.best_trial.number}")
print("\nBest hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k:<40} = {v}")

# Top-5 trials
trials_df = study.trials_dataframe()
top5 = trials_df.sort_values("value").head(5)
print("\nTop-5 trials:")
print(top5[["number", "value"] + [c for c in top5.columns if c.startswith("params_")]])
```

### Computed `lora_alpha` from best params
```python
best = study.best_params
lora_alpha = max(1, round(best["lora_r"] * best["lora_alpha_ratio"]))
print(f"lora_r={best['lora_r']}  lora_alpha_ratio={best['lora_alpha_ratio']}  → lora_alpha={lora_alpha}")
```

---

## 11. Applying Best Params to Full Training

After the HPO study finishes, copy the best values into the config files:

### `config/training.py` (optimizer / schedule / batch settings)
```python
# Example — replace with your actual best params:
learning_rate: float = 3.2e-5          # best["learning_rate"]
weight_decay: float = 0.08             # best["weight_decay"]
warmup_steps: float = 200              # compute from best["warmup_ratio"] × total_steps
per_device_train_batch_size: int = 16  # best["per_device_train_batch_size"]
gradient_accumulation_steps: int = 2   # best["gradient_accumulation_steps"]
neftune_noise_alpha: ... = 10.0        # best["neftune_noise_alpha"]
label_smoothing_factor: float = 0.05   # best["label_smoothing_factor"]
```

### `config/model.py` (LoRA settings)
```python
lora_r: int = 12               # best["lora_r"]
lora_alpha: int = 24           # max(1, round(lora_r * best["lora_alpha_ratio"]))
lora_dropout: float = 0.08     # best["lora_dropout"]
```

Then run full training:
```bash
python main.py
```

---

## 12. Troubleshooting

### `CUDA out of memory` during a trial
- Reduce `per_device_train_batch_size` in `SEARCH_SPACE` (remove 32, keep 8/16)
- Set `workers=1` (default): `bash tune.sh --workers 1`
- Enable `auto_find_batch_size=True` in `objective()` as a last resort

### `optuna.exceptions.DuplicatedStudyError`
You're running a fresh study when one already exists in the journal. Use `--resume`:
```bash
python tune_hyperparams.py --gpu-id 0 --n-trials 50 --resume
```
Or change `--study-name` to start fresh.

### Trial returns `WER = inf`
The `eval_wer` key is missing from eval metrics. Check that:
1. `predict_with_generate=True` is set in `training_args`
2. The tokenizer decodes correctly (check `forced_bos_token_id`)
3. The validation split is non-empty

### Visualisations fail with `ValueError: No completed trials`
You need at least 1 completed trial. Run with `--n-trials 1` first to verify the
pipeline works end-to-end.

### `param_importances` plot raises an error
Fanova requires at least `n_startup_trials` completed trials (default 28) to fit
the importance model. This is expected early in the study.

### Slow first trial
The first trial downloads and caches the base model. Subsequent trials load from
`optuna_results/base_model_cache/` which is much faster.

### W&B logging appears during HPO
The `WANDB_DISABLED=true` env var is set in `main()`. If you still see W&B output,
run: `export WANDB_DISABLED=true` before starting.

---

## 13. Command Reference (all flags)

### `tune_hyperparams.py`

```
usage: tune_hyperparams.py [-h] [--gpu-id GPU_ID] [--workers WORKERS]
                           [--n-trials N_TRIALS] [--n-jobs N_JOBS]
                           [--study-name STUDY_NAME]
                           [--journal-file JOURNAL_FILE]
                           [--resume] [--n-startup-trials N_STARTUP_TRIALS]
                           [--visualize] [--plot-dir PLOT_DIR]

Flags:
  --gpu-id GPU_ID               GPU index for CUDA_VISIBLE_DEVICES, or 'cpu'
                                Default: 0
  --workers WORKERS             Parallel trials per GPU (n_jobs for study.optimize).
                                Default: 1  (recommended for mbart-large-50)
  --n-trials N_TRIALS           Trials to run in this process.
                                Default: 50
  --n-jobs N_JOBS               Override n_jobs directly (ignores --workers).
  --study-name STUDY_NAME       Optuna study name shared across all processes.
                                Default: sinhala_asr_hpo
  --journal-file JOURNAL_FILE   Path to JournalFileBackend log.
                                Default: optuna_results/journal.log
  --resume                      Load existing study (load_if_exists=True).
  --n-startup-trials N          Random trials before TPE activates.
                                Default: 2 × N_PARAMS = 28
  --visualize                   Generate plots only; no training.
  --plot-dir PLOT_DIR           Output dir for Plotly HTML files.
                                Default: optuna_results/plots
```

### `tune.sh`

```
usage: bash tune.sh [OPTIONS]

Options:
  --resume               Load existing study from journal file.
  --workers N            Parallel trials per GPU process. Default: 1
  --n-jobs N             Override n_jobs passed to study.optimize().
  --n-trials N           Trials per GPU process. Default: 50
  --study-name NAME      Optuna study name. Default: sinhala_asr_hpo
  --journal-file PATH    JournalStorage file. Default: optuna_results/journal.log
  --plot-dir PATH        Plotly HTML output directory.
                         Default: optuna_results/plots
  -h / --help            Show this help message.
```

### Common recipes

```bash
# Fresh study, single GPU
python tune_hyperparams.py --gpu-id 0 --n-trials 50

# Fresh study, all GPUs (auto-detect)
bash tune.sh --n-trials 50

# Resume after interruption
bash tune.sh --resume --n-trials 50

# Resume with custom journal
bash tune.sh --resume --journal-file optuna_results/run2/journal.log --n-trials 30

# Only generate plots (no training)
python tune_hyperparams.py --visualize

# Get best params inline
python -c "
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
s = JournalStorage(JournalFileBackend('optuna_results/journal.log'))
study = optuna.load_study(study_name='sinhala_asr_hpo', storage=s)
print('Best WER:', study.best_value)
for k,v in study.best_params.items(): print(f'  {k}: {v}')
"

# Run a quick sanity check (2 trials, CPU)
python tune_hyperparams.py --gpu-id cpu --n-trials 2 --study-name sanity_check

# Start a completely new study (don't overwrite old journal)
python tune_hyperparams.py --gpu-id 0 --n-trials 50 \
    --study-name sinhala_asr_hpo_v2 \
    --journal-file optuna_results/journal_v2.log

# Multi-GPU with 100 trials each, verbose logs per GPU
bash tune.sh --n-trials 100 --resume 2>&1
```

---

## File Layout

```
optuna_results/
├── journal.log              ← shared study storage (all trials, all GPUs)
├── base_model_cache/        ← one-time base model weights cache (fast reload)
├── trials/
│   ├── trial_0000/          ← per-trial output_dir (no checkpoints saved)
│   ├── trial_0001/
│   └── ...
├── plots/
│   ├── optimization_history.html
│   ├── param_importances.html
│   ├── parallel_coordinate.html
│   ├── contour.html
│   ├── slice.html
│   ├── intermediate_values.html
│   ├── edf.html
│   ├── rank.html
│   ├── timeline.html
│   ├── duration_importances.html
│   └── best_params.txt      ← plain-text best hyperparameter summary
├── gpu_0.log                ← stdout/stderr for GPU 0 process
└── gpu_1.log                ← stdout/stderr for GPU 1 process
```
