#!/usr/bin/env python3
"""
Optuna hyperparameter optimisation for Sinhala ASR Post-Correction.

Design
------
- Storage  : JournalStorage (JournalFileBackend) — crash-safe, multi-process safe, resumable
- Sampler  : TPESampler(multivariate=True, constant_liar=True) — models inter-parameter
             correlations; constant_liar ensures safe coordination across parallel workers
- Pruner   : NopPruner — each trial runs exactly 1 epoch, so mid-trial pruning has no effect
- Objective: minimise eval WER (1-epoch proxy)
- Plots    : optuna.visualization (Plotly) saved as HTML to optuna_results/plots/

Parallelism
-----------
tune.sh spawns one process per GPU. Each process calls study.optimize() with
n_jobs=--workers (default 1). All processes share a single JournalFileBackend.
workers=1 is recommended for mbart-large-50 (~11 GB VRAM per trial).

Usage
-----
  # Single GPU, fresh study:
  python tune_hyperparams.py --gpu-id 0 --n-trials 50

  # Resume after interruption:
  python tune_hyperparams.py --gpu-id 0 --n-trials 50 --resume

  # Generate visualisations from an existing study (no training):
  python tune_hyperparams.py --visualize

  # Multi-GPU launch (recommended):
  bash tune.sh --workers 1 --n-trials 50 [--resume]
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from transformers import (
    DataCollatorForSeq2Seq,
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------
# Tuple encoding:
#   ("float",      low, high)      → trial.suggest_float(name, low, high)
#   ("float_log",  low, high)      → trial.suggest_float(name, low, high, log=True)
#   ("int",        low, high)      → trial.suggest_int(name, low, high)
#   ("categorical", [choices])     → trial.suggest_categorical(name, choices)
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, tuple] = {
    # Optimizer
    "learning_rate":               ("float_log",   5e-6,   2e-4),
    "weight_decay":                ("float",        0.0,    0.15),
    "adam_beta2":                  ("float",        0.980,  0.9999),
    "adam_epsilon":                ("float_log",    1e-9,   1e-6),
    "max_grad_norm":               ("float",        0.5,    5.0),
    # LR schedule
    "lr_scheduler_type":           ("categorical",  ["linear", "cosine", "cosine_with_restarts"]),
    "warmup_steps":                ("int",          0,      1000),
    # Batch size is fixed at 16; only gradient accumulation is tuned
    "gradient_accumulation_steps": ("categorical",  [1, 2, 4]),
    # Regularisation — neftune_noise_alpha=0.0 disables NEFTune (mapped to None in objective)
    "neftune_noise_alpha":         ("float",        0.0,    15.0),
    "label_smoothing_factor":      ("float",        0.0,    0.2),
    # LoRA
    "lora_r":                      ("int",          8,      256),
    "lora_alpha":                  ("int",          8,      512),
    "lora_dropout":                ("float",        0.0,    0.2),
    # Controls which attention/FFN layers receive LoRA adapters:
    #   "qv"      → query + value only (minimal, fastest)
    #   "qkvo"    → full attention projections
    #   "qkvo_fc" → full attention + feed-forward layers (maximum capacity)
    "lora_target_modules":         ("categorical",  ["qv", "qkvo", "qkvo_fc"]),
}

N_PARAMS: int = len(SEARCH_SPACE)   # 14
DEFAULT_N_JOBS: int = N_PARAMS       # informational; actual n_jobs = --workers

# ---------------------------------------------------------------------------
# LoRA target module names per model family
# ---------------------------------------------------------------------------
_TARGET_MODULE_MAP: dict[bool, dict[str, list[str]]] = {
    # mBART / BART-family
    True: {
        "qv":      ["q_proj", "v_proj"],
        "qkvo":    ["q_proj", "k_proj", "v_proj", "out_proj"],
        "qkvo_fc": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    },
    # mT5 / T5-family
    False: {
        "qv":      ["q", "v"],
        "qkvo":    ["q", "k", "v", "o"],
        "qkvo_fc": ["q", "k", "v", "o", "wi", "wo"],
    },
}

# ---------------------------------------------------------------------------
# Process-level globals (populated once by _init_globals)
# ---------------------------------------------------------------------------
_BASE_MODEL_CACHE_DIR: str | None = None
_BASE_TOKENIZER = None
_TOKENIZED: Any = None
_FORCED_BOS_TOKEN_ID: int | None = None
_USE_MBART: bool = True
_DEVICE: str = "cuda"
_COMPUTE_METRICS = None



# ---------------------------------------------------------------------------
# Hyperparameter suggestion
# ---------------------------------------------------------------------------

def _suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample one hyperparameter configuration from SEARCH_SPACE."""
    params: dict[str, Any] = {}
    for name, spec in SEARCH_SPACE.items():
        kind = spec[0]
        if kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
        elif kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec[1])
        else:
            raise ValueError(f"Unknown search space kind: {kind!r}")
    return params


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _make_compute_metrics(tokenizer):
    """Return a compute_metrics closure that reports BLEU, WER, and CER."""
    import evaluate as evaluate_lib

    bleu_metric = evaluate_lib.load("sacrebleu")
    wer_metric  = evaluate_lib.load("wer")
    cer_metric  = evaluate_lib.load("cer")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds  = [p.strip() for p in decoded_preds]
        decoded_labels = [lb.strip() for lb in decoded_labels]

        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[lb] for lb in decoded_labels],
        )
        wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "bleu": round(bleu["score"], 2),
            "wer":  round(wer, 4),
            "cer":  round(cer, 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# One-time initialisation
# ---------------------------------------------------------------------------

def _init_globals(gpu_id: str) -> None:
    """
    Load the dataset and base model once per process.

    The pretrained weights are saved to a local cache directory so that each
    trial can reload a fresh copy from disk without hitting the network.
    """
    global _BASE_MODEL_CACHE_DIR, _BASE_TOKENIZER, _TOKENIZED
    global _FORCED_BOS_TOKEN_ID, _USE_MBART, _DEVICE, _COMPUTE_METRICS

    import torch
    from config import CONFIG
    from src.augmentation import SinhalaASRNoiseAugmenter
    from src.data_loader import build_splits, load_all_datasets

    _DEVICE = "cuda" if (gpu_id != "cpu" and torch.cuda.is_available()) else "cpu"
    logger.info("Process device: %s (gpu_id=%s)", _DEVICE, gpu_id)

    set_seed(CONFIG.training.seed)

    logger.info("Loading and tokenising dataset…")
    augmenter  = SinhalaASRNoiseAugmenter(noise_prob=CONFIG.data.noise_prob)
    combined   = load_all_datasets(CONFIG.data.datasets, augmenter)
    dataset    = build_splits(combined, CONFIG.data.val_split, CONFIG.data.test_split)

    model_name = CONFIG.model.model_name
    _USE_MBART = "mbart" in model_name.lower()
    logger.info("Loading base model: %s", model_name)

    if _USE_MBART:
        _BASE_TOKENIZER = MBart50Tokenizer.from_pretrained(
            model_name,
            src_lang=CONFIG.model.src_lang,
            tgt_lang=CONFIG.model.tgt_lang,
        )
        base_model = MBartForConditionalGeneration.from_pretrained(model_name)
        _FORCED_BOS_TOKEN_ID = _BASE_TOKENIZER.lang_code_to_id[CONFIG.model.tgt_lang]
        base_model.config.forced_bos_token_id = None
        base_model.generation_config.forced_bos_token_id = _FORCED_BOS_TOKEN_ID
    else:
        from transformers import AutoTokenizer
        _BASE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        base_model = MT5ForConditionalGeneration.from_pretrained(model_name)

    cache_dir = Path("optuna_results") / "base_model_cache"
    if not cache_dir.exists():
        logger.info("Saving base model to local cache: %s", cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        base_model.save_pretrained(str(cache_dir))
        _BASE_TOKENIZER.save_pretrained(str(cache_dir))
    else:
        logger.info("Using existing base model cache: %s", cache_dir)
    _BASE_MODEL_CACHE_DIR = str(cache_dir)

    del base_model
    gc.collect()

    max_in     = CONFIG.model.max_input_length
    max_tgt    = CONFIG.model.max_target_length
    use_prefix = CONFIG.model.use_mt5_prefix
    prefix     = CONFIG.model.mt5_prefix

    def _preprocess(examples):
        srcs = examples["src"]
        tgts = examples["tgt"]
        if use_prefix:
            srcs = [prefix + s for s in srcs]
        model_inputs = _BASE_TOKENIZER(
            srcs, max_length=max_in, truncation=True, padding=False
        )
        labels = _BASE_TOKENIZER(
            text_target=tgts, max_length=max_tgt, truncation=True, padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    _TOKENIZED = dataset.map(
        _preprocess,
        batched=True,
        batch_size=1000,
        remove_columns=["src", "tgt"],
        desc="Tokenising",
    )
    _COMPUTE_METRICS = _make_compute_metrics(_BASE_TOKENIZER)
    logger.info("Initialisation complete. Ready to run trials.")


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial) -> float:
    """
    Train for one epoch with a sampled hyperparameter configuration and return
    the evaluation WER.

    One epoch is used as a low-cost proxy metric. NopPruner is set at the study
    level, so trial.report() here is a no-op for pruning but keeps intermediate
    value logging consistent for visualisation.
    """
    import torch

    params    = _suggest_params(trial)
    trial_dir = Path("optuna_results") / "trials" / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Trial %d | params: %s", trial.number, params)

    from config import CONFIG

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(trial_dir),
        # Tuned hyperparameters
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        adam_beta2=params["adam_beta2"],
        adam_epsilon=params["adam_epsilon"],
        max_grad_norm=params["max_grad_norm"],
        lr_scheduler_type=params["lr_scheduler_type"],
        warmup_steps=params["warmup_steps"],
        per_device_train_batch_size=16,
        gradient_accumulation_steps=params["gradient_accumulation_steps"],
        neftune_noise_alpha=params["neftune_noise_alpha"] or None,
        label_smoothing_factor=params["label_smoothing_factor"],
        # Fixed HPO settings
        num_train_epochs=1,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        per_device_eval_batch_size=CONFIG.training.per_device_eval_batch_size,
        bf16=CONFIG.training.bf16 and (_DEVICE == "cuda"),
        fp16=False,
        report_to="none",
        push_to_hub=False,
        hub_model_id=None,
        auto_find_batch_size=False,
        logging_strategy="epoch",
        logging_first_step=False,
        dataloader_num_workers=CONFIG.training.dataloader_num_workers,
        dataloader_prefetch_factor=CONFIG.training.dataloader_prefetch_factor,
        dataloader_pin_memory=CONFIG.training.dataloader_pin_memory,
        dataloader_persistent_workers=CONFIG.training.dataloader_persistent_workers,
        optim=CONFIG.training.optim,
        seed=CONFIG.training.seed,
        predict_with_generate=True,
        generation_max_length=CONFIG.model.max_target_length,
        label_names=["labels"],
    )

    if _USE_MBART:
        model = MBartForConditionalGeneration.from_pretrained(_BASE_MODEL_CACHE_DIR)
        model.config.forced_bos_token_id = None
        if _FORCED_BOS_TOKEN_ID is not None:
            model.generation_config.forced_bos_token_id = _FORCED_BOS_TOKEN_ID
    else:
        model = MT5ForConditionalGeneration.from_pretrained(_BASE_MODEL_CACHE_DIR)
    model = model.to(_DEVICE)

    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=params["lora_r"],
        lora_alpha=params["lora_alpha"],
        target_modules=_TARGET_MODULE_MAP[_USE_MBART][params["lora_target_modules"]],
        lora_dropout=params["lora_dropout"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=_BASE_TOKENIZER,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=_TOKENIZED["train"],
        eval_dataset=_TOKENIZED["validation"],
        processing_class=_BASE_TOKENIZER,
        data_collator=data_collator,
        compute_metrics=_COMPUTE_METRICS,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    wer = eval_metrics.get("eval_wer", float("inf"))

    trial.report(wer, step=1)
    logger.info("Trial %d finished | WER=%.4f", trial.number, wer)

    del trainer, model
    gc.collect()
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    return wer


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def generate_visualizations(study: optuna.Study, out_dir: Path) -> None:
    """
    Generate and save all standard Optuna Plotly visualisations as HTML files.

    Also writes a plain-text best_params.txt summary to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    from optuna.visualization import (
        plot_contour,
        plot_edf,
        plot_intermediate_values,
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_rank,
        plot_slice,
        plot_timeline,
    )

    plots = {
        "optimization_history":  lambda: plot_optimization_history(study),
        "param_importances":     lambda: plot_param_importances(study),
        "parallel_coordinate":   lambda: plot_parallel_coordinate(study),
        "contour":               lambda: plot_contour(study),
        "slice":                 lambda: plot_slice(study),
        "intermediate_values":   lambda: plot_intermediate_values(study),
        "edf":                   lambda: plot_edf(study),
        "rank":                  lambda: plot_rank(study),
        "timeline":              lambda: plot_timeline(study),
        "duration_importances":  lambda: plot_param_importances(
            study,
            target=lambda t: t.duration.total_seconds(),
            target_name="trial_duration_s",
        ),
    }

    for name, fn in plots.items():
        try:
            fig = fn()
            path = out_dir / f"{name}.html"
            fig.write_html(str(path))
            logger.info("Saved plot: %s", path)
        except Exception as exc:
            logger.warning("Could not generate %s: %s", name, exc)

    if study.best_trial:
        summary_path = out_dir / "best_params.txt"
        lines = [
            f"Study     : {study.study_name}",
            f"Best trial: #{study.best_trial.number}",
            f"Best WER  : {study.best_value:.4f}",
            "",
            "Best hyperparameters:",
        ]
        for k, v in study.best_params.items():
            lines.append(f"  {k:<40} = {v}")
        summary_path.write_text("\n".join(lines) + "\n")
        logger.info("Best params saved to: %s", summary_path)
        print("\n" + "\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HPO for Sinhala ASR Post-Correction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gpu-id",       default="0",
                   help="GPU index for CUDA_VISIBLE_DEVICES, or 'cpu'")
    p.add_argument("--workers",          type=int, default=1,
                   help="Parallel trials per GPU process (n_jobs for study.optimize). "
                        "Keep at 1 for mbart-large-50 (~11 GB VRAM per trial).")
    p.add_argument("--n-trials",         type=int, default=50,
                   help="Number of trials to run in this process")
    p.add_argument("--n-jobs",           type=int, default=None,
                   help="Override n_jobs directly (default: value of --workers)")
    p.add_argument("--study-name",       default="sinhala_asr_hpo",
                   help="Optuna study name (shared across all GPU processes)")
    p.add_argument("--journal-file",     default="optuna_results/journal.log",
                   help="Path to the JournalFileBackend log file")
    p.add_argument("--resume",           action="store_true",
                   help="Resume an existing study (load_if_exists=True)")
    p.add_argument("--n-startup-trials", type=int, default=2 * N_PARAMS,
                   help="Random exploration trials before TPE activates "
                        "(default: 2 × N_PARAMS)")
    p.add_argument("--visualize",        action="store_true",
                   help="Generate visualisations from an existing study; skip training")
    p.add_argument("--plot-dir",         default="optuna_results/plots",
                   help="Output directory for Plotly HTML plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.environ["WANDB_DISABLED"] = "true"
    Path("optuna_results").mkdir(exist_ok=True)

    journal_path = Path(args.journal_file)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    storage = JournalStorage(JournalFileBackend(file_path=str(journal_path)))

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
        constant_liar=True,
        seed=42,
    )
    pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=args.resume,
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    logger.info(
        "Study '%s' | completed trials: %d | direction: minimize WER",
        args.study_name, completed,
    )
    logger.info(
        "Sampler: TPESampler(multivariate=True, n_startup=%d) | Pruner: NopPruner",
        args.n_startup_trials,
    )

    if args.visualize:
        if completed == 0:
            logger.warning("No completed trials found — cannot generate visualisations.")
            return
        generate_visualizations(study, Path(args.plot_dir))
        return

    n_jobs = args.n_jobs if args.n_jobs is not None else args.workers
    logger.info("n_jobs = %d", n_jobs)
    if n_jobs > 1:
        logger.warning(
            "n_jobs=%d runs %d parallel trials on GPU %s via Python threads. "
            "Only safe when multiple model copies fit in VRAM.",
            n_jobs, n_jobs, args.gpu_id,
        )

    _init_globals(args.gpu_id)

    logger.info(
        "Starting optimisation: %d trials, n_jobs=%d, GPU=%s",
        args.n_trials, n_jobs, args.gpu_id,
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=n_jobs,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    completed_now = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_now    = len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED])
    logger.info(
        "Optimisation complete | completed: %d | pruned: %d",
        completed_now, pruned_now,
    )

    if study.best_trial:
        logger.info("Best WER   : %.4f", study.best_value)
        logger.info("Best trial : #%d", study.best_trial.number)
        for k, v in study.best_params.items():
            logger.info("  %-40s = %s", k, v)

    generate_visualizations(study, Path(args.plot_dir))


if __name__ == "__main__":
    main()
