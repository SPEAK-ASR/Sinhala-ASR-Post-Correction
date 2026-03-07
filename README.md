# Sinhala ASR Post-Correction

A seq2seq fine-tuning pipeline that corrects errors in Sinhala Automatic Speech Recognition (ASR) output. It trains multilingual encoder-decoder models (mBART-50, mT5, IndicBART) to transform noisy ASR transcriptions into clean Sinhala text.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running Training](#running-training)
- [Evaluation Metrics](#evaluation-metrics)
- [Data Pipeline](#data-pipeline)
- [Noise Augmentation](#noise-augmentation)
- [Publishing to Hugging Face Hub](#publishing-to-hugging-face-hub)
- [Requirements](#requirements)

---

## Overview

Sinhala ASR systems (e.g., Whisper) often introduce characteristic errors: dropped vowel diacritics, substituted similar-looking characters, missing punctuation, and spoken-form numbers. This project trains a sequence-to-sequence correction model that takes noisy ASR output as input and produces clean Sinhala text.

**Key features:**
- Supports multiple pre-trained multilingual models with native Sinhala (`si_LK`) support
- Configurable noise augmentation to synthesise ASR-style errors from clean text
- Multi-dataset loading with per-dataset oversampling
- Early stopping, FP16 training (CUDA), and W&B logging
- One-command training with automatic checkpointing and optional Hub upload

---

## Project Structure

```
.
├── main.py                  # Entry point — data, training, evaluation
├── requirements.txt
├── setup.sh                 # One-time environment setup
├── start.sh                 # Launch training in background
├── config/
│   ├── __init__.py          # Exports unified CONFIG object
│   ├── data.py              # DataConfig, DatasetEntry
│   ├── model.py             # ModelConfig
│   └── training.py          # TrainingConfig (Seq2SeqTrainingArguments)
└── src/
    ├── augmentation.py      # SinhalaASRNoiseAugmenter
    ├── corrector.py         # Inference helper
    ├── data_loader.py       # Dataset loading and splitting
    ├── text_utils.py        # Sinhala text cleaning utilities
    └── trainer.py           # ASRCorrectionTrainer (load, tokenize, train, eval)
```

---

## Supported Models

| Model | Notes |
|---|---|
| `facebook/mbart-large-50` | **Recommended** — native `si_LK` token support |
| `google/mt5-small` | Lightweight, faster to train |
| `google/mt5-base` | Balanced size and quality |
| `ai4bharat/IndicBART` | Strong on Indic scripts |

Set `use_mt5_prefix: True` in `ModelConfig` when using any mT5 variant.

---

## Setup

### 1. Clone and create the environment

```bash
git clone <repo-url>
cd Sinhala-ASR-Post-Correction
bash setup.sh
```

`setup.sh` creates a `.venv` virtual environment, installs all dependencies from `requirements.txt`, and copies `.env.example` → `.env` if it does not already exist.

### 2. Configure credentials

Edit `.env` and add any required tokens:

```env
HF_TOKEN=hf_...        # Hugging Face token (required only to push to Hub)
WANDB_API_KEY=...      # Optional, for Weights & Biases logging
```

---

## Configuration

All settings live in `config/`. Edit the relevant dataclass before running.

### `config/model.py` — `ModelConfig`

| Field | Default | Description |
|---|---|---|
| `model_name` | `facebook/mbart-large-50` | Pre-trained model identifier |
| `src_lang` | `si_LK` | Source language code |
| `tgt_lang` | `si_LK` | Target language code |
| `max_input_length` | `256` | Maximum tokenised input length |
| `max_target_length` | `256` | Maximum tokenised target length |
| `use_mt5_prefix` | `False` | Set `True` for mT5 models |

### `config/data.py` — `DataConfig` / `DatasetEntry`

| Field | Default | Description |
|---|---|---|
| `val_split` | `0.05` | Fraction of data used for validation |
| `test_split` | `0.05` | Fraction of data used for test |
| `noise_prob` | `0.35` | Probability applied by the noise augmenter |
| `push_to_hub` | `False` | Push the trained model to Hugging Face Hub |
| `hf_push_repo` | `SPEAK-PP/sinhala-asr-corrector-v1` | Target Hub repository |

Each `DatasetEntry` in the `datasets` list accepts:

| Field | Description |
|---|---|
| `name` | Descriptive identifier |
| `hf_dataset_name` | Hugging Face dataset path |
| `src_col` / `tgt_col` | Column names for noisy input and clean target |
| `apply_noise` | If `True`, synthesise noise from `tgt_col` instead of using `src_col` |
| `oversample` | Repeat this dataset N× to balance with others |
| `enabled` | Toggle dataset without removing the entry |

### `config/training.py` — `TrainingConfig`

Key fields (all map to `Seq2SeqTrainingArguments`):

| Field | Default | Description |
|---|---|---|
| `output_dir` | `checkpoints/sinhala-asr-correction` | Checkpoint directory |
| `per_device_train_batch_size` | `32` | Training batch size per GPU |
| `per_device_eval_batch_size` | `16` | Evaluation batch size per GPU |
| `learning_rate` | `5e-5` | AdamW initial learning rate |
| `weight_decay` | `0.01` | AdamW weight decay |
| `eval_strategy` | `epoch` | Evaluation frequency (`epoch` / `steps` / `no`) |
| `early_stopping_patience` | — | Stop after N non-improving evals |
| `fp16` | — | Mixed-precision training (CUDA only) |
| `use_wandb` | `False` | Enable W&B experiment tracking |
| `wandb_project` | — | W&B project name |

---

## Running Training

### Background (recommended for long runs)

```bash
bash start.sh
```

Activates `.venv`, launches `main.py` in the background, and writes timestamped logs to `logs/`.

### Foreground

```bash
source .venv/bin/activate
python main.py
```

---

## Evaluation Metrics

After each epoch (and on the held-out test set at the end), the trainer reports:

| Metric | Description |
|---|---|
| **BLEU** | Translation quality score (sacrebleu) |
| **WER** | Word Error Rate (jiwer) |
| **CER** | Character Error Rate (jiwer) |

Results are saved alongside the checkpoint as `test_results.json`.

---

## Data Pipeline

1. **Load** — each enabled `DatasetEntry` is fetched from Hugging Face with `datasets.load_dataset`.
2. **Clean** — `clean_text()` in `src/text_utils.py` normalises Unicode and whitespace.
3. **Augment** (optional) — if `apply_noise=True`, `SinhalaASRNoiseAugmenter` generates synthetic noisy inputs from clean targets.
4. **Filter** — identical `(src, tgt)` pairs are dropped (no correction needed).
5. **Oversample** — datasets are repeated according to their `oversample` factor before concatenation.
6. **Split** — the combined dataset is split into `train` / `validation` / `test` according to `val_split` and `test_split`.

---

## Noise Augmentation

`SinhalaASRNoiseAugmenter` simulates common Whisper ASR errors on clean Sinhala text:

| Error type | Description |
|---|---|
| Vowel sign dropping | Randomly removes diacritical vowel marks |
| Similar character substitution | Replaces characters with visually/phonetically similar ones (e.g. `ස` → `ශ`) |
| Punctuation removal | Strips punctuation marks |
| Number expansion | Converts digits to their spoken Sinhala form (e.g. `5` → `පහ`) |
| Filler word insertion | Inserts spoken fillers (e.g. `ඇ`, `හ්ම්`) at random positions |

The overall intensity is controlled by `noise_prob` in `DataConfig`.

---

## Publishing to Hugging Face Hub

Set the following in `config/data.py` and provide a valid `HF_TOKEN` in `.env`:

```python
push_to_hub: bool = True
hf_push_repo: str = "your-org/your-model-name"
```

The model and tokenizer are pushed automatically after training completes.

---

## Requirements

Python 3.10+ and the following packages (see `requirements.txt`):

```
torch>=2.2.0
transformers>=4.40.0
datasets>=2.19.0
sentencepiece>=0.2.0
sacrebleu>=2.4.0
jiwer>=3.0.0
evaluate>=0.4.0
accelerate>=0.30.0
wandb>=0.17.0
huggingface-hub>=0.23.0
numpy>=1.26.0
pandas>=2.2.0
```