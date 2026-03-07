---
name: notebook-to-script
description: "Convert a Jupyter notebook into a clean, robust, production-ready Python script project with config-driven architecture, setup.sh, and detached start.sh with logging."
argument-hint: "Path to the notebook to convert, e.g. training.ipynb"
agent: agent
---

Convert the provided Jupyter notebook into a clean, production-ready Python project. Do **not** over-engineer — keep it simple, focused, and scalable by configuration not abstraction.

## Output Structure

Produce the following layout (adapt names to match the notebook's domain):

```
<project_root>/
├── config/
│   ├── __init__.py          # Instantiates a single CONFIG object from all sub-configs
│   ├── model.py             # ModelConfig dataclass
│   ├── training.py          # TrainingConfig dataclass  (Seq2SeqTrainingArguments params)
│   ├── data.py              # DataConfig dataclass      (dataset paths, splits, etc.)
│   └── <domain>.py          # Any other logical config grouping from the notebook
├── src/
│   └── <domain>.py          # Core logic extracted from notebook cells (trainer, pipeline, etc.)
├── main.py                  # Entry point — wires config + src, starts training/inference
├── setup.sh                 # One-time environment setup (venv, deps, env vars)
├── start.sh                 # Runs main.py detached; all output → logs/
├── logs/                    # Created at runtime by start.sh (gitignore it)
└── requirements.txt         # Pinned dependencies from notebook imports
```

---

## Config Architecture Rules

1. **One dataclass per concern** — each `config/<name>.py` defines a single `@dataclass` with typed fields and sensible defaults.
2. **Direct unpacking into framework constructors** — never re-map keys manually:
   ```python
   # config/training.py
   from dataclasses import dataclass

   @dataclass
   class TrainingConfig:
       output_dir: str = "outputs/checkpoints"
       num_train_epochs: int = 3
       per_device_train_batch_size: int = 8
       per_device_eval_batch_size: int = 8
       learning_rate: float = 5e-5
       warmup_steps: int = 500
       weight_decay: float = 0.01
       logging_dir: str = "logs/tensorboard"
       save_total_limit: int = 2
       predict_with_generate: bool = True
       fp16: bool = True
   ```
   ```python
   # In the trainer class:
   from dataclasses import asdict
   self.training_args = Seq2SeqTrainingArguments(**asdict(CONFIG.training))
   ```
3. **Single CONFIG object** — `config/__init__.py` imports and instantiates all sub-configs:
   ```python
   from config.model import ModelConfig
   from config.training import TrainingConfig
   from config.data import DataConfig
   from dataclasses import dataclass

   @dataclass
   class Config:
       model: ModelConfig = None
       training: TrainingConfig = None
       data: DataConfig = None

       def __post_init__(self):
           self.model = ModelConfig()
           self.training = TrainingConfig()
           self.data = DataConfig()

   CONFIG = Config()
   ```
4. **Easy to change** — adding or removing a parameter means editing only the relevant `config/<name>.py` dataclass. No other files need updating for parameter changes.
5. **No YAML/JSON config files** — keep configs as Python so they are typed, autocompleted, and importable.

---

## setup.sh Rules

- Create and activate a virtualenv (or use `conda` if the notebook uses it).
- Install all requirements: `pip install -r requirements.txt`.
- Export any required environment variables (e.g. `HF_TOKEN`, `WANDB_API_KEY`) — use placeholders or source from a `.env` file that is git-ignored.
- Make the script idempotent (safe to re-run).

```bash
#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Copy env template if .env doesn't exist
[ -f .env ] || cp .env.example .env
echo "Setup complete. Edit .env with your credentials before running start.sh"
```

---

## start.sh Rules

- Run `main.py` fully **detached** from the terminal using `nohup` or equivalent.
- All stdout and stderr must go to `logs/run_<timestamp>.log`.
- Print the log file path and PID to the terminal after launching.
- Create the `logs/` directory if it does not exist.

```bash
#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/run_${TIMESTAMP}.log"

source .venv/bin/activate
nohup python main.py > "$LOG_FILE" 2>&1 &
echo "Started (PID $!). Logs: $LOG_FILE"
```

---

## Python Code Quality Rules

- **No notebooks patterns** — no `display()`, `!pip install`, magic commands, or inline plots saved to screen.
- **Logging over print** — use Python's `logging` module; configure it once in `main.py`.
- **Classes for stateful components** — trainers, pipelines, loaders; plain functions for stateless utilities.
- **No global state** outside `config/`.
- **Type hints** on all function signatures.
- **`if __name__ == "__main__":` guard** in `main.py`.

---

## Execution Instructions

1. Read the full notebook cell by cell.
2. Identify all unique config parameters (model names, hyperparameters, paths, flags) and map each to the correct config dataclass.
3. Extract logic cells into `src/` modules, grouping by responsibility.
4. Wire everything in `main.py`.
5. Write `setup.sh`, `start.sh`, and `requirements.txt`.
6. Add a `logs/` entry to `.gitignore` and a `.env.example` with placeholder values.
7. After generating all files, list any assumptions made and any parameters whose default values need user review.
