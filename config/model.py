from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Options:
    #   "facebook/mbart-large-50"   ← recommended, native si_LK support
    #   "google/mt5-small"          ← lighter, faster to train
    #   "google/mt5-base"           ← balanced
    #   "ai4bharat/IndicBART"       ← strong on Indic scripts
    model_name: str = "facebook/mbart-large-50"
    src_lang: str = "si_LK"
    tgt_lang: str = "si_LK"
    max_input_length: int = 256
    max_target_length: int = 256
    # Set to True if using mT5 (uses prompt prefix instead of lang tokens)
    use_mt5_prefix: bool = False
    mt5_prefix: str = "correct sinhala asr: "

    # LoRA / PEFT settings
    use_lora: bool = True
    lora_r: int = 16          # rank of the low-rank matrices; non-power-of-2 (e.g. 12, 24) is fine
    lora_alpha: int = 32      # LoRA scaling factor; effective scale = alpha/r.
                               # 2×r is a common default but NOT universally optimal.
                               # HPO tunes this via lora_alpha_ratio in tune_hyperparams.py.
    lora_dropout: float = 0.05
