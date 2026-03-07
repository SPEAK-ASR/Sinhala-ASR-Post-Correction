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
