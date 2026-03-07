from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetEntry:
    name: str
    hf_dataset_name: str
    src_col: str
    tgt_col: str
    hf_dataset_config: Optional[str] = None
    enabled: bool = True
    # apply_noise=False: dataset already contains (noisy_src, clean_tgt) pairs
    # apply_noise=True:  synthesise ASR noise from the clean tgt column
    apply_noise: bool = False
    oversample: int = 1         # repeat this dataset N× to balance with others


@dataclass
class DataConfig:
    val_split: float = 0.05
    test_split: float = 0.05
    noise_prob: float = 0.35    # probability used by SinhalaASRNoiseAugmenter

    # Set push_to_hub=True and fill hf_push_repo to upload after training
    push_to_hub: bool = False
    hf_push_repo: str = "SPEAK-PP/sinhala-asr-corrector-v1"

    datasets: List[DatasetEntry] = field(default_factory=lambda: [
        DatasetEntry(
            name="openslr_sinhala_spell_correction",
            hf_dataset_name="SPEAK-PP/openslr-sinhala-spelling-correction-prediction-reference-60000",
            hf_dataset_config=None,
            src_col="dyslexic_sentence",
            tgt_col="clean_sentence",
            enabled=True,
            apply_noise=False,
            oversample=1,
        ),
        # ── Example: add a second dataset and oversample it 3× ──────────────
        # DatasetEntry(
        #     name="my_clean_corpus_synthetic",
        #     hf_dataset_name="SPEAK-PP/some-clean-sinhala-dataset",
        #     src_col="text",     # clean text; noise injected to create src
        #     tgt_col="text",     # same column used as clean reference target
        #     enabled=False,
        #     apply_noise=True,
        #     oversample=3,
        # ),
    ])
