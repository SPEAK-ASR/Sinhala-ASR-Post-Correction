import logging
from typing import List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from config.data import DatasetEntry
from src.augmentation import SinhalaASRNoiseAugmenter
from src.text_utils import clean_text

logger = logging.getLogger(__name__)


def load_hf_pairs_dataset(
    cfg: DatasetEntry,
    augmenter: SinhalaASRNoiseAugmenter,
) -> Dataset:
    """
    Load a HuggingFace dataset as (src, tgt) pairs.

    - cfg.apply_noise=False: src_col and tgt_col are used directly.
    - cfg.apply_noise=True:  tgt_col is the clean reference; ASR noise is
      synthesised from it to produce src.
    """
    raw = load_dataset(cfg.hf_dataset_name, cfg.hf_dataset_config)
    all_splits = concatenate_datasets([raw[s] for s in raw])
    df = all_splits.to_pandas()

    df = df[[cfg.src_col, cfg.tgt_col]].dropna()
    df["tgt"] = df[cfg.tgt_col].apply(clean_text)

    if cfg.apply_noise:
        logger.info("  Injecting ASR noise into '%s' column ...", cfg.tgt_col)
        df["src"] = df["tgt"].apply(augmenter.apply)
        df = df[df["src"] != df["tgt"]].reset_index(drop=True)
        logger.info("  Generated %d noisy pairs", len(df))
    else:
        df["src"] = df[cfg.src_col].apply(clean_text)
        # Drop identical pairs — no correction needed, not useful for training
        df = df[df["src"] != df["tgt"]].reset_index(drop=True)
        logger.info("  Kept %d pairs after filtering identical src/tgt", len(df))

    return Dataset.from_pandas(df[["src", "tgt"]], preserve_index=False)


def load_all_datasets(
    dataset_entries: List[DatasetEntry],
    augmenter: SinhalaASRNoiseAugmenter,
) -> Dataset:
    """Load, optionally oversample, and concatenate all enabled datasets."""
    all_datasets: List[Dataset] = []

    for entry in dataset_entries:
        if not entry.enabled:
            logger.info("Skipping disabled dataset: %s", entry.name)
            continue

        noise_flag = "noise=ON" if entry.apply_noise else "noise=OFF"
        logger.info(
            "Loading: %s  [%s  oversample=%dx]",
            entry.name, noise_flag, entry.oversample,
        )

        ds = load_hf_pairs_dataset(entry, augmenter)
        logger.info("  %d examples loaded", len(ds))

        if entry.oversample > 1:
            ds = concatenate_datasets([ds] * entry.oversample)
            logger.info(
                "  After oversampling (%dx): %d examples",
                entry.oversample, len(ds),
            )

        all_datasets.append(ds)

    if not all_datasets:
        raise ValueError("No datasets enabled! Check DataConfig.datasets.")

    combined = concatenate_datasets(all_datasets).shuffle(seed=42)
    logger.info("Total combined examples: %d", len(combined))
    return combined


def build_splits(
    combined_ds: Dataset,
    val_split: float,
    test_split: float,
) -> DatasetDict:
    """Split combined dataset into train / validation / test."""
    split1 = combined_ds.train_test_split(test_size=test_split, seed=42)
    split2 = split1["train"].train_test_split(
        test_size=val_split / (1 - test_split), seed=42
    )
    return DatasetDict({
        "train":      split2["train"],
        "validation": split2["test"],
        "test":       split1["test"],
    })
