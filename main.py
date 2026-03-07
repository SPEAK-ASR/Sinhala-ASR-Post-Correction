import logging
import os

from config import CONFIG
from src.augmentation import SinhalaASRNoiseAugmenter
from src.data_loader import build_splits, load_all_datasets
from src.trainer import ASRCorrectionTrainer


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Authenticate with HuggingFace Hub if a token is available
    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if hf_token:
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token, add_to_git_credential=False)
        logger.info("Logged in to HuggingFace Hub")

    logger.info("Model  : %s", CONFIG.model.model_name)
    logger.info(
        "Active datasets: %s",
        [d.name for d in CONFIG.data.datasets if d.enabled],
    )

    # ── Data ────────────────────────────────────────────────────────────────
    augmenter = SinhalaASRNoiseAugmenter(noise_prob=CONFIG.data.noise_prob)
    combined_ds = load_all_datasets(CONFIG.data.datasets, augmenter)
    dataset = build_splits(combined_ds, CONFIG.data.val_split, CONFIG.data.test_split)

    for split, ds in dataset.items():
        logger.info("Split %-12s: %7d examples", split, len(ds))

    # ── Training ─────────────────────────────────────────────────────────────
    trainer = ASRCorrectionTrainer()
    trainer.load_model()
    tokenized = trainer.tokenize(dataset)
    trainer.build_trainer(tokenized)
    trainer.train()
    trainer.evaluate_test(tokenized)

    # ── Optional: push to HuggingFace Hub ────────────────────────────────────
    trainer.push_to_hub()


if __name__ == "__main__":
    main()
