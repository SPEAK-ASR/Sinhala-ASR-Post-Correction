import logging
from typing import List

import torch
from transformers import (
    AutoTokenizer,
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
)

from config import CONFIG

logger = logging.getLogger(__name__)


class SinhalaASRCorrector:
    """Inference wrapper for the fine-tuned ASR post-correction model."""

    def __init__(self, model_path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._use_mbart = "mbart" in CONFIG.model.model_name.lower()
        logger.info("Loading corrector from: %s", model_path)

        if self._use_mbart:
            self.tokenizer = MBart50Tokenizer.from_pretrained(
                model_path,
                src_lang=CONFIG.model.src_lang,
                tgt_lang=CONFIG.model.tgt_lang,
            )
            self.model = MBartForConditionalGeneration.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_path)

        self.model = self.model.to(self.device).eval()
        logger.info("Corrector ready")

    @torch.no_grad()
    def correct(self, texts: List[str], num_beams: int = 4) -> List[str]:
        if CONFIG.model.use_mt5_prefix:
            texts = [CONFIG.model.mt5_prefix + t for t in texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=CONFIG.model.max_input_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        gen_kwargs = dict(
            num_beams=num_beams,
            max_length=CONFIG.model.max_target_length,
            early_stopping=True,
        )
        if self._use_mbart:
            gen_kwargs["forced_bos_token_id"] = (
                self.tokenizer.lang_code_to_id[CONFIG.model.tgt_lang]
            )

        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
