import logging
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import evaluate as evaluate_lib
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    MBart50Tokenizer,
    MBartForConditionalGeneration,
    MT5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from config import CONFIG

logger = logging.getLogger(__name__)

# These TrainingConfig fields are not Seq2SeqTrainingArguments parameters
_TRAINING_ARGS_EXCLUDE = {"early_stopping_patience", "use_wandb", "wandb_project"}


def _build_seq2seq_training_args(device: str) -> Seq2SeqTrainingArguments:
    args_dict = {
        k: v for k, v in asdict(CONFIG.training).items()
        if k not in _TRAINING_ARGS_EXCLUDE
    }
    # fp16 is only valid on CUDA
    args_dict["fp16"] = CONFIG.training.fp16 and (device == "cuda")
    args_dict["report_to"] = "wandb" if CONFIG.training.use_wandb else "none"
    return Seq2SeqTrainingArguments(**args_dict)


class ASRCorrectionTrainer:
    """Trains the Sinhala ASR post-correction seq2seq model."""

    def __init__(self) -> None:
        set_seed(CONFIG.training.seed)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._use_mbart = "mbart" in CONFIG.model.model_name.lower()
        self._forced_bos_token_id: Optional[int] = None
        self.model = None
        self.tokenizer = None
        self.trainer: Optional[Seq2SeqTrainer] = None
        logger.info("Using device: %s", self.device)

    def load_model(self) -> None:
        model_name = CONFIG.model.model_name
        logger.info("Loading model: %s", model_name)

        if self._use_mbart:
            self.tokenizer = MBart50Tokenizer.from_pretrained(
                model_name,
                src_lang=CONFIG.model.src_lang,
                tgt_lang=CONFIG.model.tgt_lang,
            )
            self.model = MBartForConditionalGeneration.from_pretrained(model_name)
            self._forced_bos_token_id = self.tokenizer.lang_code_to_id[CONFIG.model.tgt_lang]
            self.model.config.forced_bos_token_id = None
            self.model.generation_config.forced_bos_token_id = self._forced_bos_token_id
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = MT5ForConditionalGeneration.from_pretrained(model_name)

        self.model = self.model.to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info("Model loaded — %.0fM parameters", param_count)

    def tokenize(self, dataset: DatasetDict) -> DatasetDict:
        max_in = CONFIG.model.max_input_length
        max_tgt = CONFIG.model.max_target_length
        use_prefix = CONFIG.model.use_mt5_prefix
        prefix = CONFIG.model.mt5_prefix

        def preprocess(examples):
            srcs = examples["src"]
            tgts = examples["tgt"]
            if use_prefix:
                srcs = [prefix + s for s in srcs]

            model_inputs = self.tokenizer(
                srcs,
                max_length=max_in,
                truncation=True,
                padding=False,
            )
            labels = self.tokenizer(
                text_target=tgts,
                max_length=max_tgt,
                truncation=True,
                padding=False,
            )
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        logger.info("Tokenizing datasets...")
        tokenized = dataset.map(
            preprocess,
            batched=True,
            batch_size=1000,
            remove_columns=["src", "tgt"],
            desc="Tokenizing",
        )
        logger.info("Tokenization complete")
        return tokenized

    def _compute_metrics(self, eval_preds):
        bleu_metric = evaluate_lib.load("sacrebleu")
        wer_metric = evaluate_lib.load("wer")
        cer_metric = evaluate_lib.load("cer")

        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = [p.strip() for p in decoded_preds]
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

    def build_trainer(self, tokenized: DatasetDict) -> None:
        if CONFIG.training.use_wandb:
            import wandb
            wandb.init(project=CONFIG.training.wandb_project)
        else:
            os.environ["WANDB_DISABLED"] = "true"

        training_args = _build_seq2seq_training_args(self.device)

        if self._forced_bos_token_id is not None:
            self.model.generation_config.forced_bos_token_id = self._forced_bos_token_id

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["validation"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=CONFIG.training.early_stopping_patience,
            )],
        )
        logger.info("Trainer ready")

    def train(self) -> None:
        logger.info("Starting training...")
        train_result = self.trainer.train()

        output_dir = CONFIG.training.output_dir
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.trainer.log_metrics("train", train_result.metrics)
        self.trainer.save_metrics("train", train_result.metrics)
        logger.info("Training complete. Model saved to: %s", output_dir)

    def evaluate_test(self, tokenized: DatasetDict) -> None:
        logger.info("Evaluating on test set...")
        test_results = self.trainer.evaluate(
            tokenized["test"], metric_key_prefix="test"
        )
        logger.info("Test Results: %s", test_results)
        self.trainer.save_metrics("test", test_results)

    def push_to_hub(self) -> None:
        if not CONFIG.data.push_to_hub:
            return
        repo = CONFIG.data.hf_push_repo
        logger.info("Pushing model to HuggingFace Hub: %s", repo)
        self.trainer.push_to_hub(repo)
        self.tokenizer.push_to_hub(repo)
        logger.info("Model pushed to: https://huggingface.co/%s", repo)
