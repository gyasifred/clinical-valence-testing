#!/usr/bin/env python3
"""
ICD Code Classification Training Script

Trains a BioBERT model for multi-label ICD code classification on MIMIC-III data.
"""
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import logging
from config_loader import get_config
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MIMICDataset(Dataset):
    """Dataset for MIMIC ICD code classification"""

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.FloatTensor(label)
        }


def load_and_prepare_data(
    data_path: str,
    code_column: str = "short_codes",
    text_column: str = "text",
    min_frequency: int = 100
) -> Tuple[pd.DataFrame, Dict[str, int], List[str]]:
    """Load and prepare MIMIC data for training"""

    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Get all unique ICD codes that occur frequently enough
    all_codes = []
    for codes in df[code_column].dropna():
        if isinstance(codes, str):
            code_list = [c.strip().strip("'\"[]") for c in codes.split(',')]
            all_codes.extend([c for c in code_list if c])

    # Count code frequencies
    from collections import Counter
    code_counts = Counter(all_codes)

    # Filter codes by minimum frequency
    frequent_codes = sorted([
        code for code, count in code_counts.items()
        if count >= min_frequency
    ])

    logger.info(f"Found {len(frequent_codes)} codes with >= {min_frequency} occurrences")

    # Create label mapping
    label2id = {code: idx for idx, code in enumerate(frequent_codes)}
    id2label = {idx: code for code, idx in label2id.items()}

    # Filter dataframe to only include samples with at least one frequent code
    def has_frequent_code(codes_str):
        if pd.isna(codes_str):
            return False
        code_list = [c.strip().strip("'\"[]") for c in str(codes_str).split(',')]
        return any(c in label2id for c in code_list if c)

    df = df[df[code_column].apply(has_frequent_code)].reset_index(drop=True)
    logger.info(f"After filtering: {len(df)} samples")

    # Create multi-label encoding
    def encode_labels(codes_str):
        label_vector = np.zeros(len(frequent_codes), dtype=np.float32)
        if pd.isna(codes_str):
            return label_vector

        code_list = [c.strip().strip("'\"[]") for c in str(codes_str).split(',')]
        for code in code_list:
            if code in label2id:
                label_vector[label2id[code]] = 1.0
        return label_vector

    df['encoded_labels'] = df[code_column].apply(encode_labels)

    return df, label2id, id2label, frequent_codes


def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
    """Compute metrics for multi-label classification"""

    labels = pred.label_ids
    preds = (torch.sigmoid(torch.tensor(pred.predictions)) > 0.5).numpy()

    # Micro-averaged metrics
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average='micro', zero_division=0
    )

    # Macro-averaged metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )

    # Sample-wise metrics
    precision_samples, recall_samples, f1_samples, _ = precision_recall_fscore_support(
        labels, preds, average='samples', zero_division=0
    )

    # ROC-AUC (only for labels with both classes present)
    try:
        roc_auc_micro = roc_auc_score(labels, pred.predictions, average='micro')
        roc_auc_macro = roc_auc_score(labels, pred.predictions, average='macro')
    except ValueError:
        roc_auc_micro = 0.0
        roc_auc_macro = 0.0

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_samples': f1_samples,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'roc_auc_micro': roc_auc_micro,
        'roc_auc_macro': roc_auc_macro,
    }


def train_model(
    train_path: str,
    val_path: str = None,
    output_dir: str = "./models/icd_classifier",
    base_model: str = "bvanaken/CORe-clinical-outcome-biobert-v1",
    config_path: str = None,
    **kwargs
):
    """Train ICD code classification model"""

    # Load configuration
    config = get_config(config_path)

    # Override with kwargs
    num_epochs = kwargs.get('num_epochs', config.training.num_epochs)
    learning_rate = kwargs.get('learning_rate', config.training.learning_rate)
    batch_size = kwargs.get('batch_size', config.model.batch_size)
    max_length = kwargs.get('max_length', config.model.max_length)
    code_column = kwargs.get('code_label', config.data.code_label)
    text_column = kwargs.get('text_label', config.data.text_label)
    min_frequency = kwargs.get('min_code_frequency', config.data.min_code_frequency)
    weight_decay = kwargs.get('weight_decay', config.training.weight_decay)
    warmup_steps = kwargs.get('warmup_steps', config.training.warmup_steps)

    # Set random seeds
    if config.random_seed:
        utils.set_random_seeds(config.random_seed)

    # Load and prepare data
    logger.info("Loading training data...")
    train_df, label2id, id2label, code_list = load_and_prepare_data(
        train_path, code_column, text_column, min_frequency
    )

    # Load validation data if provided
    if val_path:
        logger.info("Loading validation data...")
        val_df, _, _, _ = load_and_prepare_data(
            val_path, code_column, text_column, min_frequency
        )
    else:
        # Split training data for validation (80/20)
        logger.info("Splitting training data for validation (80/20)...")
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            train_df, test_size=0.2, random_state=config.random_seed
        )

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Number of ICD codes: {len(code_list)}")

    # Initialize tokenizer and model
    logger.info(f"Loading tokenizer and model from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(code_list),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id
    )

    # Create datasets
    logger.info("Creating datasets...")
    train_labels = np.stack(train_df['encoded_labels'].values)
    val_labels = np.stack(val_df['encoded_labels'].values)

    train_dataset = MIMICDataset(
        train_df[text_column].tolist(),
        train_labels,
        tokenizer,
        max_length
    )

    val_dataset = MIMICDataset(
        val_df[text_column].tolist(),
        val_labels,
        tokenizer,
        max_length
    )

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_steps if warmup_steps < 1 else None,
        warmup_steps=int(warmup_steps) if warmup_steps >= 1 else 0,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        gradient_accumulation_steps=1,
        report_to=["tensorboard"],
    )

    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Evaluate
    logger.info("Evaluating on validation set...")
    eval_results = trainer.evaluate()
    logger.info(f"Validation results: {eval_results}")

    # Save final model
    final_output_dir = f"{output_dir}/final"
    logger.info(f"Saving final model to {final_output_dir}")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    # Save code list for reference
    code_list_path = Path(final_output_dir) / "icd_codes.txt"
    with open(code_list_path, 'w') as f:
        for code in code_list:
            f.write(f"{code}\n")

    logger.info(f"Training complete! Model saved to {final_output_dir}")
    logger.info(f"ICD codes saved to {code_list_path}")

    return final_output_dir, eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ICD code classification model")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation data CSV (optional)")
    parser.add_argument("--output_dir", type=str, default="./models/icd_classifier", help="Output directory")
    parser.add_argument("--base_model", type=str, default="bvanaken/CORe-clinical-outcome-biobert-v1", help="Base model")
    parser.add_argument("--config_path", type=str, default=None, help="Config file path")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")

    args = parser.parse_args()

    train_model(
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=args.output_dir,
        base_model=args.base_model,
        config_path=args.config_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
