import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Tuple, Dict, Optional, Union
import utils
from tqdm import tqdm
import csv
from pathlib import Path
import logging
import json
from datetime import datetime
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict

@dataclass
class PredictionState:
    """Tracks the state of prediction progress for checkpointing"""
    current_group: str
    shift_type: str 
    processed_samples: int
    total_samples: int
    last_note_id: int
    timestamp: str = datetime.now().isoformat()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class PredictionError(Exception):
    """Custom exception for prediction-related errors"""
    pass

class FileHandler:
    """Manages file operations with proper resource handling"""
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.files = {}
        self.writers = {}
        self._lock = threading.Lock()

    @contextmanager
    def get_writer(self, file_type: str, headers: List[str]):
        """Thread-safe context manager for CSV writers"""
        with self._lock:
            if file_type not in self.files:
                self.files[file_type] = open(f"{self.base_path}_{file_type}.csv", 'a', newline='')
                self.writers[file_type] = csv.DictWriter(self.files[file_type], fieldnames=headers)
                
                if self.files[file_type].tell() == 0:
                    self.writers[file_type].writeheader()
            
            try:
                yield self.writers[file_type]
            except Exception as e:
                logging.error(f"Error writing to {file_type} file: {str(e)}")
                raise

    def close_all(self):
        """Safely close all open files"""
        with self._lock:
            for file in self.files.values():
                file.close()
            self.files.clear()
            self.writers.clear()

class Predictor:
    """Base class for a generic predictor"""
    def predict_group(self, samples: List[str], group_name: str):
        raise NotImplementedError

    def save_results(self, save_path: str):
        raise NotImplementedError

class TransformerPredictor(Predictor):
    def __init__(self, checkpoint_path: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            output_attentions=True,
            output_hidden_states=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    @torch.no_grad()
    def inference_from_texts(self, text: str, layer_num: int, head_num: int, aggregation: str) -> Tuple[List[float], List[str], torch.Tensor]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
        self.model.eval()
        outputs = self.model(**inputs, output_attentions=True)
        
        attentions = outputs.attentions[layer_num][0][head_num].cpu().numpy()
        cls_attention = attentions[0, :][1:-1]
        words = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[1:-1]

        final_attention_weights, input_words = [], []
        current_word, current_attention_sum, attention_count = "", 0.0, 0

        for i, token in enumerate(words):
            if token.startswith('##'):
                current_word += token[2:]
                if aggregation == "sum":
                    current_attention_sum += cls_attention[i]
                elif aggregation == "average":
                    current_attention_sum += cls_attention[i]
                    attention_count += 1
                elif aggregation == "max":
                    current_attention_sum = max(current_attention_sum, cls_attention[i])
            else:
                if current_word:
                    if aggregation == "average" and attention_count > 0:
                        final_attention_weights.append(current_attention_sum / attention_count)
                    else:
                        final_attention_weights.append(current_attention_sum)
                    input_words.append(current_word)

                current_word = token
                current_attention_sum = cls_attention[i]
                attention_count = 1

        if current_word:
            if aggregation == "average" and attention_count > 0:
                final_attention_weights.append(current_attention_sum / attention_count)
            else:
                final_attention_weights.append(current_attention_sum)
            input_words.append(current_word)

        return final_attention_weights, input_words, outputs.logits.detach()

class DiagnosisPredictor(TransformerPredictor):
    """Predictor class specifically for diagnostic predictions with improved file organization"""
    
    def __init__(self, checkpoint_path: str, test_set_path: str, gpu: bool = False, 
                 code_label: str = "short_codes", batch_size: int = 128, 
                 checkpoint_interval: int = 1000, head_num: int = 11, 
                 layer_num: int = 11):
        """
        Initialize the DiagnosisPredictor
        
        Args:
            checkpoint_path: Path to model checkpoint
            test_set_path: Path to test dataset
            gpu: Whether to use GPU
            code_label: Column name for codes in dataset
            batch_size: Batch size for predictions
            checkpoint_interval: Interval for saving checkpoints
            head_num: Attention head number to use
            layer_num: Model layer number to use
        """
        super().__init__(checkpoint_path)
        self.gpu = gpu
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        self.head_num = head_num
        self.layer_num = layer_num
        self.code_filter = utils.codes_that_occur_n_times_in_dataset(
            n=100, dataset_path=test_set_path, code_label=code_label
        )
        self.label_list = list(self.model.config.label2id.keys())
        self.label_list_filter = [self.label_list.index(label) for label in self.code_filter]
        self.file_handlers = {}
        self.current_state = None
        self.save_path = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_shift_base_name(self, group_name: str) -> str:
        """
        Extract base shift name from group name
        
        Args:
            group_name: Name of the current group
            
        Returns:
            Base shift type name
        """
        shift_types = {
            "pejorative": ["non_compliant","uncooperative","resistant","difficult"],
            "laudatory": ["compliant", "cooperative","pleasant","respectful"],
            "neutralize": ["no_mention"],
            "neutralval": ["neutral"]
        }
        
        for base_name, variants in shift_types.items():
            if any(variant in group_name.lower() for variant in variants):
                return base_name
        return "default"

    def _save_checkpoint(self, save_path: Path):
        """
        Save current prediction state
        
        Args:
            save_path: Path to save checkpoint
        """
        if self.current_state:
            checkpoint_path = save_path.parent / 'checkpoints'
            checkpoint_path.mkdir(exist_ok=True)
            checkpoint_file = checkpoint_path / f"checkpoint_{self.current_state.timestamp}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(self.current_state.to_dict(), f)

    def _load_checkpoint(self, save_path: Path) -> Optional[PredictionState]:
        """
        Load most recent checkpoint if it exists
        
        Args:
            save_path: Path to load checkpoint from
            
        Returns:
            PredictionState if checkpoint exists, None otherwise
        """
        checkpoint_path = save_path.parent / 'checkpoints'
        if not checkpoint_path.exists():
            return None

        checkpoints = list(checkpoint_path.glob("checkpoint_*.json"))
        if not checkpoints:
            return None

        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        with open(latest_checkpoint, 'r') as f:
            state_dict = json.load(f)
            return PredictionState.from_dict(state_dict)

    def initialize_for_prediction(self, save_path: str):
        """
        Initialize predictor with save path and file handlers
        
        Args:
            save_path: Path to save results
        """
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_state = self._load_checkpoint(self.save_path)

    def _get_file_handler(self, group_name: str) -> FileHandler:
        """
        Get or create file handler for specific shift type
        
        Args:
            group_name: Name of the current group
            
        Returns:
            FileHandler for the corresponding shift type
        """
        shift_base = self._get_shift_base_name(group_name)
        
        if shift_base not in self.file_handlers:
            base_path = self.save_path.parent / f"{shift_base}_{self.timestamp}"
            self.file_handlers[shift_base] = FileHandler(base_path)
            
        return self.file_handlers[shift_base]

    def predict_batch(self, batch_texts: List[str], note_ids: List[int], group_name: str):
        """
        Process a batch of texts with error handling
        
        Args:
            batch_texts: List of texts to process
            note_ids: List of note IDs
            group_name: Name of the current group
        """
        try:
            file_handler = self._get_file_handler(group_name)
            shift_type = self._get_shift_base_name(group_name)
            
            for note_id, sample in zip(note_ids, batch_texts):
                attention_weights, words, logits = super().inference_from_texts(
                    sample, 
                    layer_num=self.layer_num, 
                    head_num=self.head_num, 
                    aggregation="sum"
                )
                
                diagnosis_probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                prob_per_label = [diagnosis_probs[i] for i in self.label_list_filter]
                
                with file_handler.get_writer('diagnosis', 
                    ["NoteID", "Valence", "Val_class"] + self.code_filter) as diag_writer:
                    diag_row = dict(zip(self.code_filter, prob_per_label))
                    diag_row["Valence"] = group_name
                    diag_row["NoteID"] = note_id
                    diag_row["Val_class"] = shift_type 
                    diag_writer.writerow(diag_row)
                
                with file_handler.get_writer('attention',
                    ["NoteID", "Word", "AttentionWeight", "Valence", "Val_class"]) as attn_writer:
                    for word, weight in zip(words, attention_weights):
                        attn_writer.writerow({
                            "NoteID": note_id,
                            "Word": word,
                            "AttentionWeight": float(weight),
                            "Valence": group_name,
                            "Val_class": shift_type 
                        })
                
                with file_handler.get_writer('clinical_notes',
                    ["NoteID", "ClinicalNote", "Valence", "Val_class"]) as note_writer:
                    note_writer.writerow({
                        "NoteID": note_id,
                        "ClinicalNote": sample,
                        "Valence": group_name,
                        "Val_class": shift_type  
                    })

        except Exception as e:
            raise PredictionError(f"Batch processing failed: {str(e)}")

    def predict_group(self, samples: List[str], group_name: str):
        """
        Process all samples in a group
        
        Args:
            samples: List of samples to process
            group_name: Name of the group
        """
        if self.save_path is None:
            self.save_path = Path("predictions_default.csv")
            self.initialize_for_prediction(str(self.save_path))
            
        shift_type = self._get_shift_base_name(group_name) 
        num_samples = len(samples)
        self.current_state = PredictionState(
            current_group=group_name,
            shift_type=shift_type, 
            processed_samples=0,
            total_samples=num_samples,
            last_note_id=-1
        )
        
        batch_size = self.batch_size
        for start_idx in tqdm(range(0, num_samples, batch_size)):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = samples[start_idx:end_idx]
            note_ids = range(start_idx, end_idx)
            
            self.predict_batch(batch_samples, note_ids, group_name)
            self.current_state.processed_samples += len(batch_samples)
            self.current_state.last_note_id = note_ids[-1]
            
            if (start_idx + batch_size) % self.checkpoint_interval == 0:
                self._save_checkpoint(self.save_path)

    def save_results(self, save_path: str):
        """
        Save all results and clean up resources
        
        Args:
            save_path: Path to save results
        """
        if not self.file_handlers:
            self.initialize_for_prediction(save_path)
            
        try:
            self._save_checkpoint(self.save_path)
            # Close all file handlers
            for handler in self.file_handlers.values():
                handler.close_all()
            self.file_handlers.clear()
            
        except Exception as e:
            raise PredictionError(f"Failed to save results: {str(e)}")