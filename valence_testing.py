from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict
from test_shifts.base_shift import BaseShift
from prediction import Predictor

@dataclass
class TestResults:
    """Stores results from behavioral testing"""
    shift_statistics: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        return {
            "shift_statistics": self.shift_statistics,
            "timestamp": self.timestamp
        }

class BehavioralTestingError(Exception):
    """Custom exception for behavioral testing errors"""
    pass

class BehavioralTesting:
    """
    Implements behavioral testing for diagnostic predictions
    """
    
    def __init__(self, test_dataset_path: str, text_label: str = "text"):
        """
        Initialize behavioral testing framework
        
        Args:
            test_dataset_path: Path to test dataset CSV
            text_label: Column name containing text samples
        """
        self.test_dataset_path = Path(test_dataset_path)
        self.text_label = text_label
        
        try:
            self.test_df = pd.read_csv(self.test_dataset_path)
            if self.text_label not in self.test_df.columns:
                raise BehavioralTestingError(
                    f"Text column '{text_label}' not found in dataset"
                )
        except Exception as e:
            raise BehavioralTestingError(
                f"Failed to load test dataset: {str(e)}"
            )
    
    def _validate_inputs(self, shift: BaseShift, predictor: Predictor, save_path: str):
        """
        Validate inputs before running tests
        """
        if not isinstance(shift, BaseShift):
            raise BehavioralTestingError("shift must be instance of BaseShift")
            
        if not isinstance(predictor, Predictor):
            raise BehavioralTestingError("predictor must be instance of Predictor")
            
        save_dir = Path(save_path).parent
        if not save_dir.exists():
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise BehavioralTestingError(
                    f"Failed to create save directory: {str(e)}"
                )

    def _apply_shift(self, shift: BaseShift):
        """
        Apply shift transformation to test samples
        """
        try:
            shift_groups = shift.get_group_names()
            # Convert numpy array to list to avoid boolean ambiguity
            samples_list = self.test_df[self.text_label].tolist()
            shifted_samples, stats = shift.make_shift(
                samples=samples_list,
                return_stats=True
            )
            return dict(zip(shift_groups, shifted_samples)), stats
            
        except Exception as e:
            raise BehavioralTestingError(
                f"Failed to apply shift transformation: {str(e)}"
            )
    
    def run_test(self, shift, predictor, save_path):
        """
        Run behavioral test with given shift and predictor
        
        Args:
            shift: Shift transformation to apply
            predictor: Model predictor instance  
            save_path: Path to save results
            
        Returns:
            TestResults containing shift statistics
        """
        try:
            self._validate_inputs(shift, predictor, save_path)
            
            # Initialize predictor with save path
            predictor.initialize_for_prediction(save_path)
            
            groups, stats = self._apply_shift(shift)
            
            for group_name, samples in groups.items():
                predictor.predict_group(samples, group_name)
                
            predictor.save_results(save_path)
            
            return TestResults(shift_statistics=stats)
            
        except Exception as e:
            raise BehavioralTestingError(
                f"Behavioral test failed: {str(e)}"
            ) from e