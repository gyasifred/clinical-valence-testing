#!/usr/bin/env python3
import os
from typing import Union
import fire
import utils
from valence_testing import BehavioralTesting
from prediction import DiagnosisPredictor
from test_shifts.laudatory_shift import LaudatoryShift
from test_shifts.neutralize_shift import NeutralizeShift
from test_shifts.neutralVal_shift import NeutralValShift
from test_shifts.pejorative_shift import PejorativeShift

# Map of available shifts and tasks
SHIFT_MAP = {
    "neutralize": NeutralizeShift(),
    "pejorative": PejorativeShift(),
    "laud": LaudatoryShift(),
    "neutralval": NeutralValShift()
}

TASK_MAP = {
    "diagnosis": DiagnosisPredictor
}

def parse_argument(arg: Union[str, tuple]) -> str:
    """Parse a single argument that might be a tuple into a string"""
    if isinstance(arg, tuple):
        return str(arg[0])
    return str(arg)

def parse_shift_keys(shift_keys: Union[str, tuple]) -> list:
    """Parse shift keys from various input formats"""
    if isinstance(shift_keys, tuple):
        # Handle tuple of strings with possible commas
        shifts = []
        for item in shift_keys:
            shifts.extend(str(item).strip().rstrip(',').split(','))
        return [s.strip() for s in shifts if s.strip()]
    elif isinstance(shift_keys, str):
        return [s.strip() for s in shift_keys.split(',') if s.strip()]
    return list(shift_keys)

def run(
    test_set_path: str, 
    model_path: str, 
    shift_keys: Union[str, tuple], 
    task: str, 
    save_dir: Union[str, tuple] = "./results",
    gpu: bool = False,
    batch_size: int = 128,
    head_num: int = 11,
    layer_num: int = 11,
    code_label: str = "short_codes",
    checkpoint_interval: int = 1000
):
    """
    Run behavioral tests with specified shifts and predictor
    
    Args:
        test_set_path: Path to test dataset
        model_path: Path to model checkpoint
        shift_keys: Comma-separated string or list of shift names
        task: Name of prediction task
        save_dir: Directory to save results
        gpu: Whether to use GPU
        batch_size: Batch size for predictions
        head_num: Attention head number to use
        layer_num: Model layer number to use
        code_label: Column name for codes in dataset
        checkpoint_interval: Interval for saving checkpoints
    """
    # Parse arguments
    save_dir = parse_argument(save_dir)
    test_set_path = parse_argument(test_set_path)
    model_path = parse_argument(model_path)
    task = parse_argument(task)
    shift_keys = parse_shift_keys(shift_keys)
    
    print(f"Running with shifts: {shift_keys}")
    
    # Initialize predictor and testing framework
    predictor = TASK_MAP[task](
        checkpoint_path=model_path,
        test_set_path=test_set_path,
        gpu=gpu,
        batch_size=batch_size,
        head_num=head_num,
        layer_num=layer_num,
        code_label=code_label,
        checkpoint_interval=checkpoint_interval
    )
    
    bt = BehavioralTesting(test_dataset_path=test_set_path)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Run tests for each shift
    for shift_key in shift_keys:
        if not shift_key:  # Skip empty strings
            continue
            
        shift = SHIFT_MAP[shift_key]
        results_path = os.path.join(save_dir, f"{shift_key}_shift_{task}.csv")
        stats_path = os.path.join(save_dir, f"{shift_key}_shift_{task}_stats.txt")
        
        # Run test and save results
        stats = bt.run_test(shift, predictor, results_path)
        utils.save_to_file(stats, stats_path)
        
        print(f"Completed {shift_key} shift testing. Results saved to {save_dir}")

if __name__ == '__main__':
    fire.Fire(run)