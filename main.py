#!/usr/bin/env python3
import os
from typing import Union, Optional
import fire
import utils
from valence_testing import BehavioralTesting
from prediction import DiagnosisPredictor
from test_shifts.laudatory_shift import LaudatoryShift
from test_shifts.neutralize_shift import NeutralizeShift
from test_shifts.neutralVal_shift import NeutralValShift
from test_shifts.pejorative_shift import PejorativeShift
from config_loader import get_config
from logger import setup_logging, get_logger, log_experiment_info
from statistical_analysis import StatisticalAnalyzer

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def get_shift_map(random_seed: Optional[int] = None):
    """Create shift map with random seed for reproducibility"""
    return {
        "neutralize": NeutralizeShift(random_seed=random_seed),
        "pejorative": PejorativeShift(random_seed=random_seed),
        "laud": LaudatoryShift(random_seed=random_seed),
        "neutralval": NeutralValShift(random_seed=random_seed)
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
    test_set_path: Optional[str] = None,
    model_path: Optional[str] = None,
    shift_keys: Optional[Union[str, tuple]] = None,
    task: str = "diagnosis",
    save_dir: Optional[Union[str, tuple]] = None,
    gpu: Optional[bool] = None,
    batch_size: Optional[int] = None,
    head_num: Optional[int] = None,
    layer_num: Optional[int] = None,
    code_label: str = "short_codes",
    checkpoint_interval: int = 1000,
    config_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    run_statistical_analysis: bool = True
):
    """
    Run behavioral tests with specified shifts and predictor

    Args:
        test_set_path: Path to test dataset (uses config if not provided)
        model_path: Path to model checkpoint (uses config if not provided)
        shift_keys: Comma-separated string or list of shift names (all if not provided)
        task: Name of prediction task
        save_dir: Directory to save results (uses config if not provided)
        gpu: Whether to use GPU (uses config if not provided)
        batch_size: Batch size for predictions (uses config if not provided)
        head_num: Attention head number to use (uses config if not provided)
        layer_num: Model layer number to use (uses config if not provided)
        code_label: Column name for codes in dataset
        checkpoint_interval: Interval for saving checkpoints
        config_path: Path to config file (uses default if not provided)
        random_seed: Random seed for reproducibility (uses config if not provided)
        run_statistical_analysis: Whether to run statistical analysis on results
    """
    try:
        # Load configuration
        config = get_config(config_path)
        logger.info("Configuration loaded successfully")

        # Use config values as defaults
        test_set_path = parse_argument(test_set_path) if test_set_path else config.data.test_set_path
        model_path = parse_argument(model_path) if model_path else config.model.name
        save_dir = parse_argument(save_dir) if save_dir else config.output.results_dir
        task = parse_argument(task)

        # Use config for model parameters
        gpu = gpu if gpu is not None else config.model.use_gpu
        batch_size = batch_size if batch_size is not None else config.model.batch_size
        head_num = head_num if head_num is not None else config.model.attention['head_num']
        layer_num = layer_num if layer_num is not None else config.model.attention['layer_num']
        random_seed = random_seed if random_seed is not None else config.random_seed

        # Set random seeds for reproducibility
        if random_seed is not None:
            utils.set_random_seeds(random_seed)
            logger.info(f"Random seed set to {random_seed}")

        # Configure deterministic mode if requested
        if config.deterministic:
            utils.configure_deterministic_mode()
            logger.info("Deterministic mode enabled")

        # Parse shift keys
        if shift_keys:
            shift_keys = parse_shift_keys(shift_keys)
        else:
            shift_keys = ["neutralize", "pejorative", "laud", "neutralval"]

        # Log experiment info
        log_experiment_info(logger, {
            "test_set_path": test_set_path,
            "model_path": model_path,
            "shifts": shift_keys,
            "task": task,
            "save_dir": save_dir,
            "batch_size": batch_size,
            "random_seed": random_seed,
            "gpu": gpu
        })

        logger.info(f"Running with shifts: {shift_keys}")

        # Initialize shift map with random seed
        shift_map = get_shift_map(random_seed)

        # Validate shift keys
        invalid_shifts = [s for s in shift_keys if s not in shift_map]
        if invalid_shifts:
            logger.error(f"Invalid shift keys: {invalid_shifts}")
            logger.error(f"Available shifts: {list(shift_map.keys())}")
            raise ValueError(f"Invalid shift keys: {invalid_shifts}")

        # Initialize predictor and testing framework
        logger.info(f"Initializing {task} predictor...")
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

        logger.info("Initializing behavioral testing framework...")
        bt = BehavioralTesting(test_dataset_path=test_set_path)

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Results will be saved to: {save_dir}")

        # Initialize statistical analyzer if requested
        analyzer = StatisticalAnalyzer() if run_statistical_analysis else None

        # Run tests for each shift
        all_results = {}
        for shift_key in shift_keys:
            if not shift_key:  # Skip empty strings
                continue

            logger.info(f"Starting {shift_key} shift testing...")
            shift = shift_map[shift_key]
            results_path = os.path.join(save_dir, f"{shift_key}_shift_{task}.csv")
            stats_path = os.path.join(save_dir, f"{shift_key}_shift_{task}_stats.txt")

            # Run test and save results
            stats = bt.run_test(shift, predictor, results_path)
            utils.save_to_file(stats, stats_path)
            all_results[shift_key] = stats

            logger.info(f"Completed {shift_key} shift testing")
            logger.info(f"Results saved to {results_path}")
            logger.info(f"Statistics saved to {stats_path}")

        # Run statistical analysis if requested
        if run_statistical_analysis and analyzer:
            logger.info("Running statistical analysis...")
            analysis_path = os.path.join(save_dir, "statistical_analysis.txt")

            try:
                # Generate comprehensive statistical report
                report = analyzer.generate_analysis_report(all_results)
                utils.save_to_file(report, analysis_path)
                logger.info(f"Statistical analysis saved to {analysis_path}")
            except Exception as e:
                logger.error(f"Error during statistical analysis: {e}")

        logger.info("All behavioral tests completed successfully!")
        return all_results

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    fire.Fire(run)