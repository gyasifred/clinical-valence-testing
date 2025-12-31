#!/usr/bin/env python3
import os
import glob
from typing import Union, Optional
import fire
import pandas as pd
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
            logger.info("Running comprehensive statistical analysis...")

            try:
                # Load results from saved CSV files
                results_data = {}
                diagnosis_file_pattern = "*_diagnosis.csv"

                for shift_key in shift_keys:
                    if not shift_key:
                        continue

                    # Look for diagnosis CSV files for this shift
                    pattern = os.path.join(save_dir, f"*{shift_key.split('_')[0]}*diagnosis.csv")
                    matching_files = glob.glob(pattern)

                    if matching_files:
                        csv_file = matching_files[0]  # Use first match
                        logger.info(f"Loading results for {shift_key} from {csv_file}")
                        results_data[shift_key] = pd.read_csv(csv_file)
                    else:
                        logger.warning(f"No diagnosis CSV found for {shift_key}")

                # Perform comparative statistical analysis
                if len(results_data) >= 2:
                    # Use neutralize as baseline if available, otherwise use first shift
                    baseline_key = 'neutralize' if 'neutralize' in results_data else list(results_data.keys())[0]
                    baseline_data = results_data[baseline_key]

                    logger.info(f"Using {baseline_key} as baseline for comparisons")

                    # Extract diagnosis codes from the data
                    # Look for columns that represent diagnosis probabilities
                    diagnosis_cols = [col for col in baseline_data.columns
                                     if col not in ['note_id', 'text', 'shifted_text',
                                                   'sample_id', 'group', 'attention_weights']]

                    if not diagnosis_cols:
                        logger.warning("No diagnosis probability columns found in CSV")
                        logger.info("Skipping statistical analysis")
                    else:
                        logger.info(f"Found {len(diagnosis_cols)} diagnosis codes for analysis")

                        # Prepare baseline diagnosis probabilities
                        baseline_probs = baseline_data[diagnosis_cols]

                        # Compare each treatment shift against baseline
                        all_comparison_results = {}

                        for shift_key, shift_data in results_data.items():
                            if shift_key == baseline_key:
                                continue

                            logger.info(f"Analyzing {shift_key} vs {baseline_key}...")

                            # Extract treatment diagnosis probabilities
                            treatment_probs = shift_data[diagnosis_cols]

                            # Run comprehensive statistical analysis
                            comparison_results = analyzer.analyze_diagnosis_shifts(
                                baseline_probs=baseline_probs,
                                treatment_probs=treatment_probs,
                                diagnosis_codes=diagnosis_cols,
                                use_permutation=True,
                                n_permutations=10000,
                                random_seed=config.random_seed if hasattr(config, 'random_seed') else 42
                            )

                            all_comparison_results[shift_key] = comparison_results

                            # Save detailed results for this comparison
                            comparison_csv_path = os.path.join(
                                save_dir,
                                f"statistical_analysis_{baseline_key}_vs_{shift_key}.csv"
                            )
                            comparison_results.to_csv(comparison_csv_path, index=False)
                            logger.info(f"Saved detailed comparison to {comparison_csv_path}")

                        # Generate comprehensive analysis report
                        report_lines = []
                        report_lines.append("=" * 80)
                        report_lines.append("COMPREHENSIVE STATISTICAL ANALYSIS")
                        report_lines.append("Clinical Valence Testing - Diagnosis Prediction Shifts")
                        report_lines.append("=" * 80)
                        report_lines.append("")
                        report_lines.append(f"Baseline: {baseline_key}")
                        report_lines.append(f"Comparisons: {', '.join([k for k in results_data.keys() if k != baseline_key])}")
                        report_lines.append(f"Number of samples: {len(baseline_data)}")
                        report_lines.append(f"Number of diagnosis codes analyzed: {len(diagnosis_cols)}")
                        report_lines.append(f"Significance level: {analyzer.significance_level}")
                        report_lines.append(f"Multiple comparison correction: {analyzer.correction_method}")
                        report_lines.append("")

                        # Generate report for each comparison
                        for shift_key, comparison_results in all_comparison_results.items():
                            report = analyzer.generate_analysis_report(
                                diagnosis_results=comparison_results,
                                attention_results=None
                            )

                            report_lines.append("")
                            report_lines.append("=" * 80)
                            report_lines.append(f"COMPARISON: {baseline_key} vs {shift_key}")
                            report_lines.append("=" * 80)
                            report_lines.append(report)

                        # Save comprehensive report
                        report_path = os.path.join(save_dir, "statistical_analysis.txt")
                        with open(report_path, 'w') as f:
                            f.write('\n'.join(report_lines))

                        logger.info(f"Comprehensive statistical analysis report saved to {report_path}")
                        logger.info("Statistical analysis completed successfully!")

                else:
                    logger.warning(f"Need at least 2 shifts for statistical comparison, found {len(results_data)}")
                    logger.info("Skipping statistical analysis")

            except Exception as e:
                logger.error(f"Error during statistical analysis: {e}", exc_info=True)
                logger.warning("Statistical analysis failed, but behavioral testing results are still saved")


        logger.info("All behavioral tests completed successfully!")
        return all_results

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    fire.Fire(run)