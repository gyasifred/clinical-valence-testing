"""
Configuration management for Clinical Valence Testing.

This module provides functionality to load, validate, and access
configuration parameters from YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    name: str = "bvanaken/CORe-clinical-outcome-biobert-v1"
    max_length: int = 512
    batch_size: int = 128
    device: str = "auto"
    attention: Dict[str, Any] = field(default_factory=lambda: {
        "layer_num": 11,
        "head_num": 11,
        "aggregation": "sum"
    })


@dataclass
class DataConfig:
    """Data configuration parameters."""
    code_label: str = "short_codes"
    text_label: str = "text"
    min_code_frequency: int = 100


@dataclass
class PredictionConfig:
    """Prediction configuration parameters."""
    threshold: float = 0.5
    checkpoint_interval: int = 1000
    save_attention: bool = True
    save_clinical_notes: bool = True


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    num_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: float = 0.1
    scheduler: str = "linear"
    early_stopping_patience: int = 3


@dataclass
class AnalysisConfig:
    """Analysis configuration parameters."""
    baseline: str = "neutralize"
    significance_level: float = 0.05
    multiple_testing_correction: str = "fdr_bh"
    effect_size_threshold: float = 0.01
    visualization: Dict[str, Any] = field(default_factory=dict)
    statistical_tests: list = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "clinical_valence_testing.log"
    console: bool = True


@dataclass
class OutputConfig:
    """Output configuration parameters."""
    save_dir: str = "./results"
    create_timestamp_dirs: bool = True
    save_format: str = "csv"
    compression: Optional[str] = None


class Config:
    """
    Central configuration manager for Clinical Valence Testing.

    This class loads configuration from a YAML file and provides
    type-safe access to configuration parameters.

    Attributes:
        model: Model configuration
        data: Data configuration
        prediction: Prediction configuration
        training: Training configuration
        analysis: Analysis configuration
        logging_config: Logging configuration
        output: Output configuration
        random_seed: Random seed for reproducibility
        deterministic: Whether to use deterministic algorithms
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML configuration file.
                        If None, looks for config.yaml in current directory.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            self._config = {}
        else:
            self._config = self._load_config(config_path)

        # Initialize configuration sections
        self.model = self._init_model_config()
        self.data = self._init_data_config()
        self.prediction = self._init_prediction_config()
        self.training = self._init_training_config()
        self.analysis = self._init_analysis_config()
        self.logging_config = self._init_logging_config()
        self.output = self._init_output_config()

        # Global settings
        self.random_seed = self._config.get('random_seed', 42)
        self.deterministic = self._config.get('deterministic', True)
        self.shifts = self._config.get('shifts', {})
        self.icd_translations = self._config.get('icd_translations', {})

    @staticmethod
    def _load_config(config_path: Path) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config or {}
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return {}

    def _init_model_config(self) -> ModelConfig:
        """Initialize model configuration."""
        model_cfg = self._config.get('model', {})
        return ModelConfig(
            name=model_cfg.get('name', ModelConfig.name),
            max_length=model_cfg.get('max_length', ModelConfig.max_length),
            batch_size=model_cfg.get('batch_size', ModelConfig.batch_size),
            device=model_cfg.get('device', ModelConfig.device),
            attention=model_cfg.get('attention', ModelConfig.attention)
        )

    def _init_data_config(self) -> DataConfig:
        """Initialize data configuration."""
        data_cfg = self._config.get('data', {})
        return DataConfig(
            code_label=data_cfg.get('code_label', DataConfig.code_label),
            text_label=data_cfg.get('text_label', DataConfig.text_label),
            min_code_frequency=data_cfg.get('min_code_frequency', DataConfig.min_code_frequency)
        )

    def _init_prediction_config(self) -> PredictionConfig:
        """Initialize prediction configuration."""
        pred_cfg = self._config.get('prediction', {})
        return PredictionConfig(
            threshold=pred_cfg.get('threshold', PredictionConfig.threshold),
            checkpoint_interval=pred_cfg.get('checkpoint_interval', PredictionConfig.checkpoint_interval),
            save_attention=pred_cfg.get('save_attention', PredictionConfig.save_attention),
            save_clinical_notes=pred_cfg.get('save_clinical_notes', PredictionConfig.save_clinical_notes)
        )

    def _init_training_config(self) -> TrainingConfig:
        """Initialize training configuration."""
        train_cfg = self._config.get('training', {})
        return TrainingConfig(
            num_epochs=train_cfg.get('num_epochs', TrainingConfig.num_epochs),
            learning_rate=train_cfg.get('learning_rate', TrainingConfig.learning_rate),
            weight_decay=train_cfg.get('weight_decay', TrainingConfig.weight_decay),
            warmup_steps=train_cfg.get('warmup_steps', TrainingConfig.warmup_steps),
            scheduler=train_cfg.get('scheduler', TrainingConfig.scheduler),
            early_stopping_patience=train_cfg.get('early_stopping_patience', TrainingConfig.early_stopping_patience)
        )

    def _init_analysis_config(self) -> AnalysisConfig:
        """Initialize analysis configuration."""
        analysis_cfg = self._config.get('analysis', {})
        return AnalysisConfig(
            baseline=analysis_cfg.get('baseline', AnalysisConfig.baseline),
            significance_level=analysis_cfg.get('significance_level', AnalysisConfig.significance_level),
            multiple_testing_correction=analysis_cfg.get('multiple_testing_correction', AnalysisConfig.multiple_testing_correction),
            effect_size_threshold=analysis_cfg.get('effect_size_threshold', AnalysisConfig.effect_size_threshold),
            visualization=analysis_cfg.get('visualization', {}),
            statistical_tests=analysis_cfg.get('statistical_tests', [])
        )

    def _init_logging_config(self) -> LoggingConfig:
        """Initialize logging configuration."""
        log_cfg = self._config.get('logging', {})
        return LoggingConfig(
            level=log_cfg.get('level', LoggingConfig.level),
            format=log_cfg.get('format', LoggingConfig.format),
            file=log_cfg.get('file', LoggingConfig.file),
            console=log_cfg.get('console', LoggingConfig.console)
        )

    def _init_output_config(self) -> OutputConfig:
        """Initialize output configuration."""
        output_cfg = self._config.get('output', {})
        return OutputConfig(
            save_dir=output_cfg.get('save_dir', OutputConfig.save_dir),
            create_timestamp_dirs=output_cfg.get('create_timestamp_dirs', OutputConfig.create_timestamp_dirs),
            save_format=output_cfg.get('save_format', OutputConfig.save_format),
            compression=output_cfg.get('compression', OutputConfig.compression)
        )

    def get_shift_terms(self, shift_type: str, level: Optional[str] = None) -> Union[Dict, list]:
        """
        Get shift terms for a specific shift type and level.

        Args:
            shift_type: Type of shift (pejorative, laudatory, neutral)
            level: Specific level within shift type (optional)

        Returns:
            Dictionary of levels and terms, or list of terms for specific level
        """
        shift_config = self.shifts.get(shift_type, {})
        if level is None:
            return shift_config.get('levels', {})
        else:
            return shift_config.get('levels', {}).get(level, [])

    def save(self, path: Union[str, Path]):
        """
        Save current configuration to YAML file.

        Args:
            path: Path to save configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {path}")

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  model={self.model},\n"
            f"  data={self.data},\n"
            f"  random_seed={self.random_seed}\n"
            f")"
        )


# Global configuration instance
_config_instance: Optional[Config] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """
    Get global configuration instance (singleton pattern).

    Args:
        config_path: Path to configuration file (only used on first call)

    Returns:
        Configuration instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance


def reset_config():
    """Reset global configuration instance."""
    global _config_instance
    _config_instance = None
