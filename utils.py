"""
Utility functions for Clinical Valence Testing.

This module provides utility functions for data processing, pattern matching,
and file operations.
"""

import re
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, List, Optional, Tuple
from functools import lru_cache
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Processing Functions
# ============================================================================

def codes_that_occur_n_times_in_dataset(
    n: int,
    dataset_path: Union[str, Path],
    code_label: str = "short_codes"
) -> List[str]:
    """
    Extract codes that appear at least n times in the dataset.

    Args:
        n: Minimum number of occurrences
        dataset_path: Path to CSV dataset
        code_label: Column name containing codes

    Returns:
        List of codes sorted by frequency (most frequent first)

    Raises:
        FileNotFoundError: If dataset_path doesn't exist
        KeyError: If code_label column not found in dataset
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        logger.error(f"Error reading dataset from {dataset_path}: {e}")
        raise

    if code_label not in df.columns:
        raise KeyError(f"Column '{code_label}' not found in dataset")

    code_count = {}

    for i, row in df.iterrows():
        if pd.isna(row[code_label]):
            continue

        codes = str(row[code_label]).split(",")
        for code in codes:
            code = code.strip()
            if code:  # Only count non-empty codes
                code_count[code] = code_count.get(code, 0) + 1

    # Sort by occurrence (most frequent first)
    codes_sorted_by_occurrence = sorted(
        code_count.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Filter codes with at least n occurrences
    frequent_codes = [code for code, count in codes_sorted_by_occurrence if count >= n]

    logger.info(
        f"Found {len(frequent_codes)} codes with >= {n} occurrences "
        f"(out of {len(code_count)} total unique codes)"
    )

    return frequent_codes


def get_code_statistics(
    dataset_path: Union[str, Path],
    code_label: str = "short_codes"
) -> pd.DataFrame:
    """
    Get comprehensive statistics about code frequencies.

    Args:
        dataset_path: Path to CSV dataset
        code_label: Column name containing codes

    Returns:
        DataFrame with columns: code, count, percentage, cumulative_percentage
    """
    dataset_path = Path(dataset_path)
    df = pd.read_csv(dataset_path)

    code_count = {}
    total_samples = 0

    for i, row in df.iterrows():
        if pd.isna(row[code_label]):
            continue

        total_samples += 1
        codes = str(row[code_label]).split(",")
        for code in codes:
            code = code.strip()
            if code:
                code_count[code] = code_count.get(code, 0) + 1

    # Create statistics DataFrame
    stats_data = [
        {
            'code': code,
            'count': count,
            'percentage': (count / total_samples) * 100
        }
        for code, count in code_count.items()
    ]

    stats_df = pd.DataFrame(stats_data).sort_values('count', ascending=False)
    stats_df['cumulative_percentage'] = stats_df['percentage'].cumsum()
    stats_df = stats_df.reset_index(drop=True)

    return stats_df


# ============================================================================
# Pattern Matching Functions
# ============================================================================

@lru_cache(maxsize=2048)
def find_patient_characteristic_position_in_text(text: str) -> int:
    """
    Find the position in text to insert patient characteristic descriptors.

    This function searches for patient identifiers (age, gender, patient type,
    ethnicity, medical conditions) and returns the position after the first
    match where valence words should be inserted.

    Args:
        text: Clinical note text

    Returns:
        Position index for inserting valence words (0 if no pattern found)

    Examples:
        >>> text = "45 yo male patient with chest pain"
        >>> pos = find_patient_characteristic_position_in_text(text)
        >>> text[:pos] + "compliant " + text[pos:]
        "45 yo compliant male patient with chest pain"
    """
    # Age patterns (various formats)
    age_sub_pattern = r'(\d{2}|\[\*\*Age over 90 \*\*\])'
    age_patterns = [
        rf' ({age_sub_pattern}[ ]?M[,]?) ',
        rf' ({age_sub_pattern}[ ]?F[,]?) ',
        rf'({age_sub_pattern}[ ]?y\/o)',
        rf'({age_sub_pattern}[ ]?yo)',
        rf'({age_sub_pattern}-yo)',
        rf' ({age_sub_pattern}y) ',
        rf'({age_sub_pattern}[ ]?y\.o[\.]?)',
        rf' ({age_sub_pattern}[ ]?yF) ',
        rf' ({age_sub_pattern}[ ]?yM) ',
        rf'({age_sub_pattern} year old)',
        rf'({age_sub_pattern}-year-old)',
        rf'({age_sub_pattern}-year old)',
        rf'({age_sub_pattern} year-old)',
    ]

    # Gender patterns
    person_patterns = [
        r"(female)", r"(woman)", r"(lady)",
        r" (male) ", r" (male)\.", r" (male),",
        r" (man) ", r" (man)\.", r" (man),",
        r"(gentleman)"
    ]

    # Patient type patterns
    patient_xtics_patterns = [
        r"(patient)",
        r"(patients)",
        r"(inpatient)",
        r"(outpatient)",
        r"(bedridden)",
        r"(wheelchair-bound|chair-bound|bed-bound)",
        r"(walker-dependent|cane-dependent|assistance-dependent)",
        r"(post-operative|pre-operative|recovering|terminal)",
        r"(?:palliative|hospice|critically ill|stable|unstable)",
        r"(follow-up|new|established|consulting|referring)"
    ]

    # Ethnicity patterns
    ethnicity_pattern = [
        r"\b(?:African[ -]American|Hispanic|Latino|Latina|Asian|Caucasian|White|Black)\b",
        r"\b(?:Indigenous|Native[ -]American|Pacific Islander|Alaska Native)\b",
        r"\b(?:South Asian|East Asian|Southeast Asian|Middle Eastern|Arab)\b",
        r"\b(?:Caribbean|African|European|Mediterranean|Slavic)\b"
    ]

    # Medical condition patterns
    condition_pattern = [
        r"\b(?:diabetic|hypertensive|asthmatic|arthritic|epileptic)\b",
        r"\b(?:obese|overweight|underweight|cachectic|malnourished)\b",
        r"\b(?:pregnant|postpartum|gravida|para|nulliparous)\b",
        r"\b(?:immunocompromised|immunosuppressed|neutropenic)\b",
        r"\b(?:depressed|anxious|bipolar|schizophrenic|psychotic)\b",
        r"\b(?:demented|cognitively impaired|delirious)\b",
        r"\b(?:disabled|impaired|dependent|independent|ambulatory)\b"
    ]

    # Compile all patterns
    compiled_patterns = {
        'age': [re.compile(pattern, flags=re.IGNORECASE) for pattern in age_patterns],
        'person': [re.compile(pattern, flags=re.IGNORECASE) for pattern in person_patterns],
        'patient_xtics': [re.compile(pattern, flags=re.IGNORECASE) for pattern in patient_xtics_patterns],
        'ethnicity': [re.compile(pattern, flags=re.IGNORECASE) for pattern in ethnicity_pattern],
        'condition': [re.compile(pattern, flags=re.IGNORECASE) for pattern in condition_pattern]
    }

    # Search for patterns in order of priority
    for category, patterns in compiled_patterns.items():
        for pattern in patterns:
            result = re.search(pattern, text)
            if result is not None:
                try:
                    pattern_pos = result.regs[1]
                    if category == 'age':
                        # Return position AFTER age mention
                        return pattern_pos[1] + 1
                    else:
                        # Return position AT the start of match
                        return pattern_pos[0]
                except IndexError:
                    # If no group captured, use match start
                    return result.start()

    # Return 0 if no pattern found (will insert at beginning)
    return 0


def extract_patient_characteristics(text: str) -> Dict[str, Optional[str]]:
    """
    Extract patient characteristics from clinical text.

    Args:
        text: Clinical note text

    Returns:
        Dictionary with keys: age, gender, ethnicity, conditions
    """
    characteristics = {
        'age': None,
        'gender': None,
        'ethnicity': None,
        'conditions': []
    }

    # Extract age
    age_pattern = r'(\d{2})\s*(?:y/?o|year|yr)'
    age_match = re.search(age_pattern, text, re.IGNORECASE)
    if age_match:
        characteristics['age'] = age_match.group(1)

    # Extract gender
    if re.search(r'\b(?:male|man|gentleman)\b', text, re.IGNORECASE):
        if not re.search(r'\bfemale\b', text, re.IGNORECASE):
            characteristics['gender'] = 'male'
    if re.search(r'\b(?:female|woman|lady)\b', text, re.IGNORECASE):
        characteristics['gender'] = 'female'

    # Extract ethnicity
    ethnicity_patterns = [
        r'African[ -]American', r'Hispanic', r'Latino', r'Latina',
        r'Asian', r'Caucasian', r'White', r'Black'
    ]
    for pattern in ethnicity_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            characteristics['ethnicity'] = re.search(pattern, text, re.IGNORECASE).group()
            break

    # Extract conditions
    condition_keywords = [
        'diabetic', 'hypertensive', 'asthmatic', 'obese',
        'pregnant', 'immunocompromised'
    ]
    for keyword in condition_keywords:
        if re.search(rf'\b{keyword}\b', text, re.IGNORECASE):
            characteristics['conditions'].append(keyword)

    return characteristics


# ============================================================================
# Tokenization and Data Preparation
# ============================================================================

def collate_batch(data: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Collate batch of samples for DataLoader.

    Args:
        data: List of dictionaries containing tokenized samples

    Returns:
        Dictionary with padded tensors and lists
    """
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x['input_ids']) for x in data],
        batch_first=True
    )

    attention_masks = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x['attention_mask']) for x in data],
        batch_first=True
    )

    token_type_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x['token_type_ids']) for x in data],
        batch_first=True
    )

    return {
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "token_type_ids": token_type_ids,
        "tokens": [x['tokens'] for x in data],
        "targets": [x['target'] for x in data]
    }


def sample_to_features(
    sample: Union[Dict, pd.Series],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 512,
    text_column: str = "text",
    label_column: str = "label"
) -> Dict[str, Union[List, str]]:
    """
    Convert a sample to tokenized features.

    Args:
        sample: Sample as dictionary or pandas Series
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        text_column: Name of text column
        label_column: Name of label column

    Returns:
        Dictionary with tokenized features
    """
    tokenized = tokenizer.encode_plus(
        sample[text_column],
        truncation=True,
        padding=True,
        max_length=max_length
    )

    featurized_sample = {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "token_type_ids": tokenized["token_type_ids"],
        "tokens": tokenizer.convert_ids_to_tokens(tokenized["input_ids"]),
        "target": sample[label_column]
    }

    return featurized_sample


# ============================================================================
# File I/O Utilities
# ============================================================================

def save_to_file(
    content: Union[str, list, dict],
    file_path: Union[str, Path],
    mode: str = 'w'
) -> None:
    """
    Save content to file.

    Args:
        content: Content to save
        file_path: Path to output file
        mode: File open mode ('w' for write, 'a' for append)

    Raises:
        IOError: If file cannot be written
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, mode) as write_file:
            write_file.write(str(content))
        logger.info(f"Saved content to {file_path}")
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")
        raise IOError(f"Failed to save file: {e}")


def load_dataset(
    dataset_path: Union[str, Path],
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load dataset with validation.

    Args:
        dataset_path: Path to CSV file
        required_columns: List of required column names

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If dataset doesn't exist
        ValueError: If required columns are missing
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Loaded dataset from {dataset_path}: {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    return df


# ============================================================================
# Validation and Quality Checks
# ============================================================================

def validate_clinical_text(text: str) -> Tuple[bool, List[str]]:
    """
    Validate clinical text for common issues.

    Args:
        text: Clinical note text

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    if not text or not text.strip():
        issues.append("Empty text")
        return False, issues

    if len(text) < 10:
        issues.append("Text too short (< 10 characters)")

    if len(text) > 10000:
        issues.append("Text very long (> 10000 characters)")

    # Check for minimum medical content
    medical_keywords = ['patient', 'diagnosis', 'treatment', 'symptoms', 'history']
    has_medical_content = any(kw in text.lower() for kw in medical_keywords)

    if not has_medical_content:
        issues.append("No obvious medical keywords found")

    is_valid = len(issues) == 0
    return is_valid, issues


def check_data_quality(df: pd.DataFrame, text_column: str = "text") -> Dict:
    """
    Check data quality of dataset.

    Args:
        df: DataFrame to check
        text_column: Name of text column

    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {
        'total_samples': len(df),
        'missing_text': df[text_column].isna().sum(),
        'empty_text': (df[text_column].str.strip() == '').sum(),
        'avg_text_length': df[text_column].str.len().mean(),
        'min_text_length': df[text_column].str.len().min(),
        'max_text_length': df[text_column].str.len().max()
    }

    # Check for duplicates
    quality_metrics['duplicate_texts'] = df[text_column].duplicated().sum()

    logger.info(f"Data quality check: {quality_metrics}")

    return quality_metrics


# ============================================================================
# Reproducibility Utilities
# ============================================================================

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Set random seeds to {seed}")


def configure_deterministic_mode(deterministic: bool = True) -> None:
    """
    Configure PyTorch deterministic mode.

    Args:
        deterministic: Whether to enable deterministic algorithms
    """
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Enabled deterministic mode")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info("Disabled deterministic mode (faster but not reproducible)")
