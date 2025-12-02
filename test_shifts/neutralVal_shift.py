"""
Neutral valence shift implementation.

This module implements shifts that add neutral descriptors to clinical texts.
"""

import re
import random
from enum import Enum
from typing import Optional
import logging
import utils
from test_shifts.base_shift import BaseShift

logger = logging.getLogger(__name__)


class ValenceType(Enum):
    """Types of neutral language to test."""
    NEUTRAL = 1
    NO_MENTION = 2  # Used for identification only, not as a test group


class NeutralValShift(BaseShift):
    """
    Shift transformation that adds neutral patient descriptors.

    This shift tests how neutral/objective language affects model predictions.
    """

    # Define the neutral terms associated with the NEUTRAL valence level
    neutral_mentions = {
        ValenceType.NEUTRAL: [
            "typical", "average", "regular", "standard", "usual",
            "presenting", "referred", "evaluated", "assessed", "monitored"
        ],
    }

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize neutral valence shift.

        Args:
            random_seed: Random seed for reproducible word selection
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            logger.info(f"NeutralValShift initialized with random_seed={random_seed}")

    def get_groups(self):
        """Returns list of ValenceType enum members for testing (excludes NO_MENTION)."""
        return [level for level in ValenceType if level != ValenceType.NO_MENTION]

    def get_group_names(self):
        """Returns the names of each valence type for display or analysis."""
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: ValenceType):
        """Applies the text_to_neutral method for the specified valence level."""
        if not isinstance(sample, str) or not sample.strip():
            logger.warning("Invalid sample provided to get_shift_method")
            return None

        try:
            return self.text_to_neutral(sample, group)
        except Exception as e:
            logger.error(f"Error in get_shift_method for {group}: {e}")
            return None

    def identify_group_in_text(self, text: str) -> Optional[ValenceType]:
        """Identifies if the text contains any neutral terms.

        Args:
            text: Clinical note text

        Returns:
            ValenceType: The first matching valence level in the text,
            or NO_MENTION if no terms are found.
        """
        if not text:
            return ValenceType.NO_MENTION

        for neutral_key in self.neutral_mentions:
            indicators = self.neutral_mentions[neutral_key]
            for mention in indicators:
                pattern = r'\b' + re.escape(mention) + r'\b'
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return neutral_key

        return ValenceType.NO_MENTION

    def text_to_neutral(self, text: str, neutral: ValenceType) -> str:
        """Modifies the text to contain a neutral descriptor from the specified level.

        Args:
            text: The input text to modify.
            neutral: The target valence level for insertion.

        Returns:
            str: The modified text with the specified neutral level.
        """
        shifted_text = text

        # Check if any term from the target level is already in the text
        target_indicators = self.neutral_mentions.get(neutral, [])
        for mention in target_indicators:
            pattern = r'\b' + re.escape(mention) + r'\b'
            if re.search(pattern, shifted_text, flags=re.IGNORECASE):
                logger.debug(f"Text already contains '{mention}' from target level")
                return shifted_text

        # Remove mentions from other levels with proper word boundaries
        for neutral_key, indicators in self.neutral_mentions.items():
            if neutral_key != neutral:
                for mention in indicators:
                    pattern = r'\b' + re.escape(mention) + r'\b'
                    shifted_text = re.sub(pattern, '', shifted_text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        # Choose a term for the specified level
        if self.random_seed is not None:
            random.seed(self.random_seed + hash(text) % 10000)

        neutral_mention = random.choice(target_indicators)
        mention_insertion_position = utils.find_patient_characteristic_position_in_text(shifted_text)

        # Insert the chosen mention
        if mention_insertion_position > 0:
            shifted_text = (
                f"{shifted_text[:mention_insertion_position]}"
                f"{neutral_mention} "
                f"{shifted_text[mention_insertion_position:]}"
            )
            logger.debug(f"Inserted '{neutral_mention}' at position {mention_insertion_position}")
        else:
            words = shifted_text.split()
            if len(words) > 1:
                shifted_text = f"{words[0]} {neutral_mention} {' '.join(words[1:])}"
                logger.debug(f"Inserted '{neutral_mention}' after first word")
            else:
                shifted_text = f"{neutral_mention} {shifted_text}"
                logger.warning("No patient characteristic found, inserted at beginning")

        # Final cleanup
        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        return shifted_text
