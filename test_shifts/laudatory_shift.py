"""
Laudatory shift implementation.

This module implements shifts that add positive/laudatory descriptors to clinical texts.
"""

import re
import random
from enum import Enum
from typing import Optional
import logging
import utils
from test_shifts.base_shift import BaseShift

logger = logging.getLogger(__name__)


class LaudatoryLevel(Enum):
    """Levels of laudatory language to test."""
    COMPLIANT = 1
    COOPERATIVE = 2
    PLEASANT = 3
    RESPECTFUL = 4
    NO_MENTION = 5  # Used for identification only, not as a test group


class LaudatoryShift(BaseShift):
    """
    Shift transformation that adds laudatory patient descriptors.

    This shift tests how positive language affects model predictions.
    """

    # Define the laudatory terms associated with each LaudatoryLevel
    laudatory_mentions = {
        LaudatoryLevel.COMPLIANT: [
            "compliant", "adherent", "responsible", "reliable"
        ],
        LaudatoryLevel.COOPERATIVE: [
            "cooperative", "agreeable", "courteous", "considerate"
        ],
        LaudatoryLevel.PLEASANT: [
            "pleasant", "agreeable"
        ],
        LaudatoryLevel.RESPECTFUL: [
            "respectful", "courteous", "considerate"
        ],
    }

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize laudatory shift.

        Args:
            random_seed: Random seed for reproducible word selection
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            logger.info(f"LaudatoryShift initialized with random_seed={random_seed}")

    def get_groups(self):
        """
        Returns list of LaudatoryLevel enum members for testing.

        Note: NO_MENTION is excluded as it's not a test condition.
        """
        return [level for level in LaudatoryLevel if level != LaudatoryLevel.NO_MENTION]

    def get_group_names(self):
        """Returns the names of each laudatory level for display or analysis."""
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: LaudatoryLevel):
        """Applies the text_to_laudatory method for the specified laudatory level."""
        if not isinstance(sample, str) or not sample.strip():
            logger.warning("Invalid sample provided to get_shift_method")
            return None

        try:
            return self.text_to_laudatory(sample, group)
        except Exception as e:
            logger.error(f"Error in get_shift_method for {group}: {e}")
            return None

    def identify_group_in_text(self, text: str) -> Optional[LaudatoryLevel]:
        """
        Identifies if the text contains any laudatory terms.

        Args:
            text: Clinical note text

        Returns:
            LaudatoryLevel: The first matching laudatory level in the text,
            or NO_MENTION if no terms are found.
        """
        if not text:
            return LaudatoryLevel.NO_MENTION

        for laudatory_key in self.laudatory_mentions:
            indicators = self.laudatory_mentions[laudatory_key]
            for mention in indicators:
                pattern = r'\b' + re.escape(mention) + r'\b'
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return laudatory_key

        return LaudatoryLevel.NO_MENTION

    def text_to_laudatory(self, text: str, laudatory: LaudatoryLevel) -> str:
        """
        Modifies the text to contain a laudatory descriptor from the specified level.

        Args:
            text: The input text to modify.
            laudatory: The target laudatory level for insertion.

        Returns:
            str: The modified text with the specified laudatory level.
        """
        shifted_text = text

        # Check if any term from the target level is already in the text
        target_indicators = self.laudatory_mentions.get(laudatory, [])
        for mention in target_indicators:
            pattern = r'\b' + re.escape(mention) + r'\b'
            if re.search(pattern, shifted_text, flags=re.IGNORECASE):
                logger.debug(f"Text already contains '{mention}' from target level")
                return shifted_text

        # Remove mentions from other levels with proper word boundaries
        for laudatory_key, indicators in self.laudatory_mentions.items():
            if laudatory_key != laudatory:
                for mention in indicators:
                    pattern = r'\b' + re.escape(mention) + r'\b'
                    shifted_text = re.sub(pattern, '', shifted_text, flags=re.IGNORECASE)

        # Clean up extra whitespace
        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        # Choose a term for the specified level
        if self.random_seed is not None:
            random.seed(self.random_seed + hash(text) % 10000)

        laudatory_mention = random.choice(target_indicators)
        mention_insertion_position = utils.find_patient_characteristic_position_in_text(shifted_text)

        # Insert the chosen mention
        if mention_insertion_position > 0:
            shifted_text = (
                f"{shifted_text[:mention_insertion_position]}"
                f"{laudatory_mention} "
                f"{shifted_text[mention_insertion_position:]}"
            )
            logger.debug(f"Inserted '{laudatory_mention}' at position {mention_insertion_position}")
        else:
            words = shifted_text.split()
            if len(words) > 1:
                shifted_text = f"{words[0]} {laudatory_mention} {' '.join(words[1:])}"
                logger.debug(f"Inserted '{laudatory_mention}' after first word")
            else:
                shifted_text = f"{laudatory_mention} {shifted_text}"
                logger.warning("No patient characteristic found, inserted at beginning")

        # Final cleanup
        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        return shifted_text
