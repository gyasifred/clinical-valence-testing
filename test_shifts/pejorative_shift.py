"""
Pejorative shift implementation.

This module implements shifts that add negative/pejorative descriptors to clinical texts.
"""

import re
import random
from enum import Enum
from typing import Optional
import logging
import utils
from test_shifts.base_shift import BaseShift

logger = logging.getLogger(__name__)


class PejorativeLevel(Enum):
    """Levels of pejorative language to test."""
    NON_COMPLIANT = 1
    UNCOOPERATIVE = 2
    RESISTANT = 3
    DIFFICULT = 4
    NO_MENTION = 5  # Used for identification only, not as a test group


class PejorativeShift(BaseShift):
    """
    Shift transformation that adds pejorative patient descriptors.

    This shift tests how negative language affects model predictions.
    """

    # Define the pejorative terms associated with each PejorativeLevel
    pejorative_mentions = {
        PejorativeLevel.NON_COMPLIANT: [
            "non-compliant", "negligent", "careless", "irresponsible", "unreliable"
        ],
        PejorativeLevel.UNCOOPERATIVE: [
            "uncooperative", "difficult", "problematic", "demanding", "argumentative", "defiant"
        ],
        PejorativeLevel.RESISTANT: [
            "resistant", "hostile", "aggressive", "disruptive", "troublesome"
        ],
        PejorativeLevel.DIFFICULT: [
            "difficult", "manipulative", "malingering", "attention-seeking", "drug-seeking", "troublesome"
        ],
    }

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize pejorative shift.

        Args:
            random_seed: Random seed for reproducible word selection
        """
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            logger.info(f"PejorativeShift initialized with random_seed={random_seed}")

    def get_groups(self):
        """
        Returns list of PejorativeLevel enum members for testing.

        Note: NO_MENTION is excluded as it's not a test condition.
        """
        # Exclude NO_MENTION from test groups
        return [level for level in PejorativeLevel if level != PejorativeLevel.NO_MENTION]

    def get_group_names(self):
        """Returns the names of each pejorative level for display or analysis."""
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: PejorativeLevel):
        """Applies the text_to_pejorative method for the specified pejorative level."""
        if not isinstance(sample, str) or not sample.strip():
            logger.warning("Invalid sample provided to get_shift_method")
            return None

        try:
            return self.text_to_pejorative(sample, group)
        except Exception as e:
            logger.error(f"Error in get_shift_method for {group}: {e}")
            return None

    def identify_group_in_text(self, text: str) -> Optional[PejorativeLevel]:
        """
        Identifies if the text contains any pejorative terms.

        Args:
            text: Clinical note text

        Returns:
            PejorativeLevel: The first matching pejorative level in the text,
            or NO_MENTION if no terms are found.
        """
        if not text:
            return PejorativeLevel.NO_MENTION

        for pejorative_key in self.pejorative_mentions:
            indicators = self.pejorative_mentions[pejorative_key]
            for mention in indicators:
                # Use word boundaries and escape special characters
                pattern = r'\b' + re.escape(mention) + r'\b'
                if re.search(pattern, text, flags=re.IGNORECASE):
                    return pejorative_key

        return PejorativeLevel.NO_MENTION

    def text_to_pejorative(self, text: str, pejorative: PejorativeLevel) -> str:
        """
        Modifies the text to contain a pejorative descriptor from the specified level.

        This method:
            - Checks if any term from the target pejorative level already exists.
            - Removes terms from any other levels.
            - Inserts a randomly selected term from the target level if no existing term matches.

        Args:
            text: The input text to modify.
            pejorative: The target pejorative level for insertion.

        Returns:
            str: The modified text with the specified pejorative level.
        """
        shifted_text = text

        # Check if any term from the target level is already in the text
        target_indicators = self.pejorative_mentions.get(pejorative, [])
        for mention in target_indicators:
            pattern = r'\b' + re.escape(mention) + r'\b'
            if re.search(pattern, shifted_text, flags=re.IGNORECASE):
                # If the text already has a word from the target level, return it as-is
                logger.debug(f"Text already contains '{mention}' from target level")
                return shifted_text

        # Remove mentions from other levels with proper word boundaries
        for pejorative_key, indicators in self.pejorative_mentions.items():
            if pejorative_key != pejorative:
                for mention in indicators:
                    pattern = r'\b' + re.escape(mention) + r'\b'
                    shifted_text = re.sub(pattern, '', shifted_text, flags=re.IGNORECASE)

        # Clean up extra whitespace after removal
        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        # Choose a term for the specified level
        if self.random_seed is not None:
            # Use deterministic selection based on text for reproducibility
            random.seed(self.random_seed + hash(text) % 10000)

        pejorative_mention = random.choice(target_indicators)
        mention_insertion_position = utils.find_patient_characteristic_position_in_text(shifted_text)

        # Insert the chosen mention
        if mention_insertion_position > 0:
            shifted_text = (
                f"{shifted_text[:mention_insertion_position]}"
                f"{pejorative_mention} "
                f"{shifted_text[mention_insertion_position:]}"
            )
            logger.debug(f"Inserted '{pejorative_mention}' at position {mention_insertion_position}")
        else:
            # If no good position found, insert after first word
            words = shifted_text.split()
            if len(words) > 1:
                shifted_text = f"{words[0]} {pejorative_mention} {' '.join(words[1:])}"
                logger.debug(f"Inserted '{pejorative_mention}' after first word")
            else:
                shifted_text = f"{pejorative_mention} {shifted_text}"
                logger.warning("No patient characteristic found, inserted at beginning")

        # Final cleanup
        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        return shifted_text
