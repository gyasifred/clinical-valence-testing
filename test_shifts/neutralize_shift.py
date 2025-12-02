"""
NeutralizeShift: Removes all valence-laden terms (pejorative, laudatory, and neutral descriptors)
from clinical texts to create completely neutralized versions.

This shift helps evaluate how models respond when all subjective language is removed,
testing whether predictions rely on objective medical facts or subjective descriptions.
"""

import re
import logging
from enum import Enum
from typing import Optional

from test_shifts.base_shift import BaseShift
from test_shifts.laudatory_shift import LaudatoryShift, LaudatoryLevel
from test_shifts.pejorative_shift import PejorativeShift, PejorativeLevel
from test_shifts.neutralVal_shift import NeutralValShift, ValenceType

logger = logging.getLogger(__name__)


class NeutralizeType(Enum):
    """Enum representing neutralized text (all valence terms removed)"""
    NEUTRALIZED = 1


class NeutralizeShift(BaseShift):
    """
    Shift that removes all valence terms from clinical texts.

    This shift combines pejorative, laudatory, and neutral valence shifts
    to create completely neutralized text by stripping all subjective descriptors.
    """

    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize NeutralizeShift.

        Args:
            random_seed: Random seed for reproducibility in child shifts
        """
        self.random_seed = random_seed
        self.laudatory_shift = LaudatoryShift(random_seed=random_seed)
        self.pejorative_shift = PejorativeShift(random_seed=random_seed)
        self.neutral_shift = NeutralValShift(random_seed=random_seed)
        logger.info(f"NeutralizeShift initialized with random_seed={random_seed}")

    def get_groups(self) -> list:
        """
        Get list of neutralize groups.

        Returns:
            List containing NEUTRALIZED enum value
        """
        return list(NeutralizeType)

    def get_group_names(self) -> list:
        """
        Get names of neutralize groups.

        Returns:
            List of group names as strings
        """
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: NeutralizeType) -> str:
        """
        Get shifted version of sample for given group.

        Args:
            sample: Original text
            group: NeutralizeType group (always NEUTRALIZED)

        Returns:
            Text with all valence terms removed
        """
        return self.strip_all_valence(sample)

    def identify_group_in_text(self, text: str) -> Optional[NeutralizeType]:
        """
        Identify if text contains any valence terms.

        Args:
            text: Input clinical text

        Returns:
            NeutralizeType.NEUTRALIZED if any valence terms found, None otherwise
        """
        has_laudatory = self.laudatory_shift.identify_group_in_text(text) != LaudatoryLevel.NO_MENTION
        has_pejorative = self.pejorative_shift.identify_group_in_text(text) != PejorativeLevel.NO_MENTION
        has_neutral = self.neutral_shift.identify_group_in_text(text) != ValenceType.NO_MENTION

        if not (has_laudatory or has_pejorative or has_neutral):
            logger.debug("No valence terms found in text")
            return None

        logger.debug(f"Valence terms found - laudatory: {has_laudatory}, "
                    f"pejorative: {has_pejorative}, neutral: {has_neutral}")
        return NeutralizeType.NEUTRALIZED

    def strip_all_valence(self, text: str) -> str:
        """
        Remove all valence terms (pejorative, laudatory, neutral) from text.

        This method systematically removes all subjective descriptors identified
        by the three valence shifts, creating a maximally neutralized version.

        Args:
            text: Original clinical text

        Returns:
            Text with all valence terms removed and whitespace normalized

        Example:
            >>> shift = NeutralizeShift()
            >>> shift.strip_all_valence("The difficult but cooperative patient")
            "The patient"
        """
        shifted_text = text
        terms_removed = 0

        # Remove pejorative terms
        for level in PejorativeLevel:
            if level != PejorativeLevel.NO_MENTION:
                for term in self.pejorative_shift.pejorative_mentions.get(level, []):
                    # Use re.escape to handle special characters like hyphens
                    pattern = r'\b' + re.escape(term) + r'\b'
                    before = shifted_text
                    shifted_text = re.sub(pattern, '', shifted_text, flags=re.IGNORECASE)
                    if before != shifted_text:
                        terms_removed += 1
                        logger.debug(f"Removed pejorative term: '{term}'")

        # Remove laudatory terms
        for level in LaudatoryLevel:
            if level != LaudatoryLevel.NO_MENTION:
                for term in self.laudatory_shift.laudatory_mentions.get(level, []):
                    pattern = r'\b' + re.escape(term) + r'\b'
                    before = shifted_text
                    shifted_text = re.sub(pattern, '', shifted_text, flags=re.IGNORECASE)
                    if before != shifted_text:
                        terms_removed += 1
                        logger.debug(f"Removed laudatory term: '{term}'")

        # Remove neutral valence terms
        for val_type in ValenceType:
            if val_type != ValenceType.NO_MENTION:
                for term in self.neutral_shift.neutral_mentions.get(val_type, []):
                    pattern = r'\b' + re.escape(term) + r'\b'
                    before = shifted_text
                    shifted_text = re.sub(pattern, '', shifted_text, flags=re.IGNORECASE)
                    if before != shifted_text:
                        terms_removed += 1
                        logger.debug(f"Removed neutral term: '{term}'")

        # Clean up extra whitespace
        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        logger.info(f"Neutralization complete: removed {terms_removed} valence terms")
        return shifted_text