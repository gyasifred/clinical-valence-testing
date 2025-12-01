"""
Base class for shift transformations.

This module provides the base class that all shift implementations inherit from.
"""

from enum import Enum
from typing import List, Union, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BaseShift:
    """
    Base class for text shift transformations.

    All shift implementations should inherit from this class and implement
    the abstract methods.
    """

    def make_shift(
        self,
        samples: List[str],
        return_stats: bool = False
    ) -> Union[List[List[str]], Tuple[List[List[str]], Dict]]:
        """
        Apply shift transformation to all samples for all groups.

        CRITICAL FIX: Each group now independently determines which samples to skip.
        A sample that returns None for one group may still be valid for another group.

        Args:
            samples: List of text samples to transform
            return_stats: Whether to return statistics about the transformation

        Returns:
            If return_stats is False: List of lists (one per group)
            If return_stats is True: Tuple of (shift_groups, stats)
        """
        if not samples:
            logger.warning("Empty samples list provided to make_shift")
            return ([], {"total_samples": 0}) if return_stats else []

        shift_groups = []
        total_skipped_overall = set()  # Track samples that fail in ALL groups
        per_group_stats = []

        groups = self.get_groups()
        if not groups:
            logger.error("No groups defined for shift")
            raise ValueError("Shift must define at least one group")

        for group_idx, group in enumerate(groups):
            samples_in_group = []
            skipped_in_this_group = set()

            for i, sample in enumerate(samples):
                if not isinstance(sample, str):
                    logger.warning(f"Sample {i} is not a string, skipping")
                    skipped_in_this_group.add(i)
                    samples_in_group.append(None)
                    continue

                try:
                    shifted_sample = self.get_shift_method(sample, group)

                    if shifted_sample is None:
                        logger.debug(f"Sample {i} returned None for group {group}")
                        skipped_in_this_group.add(i)

                    samples_in_group.append(shifted_sample)

                except Exception as e:
                    logger.error(f"Error shifting sample {i} for group {group}: {e}")
                    skipped_in_this_group.add(i)
                    samples_in_group.append(None)

            # Filter out None samples for this group ONLY
            filtered_samples = [
                samp for j, samp in enumerate(samples_in_group)
                if j not in skipped_in_this_group
            ]

            shift_groups.append(filtered_samples)

            # Track stats for this group
            per_group_stats.append({
                "group": str(group.name) if isinstance(group, Enum) else str(group),
                "total_samples": len(samples),
                "included_samples": len(filtered_samples),
                "skipped_samples": len(skipped_in_this_group)
            })

            # Track samples that were skipped in this group
            if group_idx == 0:
                total_skipped_overall = skipped_in_this_group.copy()
            else:
                total_skipped_overall &= skipped_in_this_group

            logger.info(
                f"Group '{group}': {len(filtered_samples)}/{len(samples)} samples "
                f"({len(skipped_in_this_group)} skipped)"
            )

        # Statistics
        stats = {
            "total_samples": len(samples),
            "num_groups": len(groups),
            "groups": [str(g.name) if isinstance(g, Enum) else str(g) for g in groups],
            "samples_per_group": [len(group) for group in shift_groups],
            "per_group_details": per_group_stats,
            "skipped_in_all_groups": sorted(list(total_skipped_overall)),
            "num_skipped_in_all": len(total_skipped_overall),
            "total_included_samples": len(samples) - len(total_skipped_overall)
        }

        logger.info(
            f"Shift complete: {len(samples)} samples across {len(groups)} groups. "
            f"{len(total_skipped_overall)} samples skipped in all groups."
        )

        if return_stats:
            return shift_groups, stats
        else:
            return shift_groups

    def get_groups(self) -> List[Union[Enum, str]]:
        """
        Get list of all groups/levels for this shift.

        Returns:
            List of groups (typically Enum members)

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_groups()")

    def get_group_names(self) -> List[str]:
        """
        Get human-readable names for all groups.

        Returns:
            List of group names as strings

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_group_names()")

    def get_shift_method(
        self,
        sample: str,
        group: Union[Enum, str]
    ) -> Optional[str]:
        """
        Apply shift transformation for a specific group.

        Args:
            sample: Text sample to transform
            group: Group/level to apply

        Returns:
            Transformed text, or None if sample should be skipped for this group

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement get_shift_method()")

    def identify_group_in_text(self, text: str) -> Optional[Union[Enum, str]]:
        """
        Identify which group a text sample belongs to (if any).

        Args:
            text: Text to analyze

        Returns:
            Group identifier if found, None otherwise

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement identify_group_in_text()")