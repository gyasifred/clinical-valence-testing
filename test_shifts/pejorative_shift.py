import re
import random
from enum import Enum
import utils
from test_shifts.base_shift import BaseShift


class PejorativeLevel(Enum):
    NON_COMPLIANT = 1
    UNCOOPERATIVE = 2
    RESISTANT = 3
    DIFFICULT = 4
    NO_MENTION = 5


class PejorativeShift(BaseShift):
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

    def get_groups(self):
        """Returns a list of all PejorativeLevel enum members."""
        return list(PejorativeLevel)

    def get_group_names(self):
        """Returns the names of each pejorative level for display or analysis."""
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: PejorativeLevel):
        """Applies the text_to_pejorative method for the specified pejorative level."""
        return self.text_to_pejorative(sample, group)

    def identify_group_in_text(self, text: str):
        """Identifies if the text contains any terms from pejorative_mentions.
        
        Returns:
            PejorativeLevel: The first matching pejorative level in the text,
            or NO_MENTION if no terms are found.
        """
        for pejorative_key in self.pejorative_mentions:
            indicators = self.pejorative_mentions[pejorative_key]
            for mention in indicators:
                if re.search(mention, text, flags=re.IGNORECASE):
                    return pejorative_key
        return PejorativeLevel.NO_MENTION

    def text_to_pejorative(self, text: str, pejorative: PejorativeLevel):
        """Modifies the text to contain a pejorative descriptor from the specified level.
        
        This method:
            - Checks if any term from the target pejorative level already exists.
            - Removes terms from any other levels.
            - If the target level is NO_MENTION, returns text without pejorative mentions.
            - Inserts a randomly selected term from the target level if no existing term matches.

        Args:
            text (str): The input text to modify.
            pejorative (PejorativeLevel): The target pejorative level for insertion.

        Returns:
            str: The modified text with the specified pejorative level.
        """
        shifted_text = text

        # Check if any term from the target level is already in the text
        target_indicators = self.pejorative_mentions.get(pejorative, [])
        if any(re.search(mention, shifted_text, flags=re.IGNORECASE) for mention in target_indicators):
            # If the text already has a word from the target level, return it as-is
            return shifted_text

        # Remove mentions from other levels
        for pejorative_key, indicators in self.pejorative_mentions.items():
            if pejorative_key != pejorative:
                for mention in indicators:
                    shifted_text = re.sub(mention, '', shifted_text, flags=re.IGNORECASE)

        # If the target level is NO_MENTION, return text without any pejorative words
        if pejorative is PejorativeLevel.NO_MENTION:
            return shifted_text

        # Choose a random synonym for the specified level
        pejorative_mention = random.choice(target_indicators)
        mention_insertion_position = utils.find_patient_characteristic_position_in_text(shifted_text)

        # Insert the chosen mention if an insertion position is found
        if mention_insertion_position is not None:
            shifted_text = f"{shifted_text[:mention_insertion_position]}{pejorative_mention} {shifted_text[mention_insertion_position:]}"

        return shifted_text