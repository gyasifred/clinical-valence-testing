import re
import random
from enum import Enum
import utils
from test_shifts.base_shift import BaseShift


class ValenceType(Enum):
    NEUTRAL = 1
    NO_MENTION = 2


class NeutralValShift(BaseShift):
    # Define the neutral terms associated with the NEUTRAL valence level
    neutral_mentions = {
        ValenceType.NEUTRAL: [
            "typical", "average", "regular", "standard", "usual",
            "presenting", "referred", "evaluated", "assessed", "monitored"
        ],
    }

    def get_groups(self):
        """Returns a list of all ValenceType enum members."""
        return list(ValenceType)

    def get_group_names(self):
        """Returns the names of each valence type for display or analysis."""
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: ValenceType):
        """Applies the text_to_neutral method for the specified valence level."""
        return self.text_to_neutral(sample, group)

    def identify_group_in_text(self, text: str):
        """Identifies if the text contains any terms from neutral_mentions.
        
        Returns:
            ValenceType: The first matching valence level in the text,
            or NO_MENTION if no terms are found.
        """
        for neutral_key in self.neutral_mentions:
            indicators = self.neutral_mentions[neutral_key]

            for mention in indicators:
                if re.search(mention, text, flags=re.IGNORECASE):
                    return neutral_key

        return ValenceType.NO_MENTION

    def text_to_neutral(self, text: str, neutral: ValenceType):
        """Modifies the text to contain a neutral descriptor from the specified level.
        
        This method:
            - Checks if any term from the target neutral level already exists.
            - Removes terms from other levels.
            - If the target level is NO_MENTION, returns text without neutral mentions.
            - Inserts a randomly selected term from the target level if no existing term matches.

        Args:
            text (str): The input text to modify.
            neutral (ValenceType): The target valence level for insertion.

        Returns:
            str: The modified text with the specified neutral level.
        """
        shifted_text = text

        # Check if any term from the target level is already in the text
        target_indicators = self.neutral_mentions.get(neutral, [])
        if any(re.search(mention, shifted_text, flags=re.IGNORECASE) for mention in target_indicators):
            # If the text already has a word from the target level, return it as-is
            return shifted_text

        # Remove mentions from other levels (though in this case, we have just one level)
        for neutral_key, indicators in self.neutral_mentions.items():
            if neutral_key != neutral:
                for mention in indicators:
                    shifted_text = re.sub(mention, '', shifted_text, flags=re.IGNORECASE)

        # If the target level is NO_MENTION, return text without any neutral words
        if neutral is ValenceType.NO_MENTION:
            return shifted_text

        # Choose a random synonym for the specified level (for now, just using the first neutral term)
        neutral_mention = random.choice(target_indicators)
        mention_insertion_position = utils.find_patient_characteristic_position_in_text(shifted_text)

        # Insert the chosen mention if an insertion position is found
        if mention_insertion_position is not None:
            shifted_text = f"{shifted_text[:mention_insertion_position]}{neutral_mention} {shifted_text[mention_insertion_position:]}"

        return shifted_text
