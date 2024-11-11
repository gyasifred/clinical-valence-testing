import re
import random
from enum import Enum
import utils
from test_shifts.base_shift import BaseShift


class LaudatoryLevel(Enum):
    COMPLIANT = 1
    COOPERATIVE = 2
    PLEASANT = 3
    RESPECTFUL = 4
    NO_MENTION = 5


class LaudatoryShift(BaseShift):
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

    def get_groups(self):
        """Returns a list of all LaudatoryLevel enum members."""
        return list(LaudatoryLevel)

    def get_group_names(self):
        """Returns the names of each laudatory level for display or analysis."""
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: LaudatoryLevel):
        """Applies the text_to_laudatory method for the specified laudatory level."""
        return self.text_to_laudatory(sample, group)

    def identify_group_in_text(self, text: str):
        """Identifies if the text contains any terms from laudatory_mentions.
        
        Returns:
            LaudatoryLevel: The first matching laudatory level in the text,
            or NO_MENTION if no terms are found.
        """
        for laudatory_key in self.laudatory_mentions:
            indicators = self.laudatory_mentions[laudatory_key]

            for mention in indicators:
                if re.search(mention, text, flags=re.IGNORECASE):
                    return laudatory_key

        return LaudatoryLevel.NO_MENTION

    def text_to_laudatory(self, text: str, laudatory: LaudatoryLevel):
        """Modifies the text to contain a laudatory descriptor from the specified level.
        
        This method:
            - Checks if any term from the target laudatory level already exists.
            - Removes terms from other levels.
            - If the target level is NO_MENTION, returns text without laudatory mentions.
            - Inserts a randomly selected term from the target level if no existing term matches.

        Args:
            text (str): The input text to modify.
            laudatory (LaudatoryLevel): The target laudatory level for insertion.

        Returns:
            str: The modified text with the specified laudatory level.
        """
        shifted_text = text

        # Check if any term from the target level is already in the text
        target_indicators = self.laudatory_mentions.get(laudatory, [])
        if any(re.search(mention, shifted_text, flags=re.IGNORECASE) for mention in target_indicators):
            # If the text already has a word from the target level, return it as-is
            return shifted_text

        # Remove mentions from other levels
        for laudatory_key, indicators in self.laudatory_mentions.items():
            if laudatory_key != laudatory:
                for mention in indicators:
                    shifted_text = re.sub(mention, '', shifted_text, flags=re.IGNORECASE)

        # If the target level is NO_MENTION, return text without any laudatory words
        if laudatory is LaudatoryLevel.NO_MENTION:
            return shifted_text

        # Choose a random synonym for the specified level
        laudatory_mention = random.choice(target_indicators)
        mention_insertion_position = utils.find_patient_characteristic_position_in_text(shifted_text)

        # Insert the chosen mention if an insertion position is found
        if mention_insertion_position is not None:
            shifted_text = f"{shifted_text[:mention_insertion_position]}{laudatory_mention} {shifted_text[mention_insertion_position:]}"

        return shifted_text