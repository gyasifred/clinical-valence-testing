import re
from enum import Enum

from test_shifts.base_shift import BaseShift
from test_shifts.laudatory_shift import LaudatoryShift, LaudatoryLevel
from test_shifts.pejorative_shift import PejorativeShift, PejorativeLevel
from test_shifts.neutralVal_shift import NeutralValShift, ValenceType


class NeutralizeType(Enum):
    NEUTRALIZED = 1  


class NeutralizeShift(BaseShift):
    def __init__(self):
        self.laudatory_shift = LaudatoryShift()
        self.pejorative_shift = PejorativeShift()
        self.neutral_shift = NeutralValShift()

    def get_groups(self):
        return list(NeutralizeType)

    def get_group_names(self):
        return [group.name for group in self.get_groups()]

    def get_shift_method(self, sample: str, group: NeutralizeType):
        return self.strip_all_valence(sample)

    def identify_group_in_text(self, text: str):
        has_laudatory = self.laudatory_shift.identify_group_in_text(text) != LaudatoryLevel.NO_MENTION
        has_pejorative = self.pejorative_shift.identify_group_in_text(text) != PejorativeLevel.NO_MENTION
        has_neutral = self.neutral_shift.identify_group_in_text(text) != ValenceType.NO_MENTION

        if not (has_laudatory or has_pejorative or has_neutral):
            return None
        return NeutralizeType.NEUTRALIZED

    def strip_all_valence(self, text: str):
        shifted_text = text

        
        for level in PejorativeLevel:
            if level != PejorativeLevel.NO_MENTION:
                for term in self.pejorative_shift.pejorative_mentions.get(level, []):
                    shifted_text = re.sub(r'\b' + term + r'\b', '', shifted_text, flags=re.IGNORECASE)

        for level in LaudatoryLevel:
            if level != LaudatoryLevel.NO_MENTION:
                for term in self.laudatory_shift.laudatory_mentions.get(level, []):
                    shifted_text = re.sub(r'\b' + term + r'\b', '', shifted_text, flags=re.IGNORECASE)

        for val_type in ValenceType:
            if val_type != ValenceType.NO_MENTION:
                for term in self.neutral_shift.neutral_mentions.get(val_type, []):
                    shifted_text = re.sub(r'\b' + term + r'\b', '', shifted_text, flags=re.IGNORECASE)

        shifted_text = re.sub(r'\s+', ' ', shifted_text).strip()

        return shifted_text