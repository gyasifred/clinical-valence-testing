import re
from pathlib import Path
from typing import Union, Dict
from functools import lru_cache
import pandas as pd
from transformers import PreTrainedTokenizer
import torch


def codes_that_occur_n_times_in_dataset(n: int, dataset_path: str, code_label: str = "short_codes"):
    df = pd.read_csv(dataset_path)
    code_count = {}
    for i, row in df.iterrows():
        codes = row[code_label].split(",")
        for code in codes:
            if code in code_count:
                code_count[code] += 1
            else:
                code_count[code] = 1
    codes_sorted_by_occurrence = dict(sorted(code_count.items(), key=lambda item: item[1], reverse=True))

    return list(codes_sorted_by_occurrence.keys())[:n]


def collate_batch(data):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x['input_ids']) for x in data],
        batch_first=True)

    attention_masks = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x['attention_mask']) for x in data],
        batch_first=True)

    token_type_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x['token_type_ids']) for x in data],
        batch_first=True)

    return {"input_ids": input_ids,
            "attention_masks": attention_masks,
            "token_type_ids": token_type_ids,
            "tokens": [x['tokens'] for x in data],
            "targets": [x['target'] for x in data]}


def sample_to_features(sample: Union[Dict, pd.Series], tokenizer: PreTrainedTokenizer, max_length=512,
                       text_column="text",
                       label_column="label"):
    tokenized = tokenizer.encode_plus(sample[text_column],
                                      truncation=True,
                                      padding=True,
                                      max_length=max_length)

    featurized_sample = {"input_ids": tokenized["input_ids"],
                         "attention_mask": tokenized["attention_mask"],
                         "token_type_ids": tokenized["token_type_ids"],
                         "tokens": tokenized.encodings[0].tokens,
                         "target": sample[label_column]}

    return featurized_sample


def save_to_file(content: Union[str, list, dict], file_path: str):
    with open(file_path, "w") as write_file:
        write_file.write(str(content))


# def create_dir_if_not_exists(directory: str):
#     Path(directory).mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=1024)
def find_patient_characteristic_position_in_text(text: str):
    person_patterns = [
        r"(female)", r"(woman)", r"(lady)", r" (male) ", r" (male)\.", r" (male),", r" (man) ", r" (man)\.",
        r" (man),", r"(gentleman)"
    ]

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
    ethnicity_pattern = [
        r"\b(?:African[ -]American|Hispanic|Latino|Latina|Asian|Caucasian|White|Black)\b",
        r"\b(?:Indigenous|Native[ -]American|Pacific Islander|Alaska Native)\b",
        r"\b(?:South Asian|East Asian|Southeast Asian|Middle Eastern|Arab)\b",
        r"\b(?:Caribbean|African|European|Mediterranean|Slavic)\b"
    ]

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

    for category, patterns in compiled_patterns.items():
        for pattern in patterns:
            result = re.search(pattern, text)
            if result is not None:
                # Attempt to get group position or set to 0 if IndexError
                try:
                    pattern_pos = result.regs[1]
                    if category == 'age':
                        # Return position AFTER age mention (pattern_pos[1] + 1)
                        return pattern_pos[1] + 1
                    else:
                        return pattern_pos[0]
                except IndexError:
                    return 0
    
    # Return 0 if no pattern is found
    return 0