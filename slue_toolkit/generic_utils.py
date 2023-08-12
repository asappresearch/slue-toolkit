"""
Compilation of commonly used functions
"""

import json
import os
import pickle as pkl

end_char = "]"

raw_entity_to_spl_char = {
    "CARDINAL": "!",
    "DATE": "@",
    "EVENT": "#",
    "FAC": "$",
    "GPE": "%",
    "LANGUAGE": "^",
    "LAW": "&",
    "LOC": "*",
    "MONEY": "(",
    "NORP": ")",
    "ORDINAL": "~",
    "ORG": "`",
    "PERCENT": "{",
    "PERSON": "}",
    "PRODUCT": "[",
    "QUANTITY": "<",
    "TIME": ">",
    "WORK_OF_ART": "?",
}
spl_char_to_entity = {v: k for k, v in raw_entity_to_spl_char.items()}

raw_to_combined_tag_map = {
    "DATE": "WHEN",
    "TIME": "WHEN",
    "CARDINAL": "QUANT",
    "ORDINAL": "QUANT",
    "QUANTITY": "QUANT",
    "MONEY": "QUANT",
    "PERCENT": "QUANT",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "NORP": "NORP",
    "ORG": "ORG",
    "LAW": "LAW",
    "PERSON": "PERSON",
    "FAC": "DISCARD",
    "EVENT": "DISCARD",
    "WORK_OF_ART": "DISCARD",
    "PRODUCT": "DISCARD",
    "LANGUAGE": "DISCARD",
}

combined_entity_to_spl_char = {
    "LAW": "!",
    "NORP": "@",
    "ORG": "#",
    "PERSON": "$",
    "PLACE": "%",
    "QUANT": "^",
    "WHEN": "&",
}


def save_dct(fname, dict_name):
    if ".pkl" in fname:
        save_pkl(fname, dict_name)
    elif ".json" in fname:
        save_json(fname, dict_name)


def save_pkl(fname, dict_name):
    with open(fname, "wb") as f:
        pkl.dump(dict_name, f)


def save_json(fname, dict_name):
    with open(fname, "w") as f:
        f.write(json.dumps(dict_name, indent=4))


def load_dct(fname):
    if ".pkl" in fname:
        data = load_pkl(fname)
    elif ".json" in fname:
        data = load_json(fname)
    return data


def load_json(fname):
    data = json.loads(open(fname).read())
    return data


def load_pkl(fname, encdng=None):
    if encdng is None:
        with open(fname, "rb") as f:
            data = pkl.load(f)
    else:
        with open(fname, "rb") as f:
            data = pkl.load(f, encoding=encdng)
    return data


def read_lst(fname):
    with open(fname, "r") as f:
        lst_from_file = [line.strip() for line in f.readlines()]
    return lst_from_file


def write_to_file(write_str, fname):
    with open(fname, "w") as f:
        f.write(write_str)


def get_file_identifiers(
    split="dev",
    data_dir="data/slue-voxpopuli"
    ):
    """
    Specific to NEL task
    """
    if split == "test":
        split = "test_blind"
    lines = read_lst(os.path.join(data_dir, f"slue-voxpopuli_{split}.tsv"))[1:]
    utt_lst = [line.split("\t")[0] for line in lines]
    spk_lst = [line.split("\t")[3] for line in lines]
    lines = read_lst(os.path.join(data_dir, "../slue-voxpopuli_nel", f"slue-voxpopuli_nel_{split}.tsv"))[1:]
    nel_utt_lst = [line.split("\t")[0] for line in lines]
    return utt_lst, spk_lst, nel_utt_lst