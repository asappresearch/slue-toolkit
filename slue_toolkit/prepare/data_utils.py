import pickle as pkl
from slue_toolkit.generic_utils import raw_to_combined_tag_map, save_pkl, write_to_file

def load_pkl(fname, encdng=None):
    if encdng is None:
        with open(fname, "rb") as f:
            data = pkl.load(f)
    else:
        with open(fname, "rb") as f:
            data = pkl.load(f, encoding=encdng)
    return data


def remove_punc(sent):
    """
    Filter out punctuations
    """
    punc_lst = [".", ",", "!", "?", ";"]
    for punctuation in punc_lst:
        sent = sent.replace(punctuation, "")
    sent = sent.replace("  ", " ")
    return sent


def get_label_lst(label_str, label_type):
    """
    Convert list of list string into list of tuples
    """
    if label_str == "None" or label_str == "[]":
        return []
    tag_map = raw_to_combined_tag_map
    label_lst = []
    ner_labels_lst = label_str.strip("[[").strip("]]").split("], [")
    for item in ner_labels_lst:
        label, start_id, length = item.split(", ")
        label = label[1:-1]
        start_id = int(start_id)
        length = int(length)
        if label_type == "combined":
            if tag_map[label] != "DISCARD":
                label_lst.append((tag_map[label], start_id, length))
        else:
            label_lst.append((label, start_id, length))
    return label_lst


def reformat_wrd(wrd):
    """
    1. Clean up special cases of standalone apostrophes
    2. Detect valid apostrophe cases and split those into a two words
    """
    if wrd[-1] == "'":
        wrd = wrd[:-1]
    if "'" in wrd and wrd != "'s":
        wrd_lst = wrd.split("'")
        wrd_lst[-1] = "'" + wrd_lst[-1]
    else:
        wrd_lst = [wrd]
    return wrd_lst


def update_pairs_non_entity(pairs, segment):
    """
    Prepare (non-entity word, 'O') pairs
        - Filter punctuations and split the segment into words
    """
    segment = remove_punc(segment)
    segment = segment.strip()
    spl_cases = ["", "'"]
    if segment not in spl_cases:
        wrd_lst = segment.split(" ")
        for wrd in wrd_lst:
            if wrd not in spl_cases:
                wrd = reformat_wrd(wrd)
                for item in wrd:
                    pairs.append("\t".join([item, "O"]))


def update_pairs_entity(pairs, entity_seg_dct, seg_start, seg_end, orig_sent):
    """
    Prepare (entity word, entity tag) pairs
    """
    end_id, label = entity_seg_dct[seg_start]
    assert end_id == seg_end
    entity_phrase = orig_sent[seg_start:seg_end]
    assert entity_phrase[0] != " " and entity_phrase[-1] != " "
    # no punctuations in entity char
    entity_phrase = remove_punc(entity_phrase)
    assert entity_phrase[0] != " " and entity_phrase[-1] != " "
    entity_phrase = entity_phrase.split(" ")
    for idx, wrd in enumerate(entity_phrase):
        wrd = reformat_wrd(wrd)
        for idx1, item in enumerate(wrd):
            if idx == 0 and idx1 == 0:
                pairs.append("\t".join([item, "B-" + label]))
            else:
                pairs.append("\t".join([item, "I-" + label]))


def get_segment_indices(start_id_lst, end_id_lst, orig_sent):
    """
    Get segment indices, separting entity segments from the rest
    e.g) If label is (['LAW', 86, 18],['ORG', 110, 25]) then the output index list is
         >> [[0, 86], [86, 104], [104, 110], [110, 135], [135, len(orig_sent)]]
    """
    assert len(start_id_lst) == len(end_id_lst)
    num_phrases = len(start_id_lst)
    for idx, start_id in enumerate(start_id_lst):
        if idx == 0:
            out_idx_lst = []
            if start_id > 0:
                out_idx_lst.append([0, start_id_lst[0]])
        out_idx_lst.append([start_id, end_id_lst[idx]])
        if idx == num_phrases - 1:
            out_idx_lst.append([end_id_lst[idx], len(orig_sent)])
        elif end_id_lst[idx] != start_id_lst[idx + 1]:
            out_idx_lst.append([end_id_lst[idx], start_id_lst[idx + 1]])
    return out_idx_lst


def create_wrd_label_pairs(label_lst, orig_sent):
    """
    Create a list of tab separated word-label pairs (for NLP NER models)
    """
    entity_seg_dct, start_id_lst, end_id_lst = {}, [], []
    pairs = []
    for label, start_id, length in label_lst:
        start_id = int(start_id)
        end_id = start_id + int(length)
        entity_seg_dct[start_id] = (end_id, label)
        start_id_lst.append(start_id)
        end_id_lst.append(end_id)
    start_id_lst.sort()
    end_id_lst.sort()
    out_idx_lst = get_segment_indices(start_id_lst, end_id_lst, orig_sent)
    for seg_start, seg_end in out_idx_lst:
        if seg_start in entity_seg_dct:
            update_pairs_entity(pairs, entity_seg_dct, seg_start, seg_end, orig_sent)
        else:
            update_pairs_non_entity(pairs, orig_sent[seg_start:seg_end])
    return "\n".join(pairs)


def prep_text_ner_tsv(normalized_text, ner_labels_str, label_type):
    out_str = ""
    ner_labels_lst = get_label_lst(ner_labels_str, label_type)
    if len(ner_labels_lst) != 0:
        out_str += create_wrd_label_pairs(ner_labels_lst, normalized_text) + "\n\n"
    else:
        pairs = []
        update_pairs_non_entity(pairs, normalized_text)
        out_str += "\n".join(pairs) + "\n\n"
    return out_str


def prep_e2e_ner_files(entity_pair_str, label_type):
    """
    Prepare files for finetuning using CTC loss (fairseq codebase)
    """
    if label_type == "raw":
        from slue_toolkit.generic_utils import (
            raw_entity_to_spl_char as entity_to_spl_char,
        )
    elif label_type == "combined":
        from slue_toolkit.generic_utils import (
            combined_entity_to_spl_char as entity_to_spl_char,
        )
    else:
        raise ValueError(
            f'label_type={label_type} which should be either "raw" or "combined'
        )
    end_char = "]"
    entity = False
    out_sent = []
    wrd_lst = entity_pair_str.strip().split("\n")
    for idx, line in enumerate(wrd_lst):
        wrd, tag = line.split("\t")
        if tag != "O":
            if entity:
                if tag[:2] == "I-":
                    assert tag == "I-" + prev_tag
                    curr_wrd += " " + wrd
                elif tag[:2] == "B-":
                    out_sent.append(curr_wrd + " ]")
                    curr_wrd = f"{entity_to_spl_char[tag[2:]]} {wrd}"
                prev_tag = tag[2:]
            else:
                assert tag[:2] == "B-"
                prev_tag = tag[2:]
                curr_wrd = f"{entity_to_spl_char[tag[2:]]} {wrd}"
                entity = True
        else:
            if entity:
                curr_wrd += f" {end_char} {wrd}"
                entity = False
            else:
                curr_wrd = wrd
        if not entity:
            out_sent.append(curr_wrd)
    if entity:
        out_sent.append(f"{curr_wrd} {end_char}")
    wrd_str = " ".join(out_sent)
    ltr_str = " ".join(list(" ".join(out_sent).replace(" ", "|"))) + " |"
    return wrd_str, ltr_str


def prepare_tag_id_mapping(label_type):
    """
    Prepare tag2id and id2tag mappings for the text NER model
    """
    tag2id = {}
    tag_lst_ordered = []

    if label_type == "raw":
        all_tag_lst = raw_to_combined_tag_map.keys()
    elif label_type == "combined":
        all_tag_lst = list(set(raw_to_combined_tag_map.values()))
        all_tag_lst.remove("DISCARD")

    for idx, tag in enumerate(all_tag_lst):
        tag2id[f"B-{tag}"] = 2*idx
        tag2id[f"I-{tag}"] = 2*idx+1
        tag_lst_ordered.extend([f"B-{tag}", f"I-{tag}"])
    tag2id["O"] = 2*len(all_tag_lst)
    tag_lst_ordered.append("O")

    id2tag = {v: k for k, v in tag2id.items()}

    return tag2id, id2tag, tag_lst_ordered