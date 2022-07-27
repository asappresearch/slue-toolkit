"""
Evaluate the output of wav2vec finetuned on NER using CTC loss
"""

import fire
import numpy as np
import os
import sys

sys.path.insert(0, "../")

from slue_toolkit.eval import eval_utils
from slue_toolkit.generic_utils import (
    read_lst,
    save_dct,
    spl_char_to_entity,
    raw_to_combined_tag_map,
)


def make_distinct(label_lst):
    """
    Make the label_lst distinct
    """
    tag2cnt, new_tag_lst = {}, []
    if len(label_lst) > 0:
        for tag_item in label_lst:
            _ = tag2cnt.setdefault(tag_item, 0)
            tag2cnt[tag_item] += 1
            tag, wrd = tag_item
            new_tag_lst.append((tag, wrd, tag2cnt[tag_item]))
        assert len(new_tag_lst) == len(set(new_tag_lst))
    return new_tag_lst


def get_gt_pred(score_type, eval_label, eval_set, decoded_data_dir):
    """
    Read the GT and predicted utterances in the entity format [(word1, tag1), (word2, tag2), ...]
    """
    entity_end_char = "]"
    entity_to_spl_char = {}
    for spl_char, entity in spl_char_to_entity.items():
        entity_to_spl_char[entity] = spl_char

    def update_label_lst(lst, phrase, label):
        if eval_label == "combined":
            label = raw_to_combined_tag_map[label]
        if label != "DISCARD":
            if score_type == "label":
                lst.append((label, "phrase"))
            else:
                lst.append((label, phrase))

    sent_lst_dct = {"hypo": [], "ref": []}
    label_lst_dct = {"hypo": [], "ref": []}
    for pfx in ["hypo", "ref"]:
        all_text = read_lst(
            os.path.join(
                decoded_data_dir,
                f"{pfx}.word-checkpoint_best.pt-{eval_set}.txt",
            )
        )
        all_text = [line.split(" (None")[0] for line in all_text]
        for line in all_text:
            label_lst = []
            line = line.replace("  ", " ")
            wrd_lst = line.split(" ")
            sent_lst_dct[pfx].append(line)
            phrase_lst, is_entity, num_illegal_assigments = [], False, 0
            for idx, wrd in enumerate(wrd_lst):
                if wrd in spl_char_to_entity:
                    if (
                        is_entity
                    ):  # a new entity began before completion of the previous entity
                        phrase_lst = []  # discard the ongoing entity
                        num_illegal_assigments += 1
                    is_entity = True
                    entity_tag = spl_char_to_entity[wrd]
                elif wrd == entity_end_char:
                    if is_entity:
                        if len(phrase_lst) > 0:
                            update_label_lst(
                                label_lst, " ".join(phrase_lst), entity_tag
                            )
                        else:  # entity end without entity start
                            num_illegal_assigments += 1
                        phrase_lst = []
                        is_entity = False
                    else:
                        num_illegal_assigments += 1
                else:
                    if is_entity:
                        phrase_lst.append(wrd)
            label_lst_dct[pfx].append(make_distinct(label_lst))

    return label_lst_dct, sent_lst_dct


def eval_ner(
    model_dir,
    eval_set="dev",
    eval_label="combined",
    lm="nolm",
    lm_sfx=None,
    save_results=False,
):
    if "nolm" in lm:
        decoded_data_dir = os.path.join(model_dir, "decode", "nolm")
    else:
        assert lm_sfx is not None
        decoded_data_dir = os.path.join(model_dir, "decode", f"{lm}-{lm_sfx}")
    log_dir = os.path.join(model_dir, "metrics")
    os.makedirs(log_dir, exist_ok=True)
    if save_results:
        ner_results_dir = os.path.join(log_dir, "error_analysis")
        os.makedirs(ner_results_dir, exist_ok=True)

    for score_type in ["standard", "label"]:
        res_fn = "-".join([eval_set, lm, eval_label, score_type])
        labels_dct, text_dct = get_gt_pred(
            score_type, eval_label, eval_set, decoded_data_dir
        )
        if save_results and score_type == "standard":
            analysis_examples_dct = eval_utils.error_analysis(
                labels_dct["ref"], labels_dct["hypo"], text_dct["ref"]
            )
            save_dct(
                os.path.join(ner_results_dir, res_fn + ".json"), analysis_examples_dct
            )

        metrics = eval_utils.get_ner_scores(labels_dct["ref"], labels_dct["hypo"])
        save_dct(os.path.join(log_dir, res_fn + ".json"), metrics)
        print(
            "[%s, %s, %s label set, micro-averaged %s]: %.1f"
            % (os.path.basename(model_dir), eval_set, eval_label, score_type, 100 * metrics["overall_micro"]["fscore"])
        )


if __name__ == "__main__":
    fire.Fire()
