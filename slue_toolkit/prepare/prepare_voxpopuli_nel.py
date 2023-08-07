"""
Extract entity phrase and duration tuples for evaluation
"""

from datasets import load_dataset

import csv
from fire import Fire
import os
import sys
from tqdm import tqdm

from slue_toolkit.generic_utils import read_lst, save_dct


def depunctuate(text):
    """
    Also removes isolated apostrophe
    """
    text = text.replace(".", "")
    text = text.replace("' ", " ")
    text = text.replace("'s", " 's")
    text = text.replace("  ", " ")
    return text


def read_gt_sample(gt_labels, data_idx):
    entity_phrase = depunctuate(gt_labels[data_idx][0])
    wrd_lst = entity_phrase.split(" ")
    return wrd_lst, len(wrd_lst)


def update_alignment_dct(all_word_alignments, utt_id, gt_labels):
    """
    Update alignment_dct by appending word belonging to entity phrases with #
    """
    text = " ".join([item[0] for item in all_word_alignments[utt_id]])
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")
    if len(gt_labels) > 0:
        data_idx, curr_idx = 0, 0
        wrd_lst, len_phrase = read_gt_sample(gt_labels, data_idx)
        while data_idx < len(gt_labels):  # until a match for all GT entities is found
            label, _, _ = all_word_alignments[utt_id][curr_idx]
            if label == wrd_lst[0]:
                update_words = True
                if len_phrase > 1:
                    done_processing = False
                else:
                    done_processing = True
                gt_idx = 1
                tier_idx = 1
                while not done_processing:
                    label, _, _ = all_word_alignments[utt_id][curr_idx + tier_idx]
                    if label != "":
                        if label != wrd_lst[gt_idx]:
                            done_processing = True
                            update_words = False
                        else:
                            gt_idx += 1
                    tier_idx += 1
                    if gt_idx == len_phrase:
                        done_processing = True

                if update_words:
                    for idx in range(curr_idx, curr_idx + tier_idx):
                        label, start_time, end_time = all_word_alignments[utt_id][idx]
                        all_word_alignments[utt_id][idx] = (
                            "#" + label,
                            start_time,
                            end_time,
                        )
                    curr_idx += len_phrase
                    data_idx += 1
                else:
                    curr_idx += 1
                if data_idx < len(gt_labels):
                    wrd_lst, len_phrase = read_gt_sample(gt_labels, data_idx)
            else:
                curr_idx += 1
            if curr_idx == len(all_word_alignments[utt_id]) and data_idx != len(
                gt_labels
            ):
                print(data_idx, len(gt_labels))
                print(gt_labels)
                print(text)
                sys.exit("Process exited, possibly an issue with text processing.")


def modify_word_alignments(dataset_obj, manifest_dir, data_split, extract_gt=False):
    """
    Modify the word alignments to mark entity phrases for evaluation
    """
    entity_csv = []
    entity_csv_header = ["id", "entity_phrase", "start", "end", "entity_label"]
    all_word_alignments = {}
    for idx, item in enumerate(tqdm(dataset_obj)):
        utt_id = dataset_obj[idx]["id"]
        all_word_alignments[utt_id] = []
        gt_labels = []

        nel_labels = dataset_obj[idx]["ne_timestamps"]
        num_ne = len(nel_labels["ne_label"])
        if num_ne != 0:
            txt = dataset_obj[idx]["text"]
            for ne_idx in range(num_ne):
                start_id = nel_labels["start_char_idx"][ne_idx]
                length = nel_labels["char_offset"][ne_idx]
                phrase = txt[start_id : start_id + length]
                gt_labels.append((phrase, start_id, length))
                t0 = nel_labels["start_sec"][ne_idx]
                t1 = nel_labels["end_sec"][ne_idx]
                entity_label = nel_labels["ne_label"][ne_idx]
                entity_csv.append(
                    [utt_id, txt[start_id : start_id + length], t0, t1, entity_label]
                )

        wrd_durs = dataset_obj[idx]["word_timestamps"]
        num_wrds = len(wrd_durs["word"])
        for wrd_idx in range(num_wrds):
            word = wrd_durs["word"][wrd_idx]
            start_sec = wrd_durs["start_sec"][wrd_idx]
            end_sec = wrd_durs["end_sec"][wrd_idx]
            all_word_alignments[utt_id].append((word, start_sec, end_sec))
        update_alignment_dct(all_word_alignments, utt_id, gt_labels)
    save_dct(
        os.path.join(manifest_dir, f"{data_split}_tagged_word_alignments.json"),
        all_word_alignments,
    )
    if extract_gt:
        with open(
            os.path.join(manifest_dir, f"{data_split}_entity_alignments.csv"), "w"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(entity_csv_header)
            writer.writerows(entity_csv)


def main(manifest_dir="manifest/slue-voxpopuli/nel_test", is_blind=True):
    dataset = load_dataset("asapp/slue-phase-2", "vp_nel")
    split_lst = ["validation", "test"]
    os.makedirs(manifest_dir, exist_ok=True)
    for data_split in split_lst:
        extract_gt = data_split == "validation" or not is_blind
        modify_word_alignments(
            dataset[data_split], manifest_dir, data_split, extract_gt
        )


if __name__ == "__main__":
    Fire(main)
