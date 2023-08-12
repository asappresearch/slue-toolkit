"""
Evaluate entity alignments
"""

import ast
from copy import deepcopy
import editdistance
from fire import Fire
from glob import glob
import numpy as np
import os
import pandas as pd
import textgrids
from tqdm import tqdm

from slue_toolkit.generic_utils import end_char, spl_char_to_entity
from slue_toolkit.generic_utils import read_lst, load_dct, save_dct, write_to_file
from slue_toolkit.generic_utils import get_file_identifiers
import slue_toolkit.eval.eval_utils_nel as EM


class EvalMetrics:
    def __init__(
        self,
        task,
        split="dev",
        model_name="w2v2-base-debug_phase_ratio",
        lm="nolm-argmax",
        offset=0.0,
        blank=True,
        data_dir="manifest/slue-voxpopuli/nel"
    ):
        self.split = split
        self.task = task
        self.start_char_lst = list(spl_char_to_entity.keys())
        self.spl_char_tokens = self.start_char_lst + [end_char]
        self.data_dir = data_dir
        self.all_word_alignments = load_dct(os.path.join(self.data_dir, f"{self.split}_tagged_word_alignments.json"))
        
        self.output_dir = f"save/{self.get_baseline_task_name()}/{model_name}/decode/{split}/{lm}"
        self.ner_model_dir = "save/text_ner/deberta-base_raw/metrics/error_analysis/"
        if task == "ppl":
            self.ppl_output_dct = load_dct(
                os.path.join(
                    self.ner_model_dir,
                    f"{split}-pipeline-{model_name}-{lm}-combined-standard.json",
                )
            )
        elif task == "oracle_ppl":
            self.text_ner_output_dct = load_dct(
                os.path.join(
                    self.ner_model_dir, f"{split}-gt-text-combined-standard.json"
                )
            )
    
        self.utt_lst, self.spk_lst, self.nel_utt_lst = get_file_identifiers(self.split)
        self.offset = offset
        self.blank = blank
        self.save_fn = self.get_save_fname()

    def get_baseline_task_name(self):
        if self.task == "e2e":
            return "e2e_ner"
        elif self.task == "ppl":
            return "asr"
        elif self.task == "oracle_ppl":
            return "text_ner"

    def get_save_fname(self):
        save_fn = f"{self.split}_{self.task}"
        if "oracle" not in self.task:
            save_fn += f"_offset{self.offset}"
            if self.blank:
                save_fn += "_w-blank"
            else:
                save_fn += "_wo-blank"
        return f"{save_fn}.json"

    def clip_dur(self, dur):
        """
        adjust for offset
        """
        dur = np.round(dur, 2) + self.offset
        dur = np.max([dur, 0])
        return dur

    def generate_pred_alignments_ppl(self, char_lst, dur_lst, ppl_output):
        def update_wrd_lst(wrd_dur_tuple, start_time, end_time, wrd):
            """
            Add the tuple of word and duration to the list; also handles apostrophe cases
            """
            start_time = np.around(start_time, 2)
            end_time = np.round(end_time, 2)
            if len(wrd) > 2 and "'s" in wrd:
                write_lst = wrd.split("'s")
                assert len(write_lst) == 2
                write_lst[1] = "'s" + write_lst[1]
            else:
                write_lst = [wrd]
            for idx, write_sample in enumerate(write_lst):
                if idx == 0:
                    wrd_dur_tuple.append((write_sample, start_time, end_time))
                else:
                    wrd_dur_tuple.append((write_sample, end_time, end_time))

        def convert_char_to_wrd_lst(char_lst, dur_lst):
            """
            Convert list of char and corresponding duration to a list of tuple of words and corresponding duration
            """
            wrd_dur_tuple = []
            wrd = ""
            curr_time_step = 0
            record_wrd = False
            for char, duration in zip(char_lst, dur_lst):
                duration = np.round(duration, 2)
                if char != "|":
                    if self.blank:
                        if not record_wrd:
                            assert wrd == ""
                            record_wrd = True
                            start_time = curr_time_step
                    else:
                        if char != "<b>":
                            if wrd == "":
                                start_time = curr_time_step  # start recording
                                record_wrd = True
                            end_time = curr_time_step + duration
                    if record_wrd and char != "<b>":
                        wrd += char
                elif char == "|":
                    if len(wrd) > 0:
                        if self.blank:
                            update_wrd_lst(wrd_dur_tuple, start_time, curr_time_step, wrd)
                        else:
                            update_wrd_lst(wrd_dur_tuple, start_time, end_time, wrd)
                    wrd = ""
                    record_wrd = False
                curr_time_step += duration
            if len(wrd) > 0:
                if self.blank:
                    update_wrd_lst(wrd_dur_tuple, start_time, curr_time_step, wrd)
                else:
                    update_wrd_lst(wrd_dur_tuple, start_time, end_time, wrd)
            return wrd_dur_tuple
        
        wrd_dur_tuple = convert_char_to_wrd_lst(char_lst, dur_lst)
        _, _, _, pred_lst = ppl_output.split("\t")
        pred_lst = ast.literal_eval(pred_lst)
        pred_tuple = []
        for _, start_id, end_id, phrase in pred_lst:
            pred_phrase = " ".join([wrd_dur_tuple[i][0] for i in range(start_id, end_id + 1)])
            try:
                assert pred_phrase == phrase
            except:
                print(phrase)
                print(pred_phrase)
                import pdb; pdb.set_trace()
            start_dur = self.clip_dur(wrd_dur_tuple[start_id][1])
            end_dur = self.clip_dur(wrd_dur_tuple[end_id][2])
            pred_tuple.append((phrase, np.round(start_dur, 2), np.round(end_dur, 2)))

        return pred_tuple

    def generate_pred_alignments_e2e(self, char_lst, dur_lst):
        """
        start: start of start_char
        end: end of end_char
        """
        def get_end_dur(char_lst, dur_lst, idx):
            assert char_lst[idx] == end_char
            end_dur = dur_lst[idx]
            if self.blank:
                return end_dur
            else:
                end_dur -= dur_lst[idx]
                offset = 1
                time_offset = 0
                while idx-offset>0 and char_lst[idx-offset] != "|":
                    time_offset += dur_lst[idx-offset]
                    offset += 1
                time_offset += dur_lst[idx-offset]
                if offset not in [1, 2]: # no word seperator token
                    return end_dur
                end_dur -= time_offset
                return end_dur

        entity_phrase = ""
        pred_tuple = []
        record_entity = False
        curr_time_step = 0
        entity_end = False
        for char_idx, char in enumerate(char_lst):
            duration = dur_lst[char_idx]
            if record_entity:
                if char == end_char:
                    if record_entity:  # there was a corresponding start_char
                        end_dur = get_end_dur(char_lst, dur_lst, char_idx)
                        pred_tuple.append(
                            (
                                entity_phrase.strip(),
                                np.round(self.clip_dur(start_dur), 2),
                                np.round(self.clip_dur(curr_time_step + end_dur), 2),
                            )
                        )
                        entity_end = True
                    record_entity = False
                    entity_phrase = ""
                elif char == "|":
                    entity_phrase += " "
                elif char != "<b>":
                    if entity_phrase.strip() == "" and not self.blank:
                        start_dur = curr_time_step
                    entity_phrase += char
            if char in self.start_char_lst:
                if self.blank:
                    start_dur = curr_time_step
                record_entity = True
                entity_phrase = ""
            curr_time_step += duration
        return pred_tuple

    def read_pred_alignments_ctc(self, utt_idx, pred_entity_dct):
        fname = os.path.join(self.output_dir, "ctc_outputs", f"{utt_idx}_char.lst")
        assert os.path.exists(fname)
        char_lst = read_lst(fname)
        dur_lst = list(map(float, read_lst(fname.replace("char", "dur"))))
        if self.task == "e2e":
            pred_entity_dct[utt_idx] = self.generate_pred_alignments_e2e(
                char_lst, dur_lst
            )
        elif self.task == "ppl":
            pred_entity_dct[utt_idx] = self.generate_pred_alignments_ppl(
                char_lst, dur_lst, self.ppl_output_dct[str(utt_idx)]
            )

    def read_pred_alignments_text_ner(self, utt_idx, pred_entity_dct, wrd_alignments):
        ppl_output = self.text_ner_output_dct[str(utt_idx)]
        _, pred_txt, _, pred_lst = ppl_output.split("\t")
        pred_lst = ast.literal_eval(pred_lst)
        pred_tuple = []
        wrd_dur_tuple = []
        for wrd, start_time, end_time in wrd_alignments:
            if wrd != "" and wrd != "#": # no empty string
                if wrd[0] == "#":
                    wrd = wrd[1:]
                if wrd == "dieselgate'": # special case in the transcript
                    wrd = "dieselgate"
                if wrd[0] == "'":
                    lst_of_wrds = [wrd]
                else:
                    lst_of_wrds = wrd.split("'")
                for idx, wrd in enumerate(lst_of_wrds):
                    if idx == 0:
                        wrd_dur_tuple.append((wrd, start_time, end_time))
                    else:
                        wrd_dur_tuple.append(("'" + wrd, end_time, end_time))
                    assert idx < 2
        for _, start_id, end_id in pred_lst:
            phrase = " ".join(pred_txt.split(" ")[start_id : end_id + 1])
            pred_phrase = " ".join([wrd_dur_tuple[i][0] for i in range(start_id, end_id + 1)])
            try:
                assert phrase == pred_phrase
            except:
                print(phrase)
                print(pred_phrase)
                import pdb; pdb.set_trace()
            start_dur = wrd_dur_tuple[start_id][1]
            end_dur = wrd_dur_tuple[end_id][2]
            pred_tuple.append(
                (pred_phrase, np.round(start_dur, 2), np.round(end_dur, 2))
            )
        pred_entity_dct[utt_idx] = pred_tuple

    def read_gt_entity_alignments(self):
        """
        Generates a dictionary, where each element is a file mapped to a 
        list of tuples with entity phrase and timestamps
        """
        gt_dct = {}
        csv_lst = read_lst(os.path.join(self.data_dir, f"{self.split}_entity_alignments.csv"))
        for line in csv_lst[1:]:
            utt_id, phrase, start, end, _ = line.split(",")
            utt_idx = self.utt_lst.index(utt_id)
            if utt_id in self.nel_utt_lst:
                _ = gt_dct.setdefault(utt_idx, [])
                gt_dct[utt_idx].append((phrase, float(start), float(end)))
        return gt_dct

    def process_alignments(self):
        """
        Format for gt_entity_dct and pred_entity_dct
        {utt_idx: [(phrase_1, start_time_1, end_time_1), .., (phrase_k, start_time_k, end_time_k)]}
        """
        gt_entity_dct = self.read_gt_entity_alignments()
        pred_entity_dct = {}
        all_word_alignments = {}
        for utt_idx in tqdm(range(len(self.utt_lst))):
            utt_id = self.utt_lst[utt_idx]
            if utt_id in self.nel_utt_lst:
                all_word_alignments[utt_idx] = self.all_word_alignments[utt_id]
                if self.task == "oracle_ppl":
                    self.read_pred_alignments_text_ner(
                        utt_idx, pred_entity_dct, all_word_alignments[utt_idx]
                    )
                else:
                    self.read_pred_alignments_ctc(utt_idx, pred_entity_dct)

        res_dct = {
            "word": {"f1": {}, "prec": {}, "recall": {}},
            "frame": {},
        }
        frac_lst = [1, 0.9, 0.8, 0.7, 0.6, 0.5]

        for frac_tol in frac_lst:
            prec, recall, f1 = EM.evaluate_alignments_word(all_word_alignments, pred_entity_dct, gt_entity_dct, frac_tol)
            if frac_tol == 0.8:
                print(frac_tol, np.round(100*prec, 1), np.round(100*recall, 1), np.round(100*f1, 1))
            res_dct["word"]["f1"][frac_tol] = f1
            res_dct["word"]["prec"][frac_tol] = prec
            res_dct["word"]["recall"][frac_tol] = recall

        prec, recall, f1 = EM.evaluate_alignments_frames(
            all_word_alignments, pred_entity_dct, gt_entity_dct
        )
        print(
            "n/a",
            np.round(100 * prec, 1),
            np.round(100 * recall, 1),
            np.round(100 * f1, 1),
        )
        res_dct["frame"]["f1"] = f1
        res_dct["frame"]["prec"] = prec
        res_dct["frame"]["recall"] = recall
        print(self.save_fn)

        save_dct(os.path.join("results", self.save_fn), res_dct)

def evaluate(
    split="dev",
    task="e2e", # ppl | e2e | oracle_ppl
    model_name="w2v2-base",
    lm="nolm-argmax",
    offset=0.0,
    blank=False,
):
    if split == "test" and task != "oracle_ppl":
        best_params = load_dct(f"results/best_params_{task}.json")
        offset = float(best_params["offset"])
        blank = best_params["blank"] == "True"

    eval_obj = EvalMetrics(task, split, model_name, lm, offset, blank)
    eval_obj.process_alignments()

def choose_best(
    task="e2e", # ppl | e2e | oracle_ppl
):
    def get_offset_blank(fname):
        fname = fname.split("/")[-1]
        blank_label = fname.split("_")[-1].split(".")[0]
        if "wo" in blank_label:
            blank_label = "False"
        else:
            blank_label = "True"
        offset = fname.split("_")[-2][6:]
        return blank_label, offset

    best_params_dct = {}
    res_fnames = glob(os.path.join("results", f"dev_{task}_*.json"))
    best_score = 0
    for fname in res_fnames:
        score = load_dct(fname)["frame"]["f1"]
        if score > best_score:
            best_score = score
            best_blank, best_offset = get_offset_blank(fname)
    best_params_dct["blank"] = best_blank
    best_params_dct["offset"] = best_offset
    save_dct(f"results/best_params_{task}.json", best_params_dct)
    print(f"Best frame-F1 score: {best_score} at hyperparameters, offset: {best_offset} and blank: {best_blank}")
    
if __name__ == "__main__":
    Fire()