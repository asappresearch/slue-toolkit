"""
Converts ctc emissions into deduplicated character-level time stamps:
{idx}_char.lst 
{idx}_dur.lst (in seconds)
"""

from fire import Fire
import numpy as np
import os

from slue_toolkit.generic_utils import get_file_identifiers
from slue_toolkit.generic_utils import end_char, spl_char_to_entity
from slue_toolkit.generic_utils import read_lst, load_dct, save_dct, write_to_file

class ReadCTCEmissions():
    def __init__(self, split="dev", task="e2e_ner", model_name="w2v2-base", lm="nolm-argmax"):
        self.parent_dir = "/share/data/lang/users/ankitap/entity_localization/"
        self.output_dir = f"save/{task}/{model_name}/decode/{split}"
        self.manifest_dir = "manifest/slue-voxpopuli"
        if task == "e2e_ner":
            self.manifest_dir = os.path.join(self.manifest_dir, "e2e_ner")
        self.data_dir = "data/slue-voxpopuli"
        self.lm = lm
        self.split = split
        self.task = task

    def get_ltr_arr(self):
        if self.task == "e2e_ner":
            dict_fn = os.path.join(self.manifest_dir, "dict.raw.ltr.txt")
        else:
            dict_fn = os.path.join(self.manifest_dir, "dict.ltr.txt")
        with open(dict_fn, "r") as f:
            ltr_lst = [line.split(" ")[0] for line in f.readlines()]
        ltr_lst.insert(0, "<b>")
        ltr_lst.insert(1, "unk")
        ltr_lst.insert(2, "unk")
        ltr_lst.insert(3, "unk")

        return np.array(ltr_lst)

    def deduplicate(self, output_chars, idx):
        """
        Get a list of unique char sequence 
        """
        frame_len = 2e-2
        concise_lst = [output_chars[0]]
        dur_lst = []
        curr_length = 1
        for char in output_chars[1:]:
            if char != concise_lst[-1]:
                concise_lst.append(char)
                dur_lst.append(frame_len*curr_length)
                curr_length = 0
            curr_length += 1
        dur_lst.append(frame_len*curr_length)
        assert len(concise_lst) == len(dur_lst)
        assert np.round(np.sum(dur_lst), 1) == np.round(frame_len*len(output_chars), 1)
        return concise_lst, dur_lst

    def get_char_outputs(self):
        emissions = np.load(os.path.join(self.output_dir, "emissions.npy"), allow_pickle=True)
        write_dir = os.path.join(self.output_dir, self.lm, "ctc_outputs")
        os.makedirs(write_dir, exist_ok=True)
        ltr_arr = self.get_ltr_arr()
        utt_lst, _, nel_utt_lst = get_file_identifiers(self.split)
        for idx in range(len(utt_lst)):
            if utt_lst[idx] in nel_utt_lst: # process NEL corpus utterances only
                item = emissions[idx]
                indices = np.argmax(item, axis=1)
                chars = ltr_arr[indices]
                char_lst, dur_lst = self.deduplicate(chars, idx)
                write_to_file("\n".join(char_lst), os.path.join(write_dir, f"{idx}_char.lst"))
                write_to_file("\n".join(list(map(str, dur_lst))), os.path.join(write_dir, f"{idx}_dur.lst"))


def main(split="dev", task="e2e_ner", model_name="w2v2-base-debug_phase_ratio", lm="nolm-argmax"):
    obj = ReadCTCEmissions(split, task, model_name, lm)
    obj.get_char_outputs()

if __name__ == "__main__":
    Fire(main)
