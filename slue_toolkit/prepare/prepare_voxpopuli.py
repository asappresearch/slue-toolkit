import os
import fire
import numpy as np
import pandas as pd
import shutil
import soundfile
import re

from slue_toolkit.prepare import data_utils

splits = {"fine-tune", "dev", "test"}


def create_split(
    input_dir="/persist/data/voxpopuli", output_dir="/persist/data/slue-voxpopuli"
):
    os.makedirs(output_dir, exist_ok=True)
    for split in splits:
        print(f"processing {split}")
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        metafile = os.path.join(input_dir, f"voxpopuli_asr_en_{split}_ner.tsv")
        shutil.copy(metafile, os.path.join(output_dir, f"{split}.tsv"))
        df = pd.read_csv(metafile, sep="\t")
        for id in df["id"].array:
            filename = os.path.join(
                input_dir, f"voxpopuli_asr_en/transcribed_data/en/{id[:4]}/{id}.ogg"
            )
            shutil.copy(filename, split_dir)


def create_manifest(
    data_dir="datasets/slue-voxpopuli",
    manifest_dir="manifest/slue-voxpopuli",
    is_blind=True,
):
    os.makedirs(manifest_dir, exist_ok=True)
    for split in splits:
        if (split == "test") and is_blind:
            df = pd.read_csv(
                os.path.join(data_dir, f"slue-voxpopuli_{split}_blind.tsv"), sep="\t"
            )
        else:
            df = pd.read_csv(
                os.path.join(data_dir, f"slue-voxpopuli_{split}.tsv"), sep="\t"
            )

        with open(os.path.join(manifest_dir, f"{split}.tsv"), "w") as f:
            print(os.path.abspath(os.path.join(data_dir, split)), file=f)
            for uid in df["id"].array:
                frames = soundfile.info(
                    os.path.join(data_dir, split, f"{uid}.ogg")
                ).frames
                print(f"{uid}.ogg\t{frames}", file=f)

        if not (split == "test") and is_blind:
            with open(os.path.join(manifest_dir, f"{split}.wrd"), "w") as f:
                for text in df["normalized_text"].array:
                    text = re.sub(r"[\.;?!]", "", text)
                    text = re.sub(r"\s+", " ", text)
                    print(text, file=f)

            with open(os.path.join(manifest_dir, f"{split}.ltr"), "w") as f:
                for text in df["normalized_text"].array:
                    text = re.sub(r"[\.;?!]", "", text)
                    text = re.sub(r"\s+", " ", text)
                    print(" ".join(text.replace(" ", "|")), file=f)

            # prepare NER files (for Fairseq and HugginFace)
            for sub_dir_name in ["e2e_ner", "nlp_ner"]:
                os.makedirs(os.path.join(manifest_dir, sub_dir_name), exist_ok=True)
            for label_type in ["raw", "combined"]:
                wrd_fn = os.path.join(
                    manifest_dir, "e2e_ner", f"{split}_{label_type}.wrd"
                )
                ltr_fn = os.path.join(
                    manifest_dir, "e2e_ner", f"{split}_{label_type}.ltr"
                )
                tsv_fn = os.path.join(
                    manifest_dir, "nlp_ner", f"{split}_{label_type}.tsv"
                )
                with open(wrd_fn, "w") as f_wrd, open(ltr_fn, "w") as f_ltr, open(
                    tsv_fn, "w"
                ) as f_tsv:
                    for data_sample in df.iterrows():
                        entity_pair_str = data_utils.prep_text_ner_tsv(
                            data_sample[1].normalized_text,
                            data_sample[1].normalized_ner,
                            label_type,
                        )
                        print(entity_pair_str, file=f_tsv, end="")
                        wrd_str, ltr_str = data_utils.prep_e2e_ner_files(
                            entity_pair_str, label_type
                        )
                        print(wrd_str, file=f_wrd)
                        print(ltr_str, file=f_ltr)


if __name__ == "__main__":
    fire.Fire()
