import os
import fire
import numpy as np
import pandas as pd
import soundfile

from multiprocessing import Pool
from tqdm.auto import tqdm

# split2kaldi = {
#     "fine-tune": "fine-tune",
#     "dev": "dev",
#     "test": "test",
# }

splits = {"fine-tune", "dev", "test"}


def read_utt2xxx(filename):
    with open(filename) as f:
        return dict(line.strip().split(" ", 1) for line in f)


def read_segments(filename):
    with open(filename) as f:
        lines = [line.strip().split() for line in f]
    return {line[0]: (line[1], float(line[2]), float(line[3])) for line in lines}


def segment_file(args):
    input_file, output_file, start, end = args
    data, sr = soundfile.read(input_file)
    start, end = int(start * sr), int(end * sr)
    soundfile.write(output_file, data[start:end], sr)


def segment_files(data, output_dir):
    inputs = [
        (
            os.path.join(output_dir + "_raw/", d["id"] + ".flac"),
            os.path.join(output_dir, d["id"] + ".flac"),
            d["start_second"],
            d["end_second"],
        )
        for d in data
    ]
    with Pool() as p:
        list(
            tqdm(
                p.imap(segment_file, inputs),
                total=len(inputs),
                desc=f"creating segmented audios in {output_dir}",
            )
        )


def create_split(
    input_dir="slue/slue-voxceleb1-kaldi", output_dir="slue/slue-voxceleb1"
):
    os.makedirs(output_dir, exist_ok=True)
    for split in split2kaldi:
        print(f"processing {split}")
        kaldi_split = split2kaldi[split]
        kaldi_split_dir = os.path.join(input_dir, f"voxceleb1_slue_{kaldi_split}")

        utt2text = read_utt2xxx(os.path.join(kaldi_split_dir, "text"))
        utt2sentiment = read_utt2xxx(os.path.join(kaldi_split_dir, "utt2sentiment"))
        utt2spk = read_utt2xxx(os.path.join(kaldi_split_dir, "utt2spk"))
        utt2dur = read_utt2xxx(os.path.join(kaldi_split_dir, "utt2dur"))
        utt2seg = read_segments(os.path.join(kaldi_split_dir, "segments"))
        id2file = read_utt2xxx(os.path.join(kaldi_split_dir, "wav.scp"))
        id2file = {k: v.replace("data", input_dir) for k, v in id2file.items()}

        assert (
            len(utt2dur)
            == len(utt2sentiment)
            == len(utt2spk)
            == len(utt2dur)
            == len(utt2seg)
        )

        for key in utt2sentiment:
            if utt2sentiment[key] == "<mixed>" or utt2sentiment[key] == "Disagreement":
                utt2sentiment[key] = "Neutral"

        data = []
        for uid in utt2seg:
            example = {
                "uid": uid,
                "id": utt2seg[uid][0],
                "start": utt2seg[uid][1],
                "end": utt2seg[uid][2],
                "speaker_id": utt2spk[uid],
                "text": utt2text[uid],
                "sentiment": utt2sentiment[uid],
                "split": split,
            }
            data.append(example)

        df = pd.DataFrame.from_dict(data)
        df.to_csv(os.path.join(output_dir, f"{split}.tsv"), sep="\t")

        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        segment_files(data, id2file, split_dir)


def create_manifest(
    data_dir="datasets/slue-voxceleb",
    manifest_dir="manifest/slue-voxceleb",
    is_blind=True,
):
    os.makedirs(manifest_dir, exist_ok=True)

    labels = sorted(["Negative", "Neutral", "Positive"])
    with open(os.path.join(manifest_dir, f"labels.sent.txt"), "w") as f:
        for l in labels:
            print(l, file=f)

    for split in splits:
        target_sentiments = ["Negative", "Neutral", "Positive"]
        if (split == "test") and is_blind:
            df = pd.read_csv(
                os.path.join(data_dir, f"slue-voxceleb_{split}_blind.tsv"), sep="\t"
            )
        else:
            df = pd.read_csv(
                os.path.join(data_dir, f"slue-voxceleb_{split}.tsv"), sep="\t"
            )
            df = df[df["sentiment"].isin(target_sentiments)]

        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        segment_files(df.to_dict("records"), split_dir)

        with open(os.path.join(manifest_dir, f"{split}.tsv"), "w") as f:
            print(os.path.abspath(os.path.join(data_dir, split)), file=f)
            for id, start, end in zip(
                df["id"].array, df["start_second"].array, df["end_second"].array
            ):
                frames = int(16000 * (end - start))
                print(f"{id}.flac\t{frames}", file=f)

        if not (split == "test") and is_blind:
            with open(os.path.join(manifest_dir, f"{split}.wrd"), "w") as f:
                for text in df["normalized_text"].array:
                    print(text, file=f)
            with open(os.path.join(manifest_dir, f"{split}.ltr"), "w") as f:
                for text in df["normalized_text"].array:
                    print(" ".join(text.replace(" ", "|")), file=f)
            with open(os.path.join(manifest_dir, f"{split}.sent"), "w") as f:
                for text in df["sentiment"].array:
                    print(text, file=f)


if __name__ == "__main__":
    fire.Fire()
