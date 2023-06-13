import os, fire
import pandas as pd

splits = {"fine-tune", "dev", "test"}


def read_utt2xxx(filename):
    with open(filename) as f:
        return dict(line.strip().split(" ", 1) for line in f)


def create_manifest(
    data_dir="data/slue-hvb",
    manifest_dir="manifest/slue-hvb",
    is_blind=True,
):
    os.makedirs(manifest_dir, exist_ok=True)

    labels = sorted(
        [
            "acknowledge",
            "answer_agree",
            "answer_dis",
            "answer_general",
            "apology",
            "backchannel",
            "disfluency",
            "other",
            "question_check",
            "question_general",
            "question_repeat",
            "self",
            "statement_close",
            "statement_general",
            "statement_instruct",
            "statement_open",
            "statement_problem",
            "thanks",
        ]
    )

    with open(os.path.join(manifest_dir, f"labels.da.txt"), "w") as f:
        for l in labels:
            print(l, file=f)

    for split in splits:
        print(f"Processing slue-voxceleb {split} set")
        target_labels = labels
        if (split == "test") and is_blind:
            df = pd.read_csv(
                os.path.join(data_dir, f"slue-hvb_{split}_blind.tsv"), sep="\t"
            )
        else:
            df = pd.read_csv(os.path.join(data_dir, f"slue-hvb_{split}.tsv"), sep="\t")

        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        with open(os.path.join(manifest_dir, f"{split}.tsv"), "w") as f:
            print(os.path.abspath(os.path.join(data_dir, split)), file=f)
            for id, start, dur in zip(
                df["issue_id"].array, df["start_ms"].array, df["duration_ms"].array
            ):
                end = start + dur
                frames = int(8000 * (end - start) / 1000.0)
                print(f"{id}_{start}_{start+dur}.wav\t{frames}", file=f)

        if split != "test" or not is_blind:
            with open(os.path.join(manifest_dir, f"{split}.wrd"), "w") as f:
                for text in df["text"].array:
                    print(text, file=f)
            with open(os.path.join(manifest_dir, f"{split}.ltr"), "w") as f:
                for text in df["text"].array:
                    if pd.isna(text):
                        print(" ", file=f)
                    else:
                        print(" ".join(text.replace(" ", "|")), file=f)
            with open(os.path.join(manifest_dir, f"{split}.da"), "w") as f:
                for text in df["dialog_acts"].array:
                    print(",".join([t for t in eval(text)]), file=f)


if __name__ == "__main__":
    fire.Fire()
