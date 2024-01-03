import os
import fire
import numpy as np
import pandas as pd
import soundfile, librosa

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


def segment_resample_file(args):
    input_file, output_file, start, end = args
    data, sr = soundfile.read(input_file)
    start, end = int(start * sr), int(end * sr)
    try:
        data_16k = librosa.resample(data[start:end], orig_sr=sr, target_sr=16000)
        soundfile.write(output_file, data_16k, 16000)
        
    except:
        print(input_file, start,end)
        # assert
    


def segment_files(data, output_dir):
    inputs = [
        (
            # os.path.join(f'data/hvb/data/audio/{"caller" if d["role"] == "caller" else "agent"}', d["issue_id"] + ".wav"),
            os.path.join(f'data/hvb/data/audio/{"caller" if d["channel"] == 1 else "agent"}', d["issue_id"] + ".wav"),
            os.path.join(output_dir, f'{d["issue_id"]}_{d["start_ms"]}_{d["start_ms"]+d["duration_ms"]}.wav'),
            d["start_ms"]/1000.0,
            (d["start_ms"]+d["duration_ms"])/1000.0,
        )
        for d in data
    ]    
    
    with Pool() as p:
        list(
            tqdm(
                p.imap(segment_resample_file, inputs),
                total=len(inputs),
                desc=f"creating segmented audios in {output_dir}",
            )
        )

def create_manifest(
    data_dir="data/slue-hvb",
    manifest_dir="manifest/slue-hvb",
    is_blind=False,
):
    os.makedirs(manifest_dir, exist_ok=True)

    labels = sorted(['acknowledge', 'answer_agree', 'answer_dis', 'answer_general', 'apology',
 'backchannel', 'disfluency', 'other', 'question_check', 'question_general',
 'question_repeat', 'self', 'statement_close', 'statement_general',
 'statement_instruct', 'statement_open', 'statement_problem', 'thanks'])
    
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
            df = pd.read_csv(
                os.path.join('data', f"harpervalleybank_dialog_acts_20221108_asapp_annotated_ver3.csv"), sep=","
            )
            df = df[ df["split"] == split ]

        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        # df = df[ df[ "duration_ms" ] > 300 ]
        
        segment_files(df.to_dict("records"), split_dir)

        with open(os.path.join(manifest_dir, f"{split}.tsv"), "w") as f:
            print(os.path.abspath(os.path.join(data_dir, split)), file=f)
            for id, start, dur in zip(
                df["issue_id"].array, df["start_ms"].array, df["duration_ms"].array
            ):
                end = start+dur
                frames = int(8000 * (end - start)/1000.0)
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
                for text in df["asapp_dialog_acts"].array:
                    print(','.join([t for t in eval(text)]), file=f)


if __name__ == "__main__":
    fire.Fire()
