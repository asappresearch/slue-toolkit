# Copyright (c) ASAPP Inc.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import fire
import shlex
import subprocess

lm_dict = {
    "vc": "lower-t3/3",
    "vp": "lower-t3/3",
}

data_dict = {
    "vc": "manifest/slue-voxceleb",
    "vp": "manifest/slue-voxpopuli",
}


def eval_asr(
    model="save/asr/w2v2-base-vc",
    lm="nolm-argmax",
    beam_size=500,
    lm_weight=2.0,
    word_score=-1.0,
    subset="dev",
    data="manifest/slue-voxceleb",
    upsample=1.0,
    save_results=True,
    dump_emissions=False,
    csv_log_file="exp-eval-logs.csv",
    fp16=False,
    batch_size=-1,
    quiet=False,
    use_bpe=False,
    user_dir="slue_toolkit/fairseq_addon",
    dry_run=False,
    max_tokens=4000000,
):
    if data in data_dict:
        data = data_dict[data]

    if os.path.isdir(model):
        ckpt = os.path.join(model, "checkpoints/checkpoint_best.pt")
        # eval_log_file = os.path.join(ckpt, 'eval.log')
        eval_log_file = None
        if save_results:
            results_path = os.path.join(model, "decode")
        else:
            results_path = None
        emission_path = (
            os.path.join(model, "decode", subset, "emissions.npy")
            if dump_emissions
            else None
        )
    else:
        raise NotImplementedError(f"model={model} is not a folder")

    if not quiet:
        print(f"ckpt: {ckpt}")
        print(f"lm: {lm}")
        if "nolm" not in lm:
            print(
                f"lm_weight: {lm_weight} word_score: {word_score} beam_size: {beam_size}"
            )

    user_dir = os.path.abspath(user_dir)

    cmd = (
        f"python -m slue_toolkit.eval.infer_asr {data}"
        f" --user-dir {user_dir}"
        f" --task audio_finetuning"
        f" --nbest 1 --path {ckpt} --gen-subset {subset}"
        f" --sil-weight 0 --max-tokens {max_tokens}"
        f" --lm-weight {lm_weight} --word-score {word_score}"
        f" --criterion ctc"
        f" --beam {beam_size}"
        f" --eval-upsample {upsample}"
    )

    if results_path is not None:
        cmd += f" --results-path {results_path}"
    if emission_path is not None:
        cmd += f" --dump-emissions {emission_path}"
    if "bpe" in ckpt or use_bpe:
        cmd += " --labels bpe --post-process sentencepiece"
    else:
        cmd += " --labels ltr --post-process letter"
    if batch_size > 0:
        cmd += f" --batch-size {batch_size}"

    if lm == "nolm":
        cmd += " --w2l-decoder viterbi"
    elif lm == "nolm-argmax":
        cmd += " --w2l-decoder argmax"
    elif "/" in lm:
        cmd += f" --w2l-decoder kenlm --lm-model save/kenlm/{lm}gram.bin --lexicon save/kenlm/{lm.split('/')[0]}/lexicon.lst"
    else:
        cmd += f" --w2l-decoder kenlm --lm-model save/kenlm/{lm}/4gram.bin --lexicon save/kenlm/{lm}/lexicon.lst"

    if fp16:
        cmd += " --fp16"

    if "vox" in ckpt:
        cmd += " --normalize"

    if eval_log_file is not None:
        os.makedirs(os.path.dirname(eval_log_file), exist_ok=True)
        cmd += f" | tee -a {eval_log_file}"

    if not quiet:
        print("cmd:")
        print(cmd)
    if dry_run:
        return cmd, results_path
    result = subprocess.run(
        shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    wer, time_used, model_size, extract_size = parse_result(result, quiet=quiet)

    if not quiet:
        print(
            f"WER: {wer} time_used: {time_used} model_size: {model_size} extract_size: {extract_size}"
        )
    msg = f"{subset},{model},{lm},{model_size},{extract_size},{time_used},{wer}"
    if not quiet:
        print(msg)
    if csv_log_file is not None:
        with open(csv_log_file, "a") as f:
            print(msg, file=f)

    # convert to predictions
    if results_path is not None:
        preds = []
        with open(
            os.path.join(results_path, "hypo.word-checkpoint_best.pt-dev.txt")
        ) as f:
            for line in f:
                line = line.strip().rsplit(" ", 1)
                if len(line) == 1:
                    line = [""] + line
                idx = int(line[1][len("(None-") : -1])
                preds.append((idx, line[0]))
        preds.sort()
        with open(os.path.join(model, f"pred-{subset}.wrd"), "w") as f:
            for idx, pred in preds:
                print(pred, file=f)

    return wer, time_used, model_size, extract_size, results_path


def parse_result(result, quiet=False):
    extract_size = 0
    model_size = 0
    wer = -1
    time_used = -1
    for line in result.stderr.decode("utf-8").split("\n"):
        if not quiet:
            print(line)
        pos = line.find("WER: ")
        if pos >= 0:
            wer = float(line[pos + 5 :].rstrip())

        pos = line.find("time used: ")
        if pos >= 0:
            time_used = float(line[pos + 11 :].rstrip())

        query = "model 0 size: "
        pos = line.find(query)
        if pos >= 0:
            model_size = int(line[pos + len(query) :].rstrip())

        query = "w2v_encoder.w2v_model.feature_extractor size: "
        pos = line.find(query)
        if pos >= 0:
            extract_size += int(line[pos + len(query) :].rstrip())

        query = "w2v_encoder.w2v_model.spec_feature_extractor size: "
        pos = line.find(query)
        if pos >= 0:
            extract_size += int(line[pos + len(query) :].rstrip())

    return wer, time_used, model_size, extract_size


def dump_asr_preds(
    model="save/asr/w2v2-base-vc",
    task="vc",
    splits=["fine-tune", "dev", "test"],
    lm_model="",
    beam_size=500,
    lm_weight=2.0,
    word_score=-1.0,
    use_lm=False,
):
    if use_lm:
        dump_folder = os.path.join("preds", task + "+lm", os.path.basename(model))
    else:
        dump_folder = os.path.join("preds", task, os.path.basename(model))
    os.makedirs(dump_folder, exist_ok=True)

    if not use_lm:
        lm_model = "nolm-argmax"
    elif lm_model == "":
        lm_model = lm_dict[task]

    for split in splits:
        dump_file = os.path.join(dump_folder, f"{split}.pred.tsv")
        if os.path.exists(dump_file):
            print(f"{dump_file} exists")
            continue
        cmd, results_path = eval_asr(
            model=model,
            lm=lm_model,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
            subset=split,
            data=data_dict[task],
            save_results=True,
            csv_log_file="dump-pred.csv",
            fp16=True,
            dry_run=True,
        )
        pred_file = os.path.join(
            results_path, f"hypo.word-checkpoint_best.pt-{split}.txt"
        )
        if not os.path.exists(pred_file):
            os.system(cmd)
        hyps = []
        with open(pred_file) as f:
            for line in f:
                line = line.strip().rsplit(" ", 1)
                if len(line) == 1:
                    line = [""] + line
                idx = int(line[1][len("(None-") : -1])
                hyps.append((line[0], idx))
        hyp_dict = {idx: txt for txt, idx in hyps}

        with open(os.path.join(data_dict[task], f"{split}.tsv")) as f:
            f.readline()
            filenames = [line.strip().split()[0] for line in f]

        # assert len(filenames) == len(hyps), f'{len(filenames)}, {len(hyps)}'
        with open(dump_file, "w") as f:
            print("filename\tpred_text", file=f)
            for i, filename in enumerate(filenames):
                if i in hyp_dict:
                    print(filename, hyp_dict[i], sep="\t", file=f)
                else:
                    print(f"missing {i} {filename}")


if __name__ == "__main__":
    fire.Fire()
