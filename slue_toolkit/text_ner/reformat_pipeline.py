import fire
import glob
import os
import sys

from slue_toolkit.generic_utils import read_lst, write_to_file


def prep_data(
    model_type, asr_data_dir, asr_model_dir, out_data_dir, eval_set, lm="nolm"
):
    """
    Create tsv files for pipeline evaluation from the decoded ASR transcripts
    """
    if "nolm" not in lm:
        lm = "t3-b500-lw2-ws-1"
    manifest_data_fn = os.path.join(asr_data_dir, eval_set + ".wrd")
    decoded_data_dir = os.path.join(asr_model_dir, "decode", lm)

    out_fn = f"{eval_set}-{model_type}-asr-{lm}.tsv"
    out_fn = os.path.join(out_data_dir, out_fn)
    sent_lst = get_correct_order(decoded_data_dir, manifest_data_fn, eval_set)

    # Space separating trailing "'s" in accordance with the slue voxpopuli NER post processing step.
    # This avoids over-penalizing the model just because the LM used for ASR decoding might not
    # be trained on the text that is similarly post-processed.
    sent_lst = [line.replace("'s", " 's").replace("  ", " ") for line in sent_lst]

    out_str = ""
    for sent in sent_lst:
        for wrd in sent.split(" "):
            out_str += wrd + "\tO\n"
        out_str += "\n"
    write_to_file(out_str, out_fn)
    print("Data prepared for model %s and lm %s" % (model_type, lm))


def get_correct_order(decoded_data_dir, manifest_data_fn, eval_set):
    """
    Reorder decoded sentenced to match the original order
    """
    if not os.path.exists(decoded_data_dir):
        print("Decoded data %s not found" % (decoded_data_dir))
        sys.exit()
    else:
        decoded_sent_lst_gt = read_lst(
            os.path.join(
                decoded_data_dir, f"ref.word-checkpoint_best.pt-{eval_set}.txt"
            )
        )

        decoded_sent_lst_hyp = read_lst(
            os.path.join(
                decoded_data_dir, f"hypo.word-checkpoint_best.pt-{eval_set}.txt"
            )
        )

        manifest_sent_lst = read_lst(manifest_data_fn)

        assert len(decoded_sent_lst_gt) == len(manifest_sent_lst)
        assert len(decoded_sent_lst_hyp) == len(decoded_sent_lst_gt)

        decoded_sent_lst_hyp_select = [
            line.split(" (None-")[0] for line in decoded_sent_lst_hyp
        ]
        decoded_sent_lst_gt = [
            line.split(" (None-")[0] for idx, line in enumerate(decoded_sent_lst_gt)
        ]
        decoded_sent_lst_reordered = [None] * len(manifest_sent_lst)
        for idx, line in enumerate(decoded_sent_lst_gt):
            assert line != -1
            idx_new = manifest_sent_lst.index(line)
            manifest_sent_lst[idx_new] = -1  # to ensure that it's not chosen again
            decoded_sent_lst_reordered[idx_new] = decoded_sent_lst_hyp_select[idx]
        return decoded_sent_lst_reordered


if __name__ == "__main__":
    fire.Fire()
