import os, fire

import slue_toolkit.text_ner.ner_deberta_modules as NDM
from slue_toolkit.generic_utils import read_lst, save_dct


def train(
    data_dir,
    model_dir,
    model_type,
    cfg_file,
    label_type="raw",
    train_subset="fine-tune",
    valid_subset="dev",
):
    data_obj = NDM.DataSetup(data_dir, model_type)
    _, _, _, _, train_dataset = data_obj.prep_data(train_subset, label_type)
    _, _, _, _, val_dataset = data_obj.prep_data(valid_subset, label_type)
    label_list = read_lst(os.path.join(data_dir, f"{label_type}_tag_lst_ordered"))
    NDM.train_module(
        model_dir, train_dataset, val_dataset, label_list, model_type, cfg_file
    )


def eval(
    data_dir,
    model_dir,
    model_type,
    eval_asr=False,
    eval_subset="dev",
    train_label="raw",
    eval_label="combined",
    lm="nolm",
    asr_model_type="w2v2-base",
    save_results=False,
):
    lm = lm.replace("/", "_")
    log_dir = os.path.join(model_dir, "metrics")
    if save_results:
        ner_results_dir = os.path.join(log_dir, "error_analysis")
        os.makedirs(ner_results_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    data_obj = NDM.DataSetup(data_dir, model_type)

    val_texts, val_tags, _, _, val_dataset = data_obj.prep_data(
        eval_subset, train_label
    )
    if eval_asr:
        asr_val_texts, _, _, _, val_dataset = data_obj.prep_data(
            f"{eval_subset}-{asr_model_type}-asr-{lm}", train_label, eval_asr=True
        )
    else:
        asr_val_texts = None
    eval_obj = NDM.Eval(data_dir, model_dir, train_label, eval_label, eval_asr)
    for score_type in ["standard", "label"]:
        if eval_asr:
            res_fn = "-".join(
                [eval_subset, "pipeline", asr_model_type, lm, eval_label, score_type]
            )
        else:
            res_fn = "-".join([eval_subset, "gt-text", eval_label, score_type])
        metrics_dct, analysis_examples_dct = eval_obj.get_scores(
            score_type, val_dataset, val_texts, eval_subset, val_tags, asr_val_texts
        )
        save_dct(os.path.join(log_dir, res_fn + ".json"), metrics_dct)
        if save_results and score_type == "standard":
            save_dct(
                os.path.join(ner_results_dir, res_fn + ".json"), analysis_examples_dct
            )


if __name__ == "__main__":
    fire.Fire()
