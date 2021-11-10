import argparse, json, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from pytorch_lightning import metrics
from torch import cuda
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation result for sentiment analysis task",
    )
    parser.add_argument(
        "--data", type=str, required=True, help="manifest dir for data loading"
    )
    parser.add_argument(
        "--subset", type=str, required=True, help="split name (test, dev, finetune"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=False,
        help="Transformers model output name. ,"
        "Model must be finetuned for sentiment for sequence classification task.",
    )
    parser.add_argument(
        "--use-gpu",
        default=False,
        action="store_true",
        help="use gpu if available default: False",
    )
    parser.add_argument(
        "--eval",
        default=True,
        action="store_true",
        help="eval after inference. save in the same filename with output but .json format. default: True",
    )

    args = parser.parse_args()

    # check gpu availability
    if args.use_gpu:
        device = "cuda:0" if cuda.is_available() else "cpu"
    else:
        device = "cpu"

    # model loading
    config = AutoConfig.from_pretrained(args.save_dir, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.save_dir, config=config
    ).to(device)

    texts = []
    gt = []
    for line in open(f"{args.data}/{args.subset}.wrd").readlines():
        texts.append(line.strip())
    for line in open(f"{args.data}/{args.subset.split('.')[0]}.sent").readlines():
        gt.append(config.label2id[line.strip()])

    # feeding input to model
    scores = []
    for line in tqdm(texts):
        inputs = tokenizer(line.strip(), return_tensors="pt").to(device)
        outputs = model(**inputs)
        score = outputs["logits"][0].detach().cpu().numpy()
        scores.append(score)
    preds = np.argmax(scores, axis=1)

    id2label = {i: l for i, l in enumerate(config.label2id)}

    output_tsv = os.path.join(args.save_dir, f"pred-{args.subset}.sent")
    fid = open(output_tsv, "w")
    for sent_id in preds:
        fid.write(f"{id2label[sent_id]}\n")
    fid.close()

    if args.eval:
        output_json = os.path.splitext(output_tsv)[0] + ".json"
        json_dict = {}
        json_dict["macro"] = {
            "precision": precision_score(gt, preds, average="macro") * 100,
            "recall": recall_score(gt, preds, average="macro") * 100,
            "f1": f1_score(gt, preds, average="macro") * 100,
        }
        json_dict["micro"] = {
            "precision": precision_score(gt, preds, average="weighted") * 100,
            "recall": recall_score(gt, preds, average="weighted") * 100,
            "f1": f1_score(gt, preds, average="weighted") * 100,
        }
        json_dict["per_classes"] = {
            config.id2label[idx]: score
            for idx, score in enumerate(f1_score(gt, preds, average=None) * 100)
        }

        with open(output_json, "w") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
