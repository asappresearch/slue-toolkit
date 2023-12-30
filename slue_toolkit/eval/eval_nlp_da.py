import argparse, json, os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import cuda
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
import editdistance

def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation result for sentiment analysis task",
    )
    parser.add_argument(
        "--manifest-dir", type=str, required=True, help="manifest dir for data loading",
        default="manifest/slue-hvb",
    )
    parser.add_argument(
        "--split", type=str, required=True, help="split name (test, dev, finetune",
        default="dev",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
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
    parser.add_argument(
        "--modelname",
        type=str,
        default='',
    )

    args,_ = parser.parse_known_args()

    # check gpu availability
    if args.use_gpu:
        device = "cuda:0" if cuda.is_available() else "cpu"
    else:
        device = "cpu"

    # model loading
    config = AutoConfig.from_pretrained(args.save_dir)
    # config = AutoConfig.from_pretrained(args.save_dir, num_labels=18)
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.save_dir, config=config
    ).to(device)

    id2label = {i: l for i, l in enumerate(config.label2id)}
    label_list = np.array([key for key in config.label2id])

    texts = []
    gt = []
    if args.modelname:
        input_file = f"{args.manifest_dir}/{args.split}.{args.modelname}.wrd"
    else:
        input_file = f"{args.manifest_dir}/{args.split}.wrd"
    print(input_file)

    for line in open(input_file).readlines():
        texts.append(line.strip())

    gt = []
    for line in open(f"{args.manifest_dir}/{args.split}.da").readlines():
        # label = line.strip().split(',')
        label = np.zeros(len(label_list))
        for l in line.strip().split(','):
            label[config.label2id[l]]=1
        gt.append(label)
    gt = np.array(gt)

    # feeding input to model
    scores = []
    preds=[]
    for line in tqdm(texts):
        inputs = tokenizer(line.strip(), return_tensors="pt").to(device)
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits.squeeze()).detach().cpu().numpy()
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1.0    
        preds.append(predictions)
        scores.append(probs)

    if args.modelname:
        output_tsv = os.path.join(args.save_dir, f"pred-{args.split}.{args.modelname}.da")
    else:
        output_tsv = os.path.join(args.save_dir, f"pred-{args.split}.da")

    fid = open(output_tsv, "w")
    for pred in preds:
        fid.write(f"{','.join(label_list[pred==1])}\n")
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
        
        gt_texts = pd.read_csv(f"{args.manifest_dir}/{args.split}.wrd",header=None)[0].tolist()    
        err = 0
        total_len = 0
        for gt_text,pred_text in zip(gt_texts,texts):
            err += editdistance.eval(pred_text, gt_text)
            total_len += len(gt_text)
        wer = err * 100.0 / total_len
        json_dict["wer"] = wer        

        with open(output_json, "w") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()
