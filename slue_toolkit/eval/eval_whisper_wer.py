import argparse, json, os
import pandas as pd
from tqdm import tqdm
import editdistance
from whisper.normalizers import EnglishTextNormalizer

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
        "--modelname",
        type=str,
        default='',
    )

    args,_ = parser.parse_known_args()
    normalizer = EnglishTextNormalizer()


    gt_texts = [ normalizer(t) for t in pd.read_csv(f"{args.manifest_dir}/{args.split}.wrd",header=None)[0].tolist() ]
    # hyp_texts = [ normalizer(t) for t in pd.read_csv(f"{args.manifest_dir}/{args.split}.{args.modelname}.wrd",header=None)[0].tolist() ]
    hyp_texts = [ normalizer(t) for t in open(f"{args.manifest_dir}/{args.split}.{args.modelname}.wrd").readlines() ]
    err = 0
    total_len = 0
    for gt_text,pred_text in zip(gt_texts,hyp_texts):
        err += editdistance.eval(pred_text, gt_text)
        total_len += len(gt_text)
    wer = err * 100.0 / total_len
    json_dict={}
    json_dict["wer"] = wer        
    
    output_json = os.path.join(args.save_dir, f"pred-{args.split}.{args.modelname}.wer.json")

    with open(output_json, "w") as fp:
        json.dump(json_dict, fp, sort_keys=True, indent=4)

if __name__ == "__main__":
    main()
