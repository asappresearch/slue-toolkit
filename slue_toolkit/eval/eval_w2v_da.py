import numpy as np
import torch, argparse, json, os
from tqdm import tqdm
from torch import cuda
from fairseq.dataclass import FairseqDataclass
from slue_toolkit.fairseq_addon.models.wav2vec2_cls import Wav2Vec2SeqCls
from slue_toolkit.fairseq_addon.tasks import audio_classification
from sklearn.metrics import f1_score, precision_score, recall_score


def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation result for sentiment analysis task",
    )
    parser.add_argument(
        "--manifest-dir", type=str, required=True, help="manifest dir for data loading",
    )
    parser.add_argument(
        "--split", type=str, required=True, help="split name (test, dev, finetune",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="save dir containing checkpoints folder",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        required=False,
        default="checkpoint_best.pt",
        help=".pt file you want to use",
    )
    parser.add_argument(
        "--use-gpu",
        default=True,
        action="store_true",
        help="use gpu if available default: False",
    )
    parser.add_argument(
        "--eval",
        default=True,
        action="store_true",
        help="eval after inference. save in the same filename with output but .json format. default: True",
    )
    args,_ = parser.parse_known_args()

    if args.use_gpu:
        device = "cuda:0" if cuda.is_available() else "cpu"
    else:
        device = "cpu"
    checkpoint_dir = os.path.join(args.save_dir, "checkpoints")
    checkpoint = Wav2Vec2SeqCls.from_pretrained(
        checkpoint_dir, checkpoint_file=args.checkpoint_file
    )

    checkpoint.task.cfg.data = args.manifest_dir
    checkpoint.task.load_dataset(args.split)
    checkpoint.task.load_label2id
    checkpoint.to(device)
    data = checkpoint.task.datasets[args.split]
    model = checkpoint.models[0]
    model.eval()
    preds = []
    gt = []

    with torch.no_grad():
        for iter in tqdm(range(len(data))):
            input = data.__getitem__(iter)
            output = model(
                source=input["source"].unsqueeze(0).to(device), padding_mask=None
            )
            probs = torch.sigmoid(output['pooled'].squeeze()).detach().cpu().numpy()
            predictions = np.zeros(probs.shape)
            predictions[np.where(probs >= 0.5)] = 1.0    
            preds.append(predictions)
            gt.append(input["label"])

    id2label = {i: l for i, l in enumerate(checkpoint.task.label2id)}
    label_list = np.array([key for key in checkpoint.task.label2id])

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
            id2label[idx]: score
            for idx, score in enumerate(f1_score(gt, preds, average=None) * 100)
        }

        with open(output_json, "w") as fp:
            json.dump(json_dict, fp, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
