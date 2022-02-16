import argparse, os
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation result for sentiment analysis task",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        default="manifest/slue-voxceleb",
        help="Root directory containing voxceleb1_slue data files,"
        "This dir should contain audio/ voxceleb1_slue_{finetune,dev,test} folders ",
    )
    parser.add_argument(
        "--pred-data",
        type=str,
        required=True,
        default="dataset/slue-voxceleb/preds/vc1/w2v2-large-lv60k-ft-slue-vc1-12h-lr1e-5-s1-mt800000-8gpu-update280000",
        help="Root directory containing voxceleb1_slue data files,"
        "This dir should contain audio/ voxceleb1_slue_{finetune,dev,test} folders ",
    )
    args, _ = parser.parse_known_args()

    for subset in ["dev", "test"]:
        pred_csv = os.path.join(args.pred_data, f"{subset}.asr-pred.tsv")
        data = pd.read_csv(pred_csv, delimiter="\t")
        manifest_tsv = os.path.join(args.data, subset) + ".tsv"
        output_tsv = os.path.join(args.data, subset) + ".pred.wrd"

        try:
            fid = open(output_tsv, "w")
            for line in open(manifest_tsv).readlines()[1:]:
                fileid, _ = line.strip().split("\t")
                fileid = (
                    f"{fileid.split('.flac')[0]}-1.flac"  # temp. need to delete future
                )
                fid.write(f"{list(data.pred_text[data.filename==fileid])[0]}\n")
            fid.close()
            print(f"Successfully generated file at {output_tsv}")

        except:
            print(f"something wrong when generating {output_tsv}")
            return


if __name__ == "__main__":
    main()
