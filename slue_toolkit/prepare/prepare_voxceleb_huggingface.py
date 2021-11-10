import argparse
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
    args = parser.parse_args()
    print(args)

    for subset in ["fine-tune", "dev", "test"]:
        data = {}
        data["sentence"] = []
        data["label"] = []
        for line in open(f"{args.data}/{subset}.wrd").readlines():
            data["sentence"].append(line.strip())
        for line in open(f"{args.data}/{subset}.sent").readlines():
            data["label"].append(line.strip())

        df = pd.DataFrame(data=data)
        output_filename = f"{args.data}/{subset}.huggingface.csv"
        try:
            df.to_csv(output_filename, index=False)
            print(f"Successfully generated file at {output_filename}")

        except:
            print(f"something wrong when generating {output_filename}")
            return


if __name__ == "__main__":
    main()
