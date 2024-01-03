import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation result for sentiment analysis task",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=False,
        default="manifest/slue-hvb",
        help="Root directory containing voxceleb1_slue data files,"
        "This dir should contain audio/ voxceleb1_slue_{finetune,dev,test} folders ",
    )
    args,_ = parser.parse_known_args()
    print(args)

    label_list = sorted(['acknowledge', 'answer_agree', 'answer_dis', 'answer_general', 'apology',
    'backchannel', 'disfluency', 'other', 'question_check', 'question_general',
    'question_repeat', 'self', 'statement_close', 'statement_general',
    'statement_instruct', 'statement_open', 'statement_problem', 'thanks'])


    for subset in ["fine-tune", "dev", "test"]:
        data = {}
        data["sentence"] = []
        # data["label"] = []
        for line in open(f"{args.data}/{subset}.wrd").readlines():
            data["sentence"].append(line.strip())
        for line in open(f"{args.data}/{subset}.da").readlines():
            labels = line.strip().split(',')
            for label in label_list:
                if label not in data:
                    data[label] = []
                if label in labels:
                    data[label].append(1)
                else:
                    data[label].append(0)

        df = pd.DataFrame(data=data)
        df['label'] = df[df.columns[1:]].values.tolist()
        for l in label_list:
            df.pop(l)

        output_filename = f"{args.data}/{subset}.huggingface.csv"
        try:
            df.to_csv(output_filename, index=False)
            print(f"Successfully generated file at {output_filename}")

        except:
            print(f"something wrong when generating {output_filename}")
            # return

if __name__ == "__main__":
    main()
