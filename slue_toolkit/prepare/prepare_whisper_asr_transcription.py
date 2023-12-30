import os
import pandas as pd
import whisper
import logging
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Get evaluation result for sentiment analysis task",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        required=True,
        default="manifest/slue-hvb",
        help="Root directory containing voxceleb1_slue data files,"
        "This dir should contain audio/ voxceleb1_slue_{finetune,dev,test} folders ",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        default="dev",
        help="Root directory containing voxceleb1_slue data files,"
        "This dir should contain audio/ voxceleb1_slue_{finetune,dev,test} folders ",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        required=True,
        default="dev",
        help="Root directory containing voxceleb1_slue data files,"
        "This dir should contain audio/ voxceleb1_slue_{finetune,dev,test} folders ",
    )
        
    args,_ = parser.parse_known_args()
    print(args)
    
    logging.basicConfig(level=logging.INFO)

    df = pd.read_csv(f"{args.manifest_dir}/{args.split}.tsv",sep='\t')
    wavs = [f"{os.path.join(df.columns[0],w)}" for w in df.index]

    # for modelname in available_model_list:
    modelname= args.modelname
    logging.info(f"decoding wavs using {modelname}")
    model = whisper.load_model(modelname)
    texts = [model.transcribe(wav)['text'].strip() for wav in wavs]
    huggingface_df = pd.read_csv(f"{args.manifest_dir}/{args.split}.huggingface.csv")
    try:
        huggingface_df['sentence'] = texts
    except:
        huggingface_df['sentence'] = texts[0]
    
    fid = open(f"{args.manifest_dir}/{args.split}.whisper_{modelname}.wrd",'w')
    for line in list(huggingface_df['sentence']):
        if pd.isna(line):
            fid.write(f"\n")
        else:
            fid.write(f"{line}\n")
    fid.close()
    
    
if __name__ == "__main__":
    main()