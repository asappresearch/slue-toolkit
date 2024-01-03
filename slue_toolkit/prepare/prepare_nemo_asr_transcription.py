import os
import pandas as pd
import nemo.collections.asr as nemo_asr
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

    # manifest_dir = 'manifest/slue-hvb'
    # split = 'dev'
    df = pd.read_csv(f"{args.manifest_dir}/{args.split}.tsv",sep='\t')
    wavs = [f"{os.path.join(df.columns[0],w)}" for w in df.index]

    # available_model_list = [model.pretrained_model_name for model in nemo_asr.models.ASRModel.list_available_models() if "stt_en" in model.pretrained_model_name ]
    # available_model_list = ['QuartzNet15x5Base-En']+ available_model_list

    # for modelname in available_model_list:
    modelname= args.modelname
    logging.info(f"decoding wavs using {modelname}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=modelname)
    # texts = model.transcribe(paths2audio_files=wavs,batch_size=1,logprobs=False)
    texts = model.transcribe(paths2audio_files=wavs,batch_size=1)
    huggingface_df = pd.read_csv(f"{args.manifest_dir}/{args.split}.huggingface.csv")
    try:
        huggingface_df['sentence'] = texts
    except:
        huggingface_df['sentence'] = texts[0]
    # huggingface_df.to_csv(f"{args.manifest_dir}/{args.split}.huggingface.nemo_{modelname}.csv",index=None)
    
    fid = open(f"{args.manifest_dir}/{args.split}.nemo_{modelname}.wrd",'w')
    for line in list(huggingface_df['sentence']):
        if pd.isna(line):
            fid.write(f"\n")
        else:
            fid.write(f"{line}\n")
    fid.close()
    
    
if __name__ == "__main__":
    main()