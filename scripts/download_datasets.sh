#!/bin/bash

for split in voxceleb voxpopuli; do

if [ -d data/slue-${split} ]; then
    echo "data/slue-${split} exists. Skip download & extract."
else
    #1. Download
    zip_file="data/slue-${split}_v0.2_blind.zip"
    if [ -f $zip_file ]; then
        echo "$zip_file exists. Skip download."
    else
        zip_file_url="https://public-dataset-model-store.awsdev.asapp.com/users/sshon/public/slue/slue-${split}_v0.2_blind.zip"
        wget $zip_file_url -P data/
    fi

    #2. Extract
    unzip $zip_file -d data/
fi

#3. preprocess

python slue_toolkit/prepare/prepare_${split}.py create_manifest

#4. create dict
for label in ltr wrd; do
    python slue_toolkit/prepare/create_dict.py manifest/slue-${split}/fine-tune.${label} manifest/slue-${split}/dict.${label}.txt
done

done
