#!/bin/bash

for split in voxceleb voxpopuli; do

if [ -d data/slue-${split} ]; then
    echo "data/slue-${split} exists. Skip download & extract."
else
    #1. Download
    tar_file="data/slue-${split}_blind.tar.gz"
    if [ -f $tar_file ]; then
        echo "$tar_file exists. Skip download."
    else
        tar_file_url="https://papers-slue.awsdev.asapp.com/slue-${split}_blind.tar.gz"
        wget $tar_file_url -P data/
    fi

    #2. Extract
    tar -xzvf $tar_file -C data/
fi

#3. preprocess

python slue_toolkit/prepare/prepare_${split}.py create_manifest

#4. create dict
for label in ltr wrd; do
    python slue_toolkit/prepare/create_dict.py manifest/slue-${split}/fine-tune.${label} manifest/slue-${split}/dict.${label}.txt
done

done
