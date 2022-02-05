#!/bin/bash

#1. Download
wget https://papers-slue.awsdev.asapp.com/slue-voxceleb_blind.tar.gz -P datasets/
wget https://papers-slue.awsdev.asapp.com/slue-voxpopuli_blind.tar.gz -P datasets/

#2. Extract
tar -xzvf datasets/slue-voxceleb_blind.tar.gz -C datasets/
tar -xzvf datasets/slue-voxpopuli_blind.tar.gz -C datasets/

#3. preprocess

python slue_toolkit/prepare/prepare_voxceleb.py create_manifest
python slue_toolkit/prepare/prepare_voxpopuli.py create_manifest

#4. create dict
for split in voxceleb voxpopuli; do
for label in ltr wrd; do
    python slue_toolkit/prepare/create_dict.py manifest/slue-${split}/fine-tune.${label} manifest/slue-${split}/dict.${label}.txt
done
done

#5. copy files
cp ./manifest/slue-voxpopuli/dev.tsv ./manifest/slue-voxpopuli/e2e_ner/dev_raw.tsv
cp ./manifest/slue-voxpopuli/fine-tune.tsv ./manifest/slue-voxpopuli/e2e_ner/fine-tune_raw.tsv
cp ./manifest/slue-voxpopuli/dict.ltr.txt ./manifest/slue-voxpopuli/e2e_ner/dict.ltr.txt
