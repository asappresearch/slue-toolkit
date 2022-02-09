#!/bin/bash

#1. Download
wget https://papers-slue.awsdev.asapp.com/slue-voxceleb_blind.tar.gz -P dataset/
wget https://papers-slue.awsdev.asapp.com/slue-voxpopuli_blind.tar.gz -P dataset/

#2. Extract
tar -xzvf dataset/slue-voxceleb_blind.tar.gz -C dataset/
tar -xzvf dataset/slue-voxpopuli_blind.tar.gz -C dataset/

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
for session in dev fine-tune; do
for label in raw combined; do
    cp ./manifest/slue-voxpopuli/${session}.tsv ./manifest/slue-voxpopuli/e2e_ner/${session}_${label}.tsv
done
done
