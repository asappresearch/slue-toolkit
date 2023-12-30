#!/bin/bash


for split in hvb; do
  for label in ltr wrd; do
    python slue_toolkit/prepare/create_dict.py manifest/slue-${split}/fine-tune.${label} manifest/slue-${split}/dict.${label}.txt
  done
done


bash baselines/asr/ft-w2v2-base.sh manifest/slue-hvb save/asr/w2v2-base-hvb

