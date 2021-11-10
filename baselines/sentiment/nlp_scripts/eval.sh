#!/bin/bash

python3 slue_toolkit/eval/eval_nlp_sentiment.py \
        --data manifest/slue-voxceleb \
        --subset test \
        --save-dir save/sentiment/nlp_topline_bert-base-cased \
        --use-gpu \
        --eval