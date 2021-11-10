#!/bin/bash

python3 slue_toolkit/eval/eval_w2v_sentiment.py \
        --data manifest/slue-voxceleb \
        --subset test \
        --save-dir save/sentiment/w2v2-base \
        --use-gpu
        --eval