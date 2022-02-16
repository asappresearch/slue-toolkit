#!/bin/bash

python3 slue_toolkit/prepare/prepare_voxceleb_asr_pred.py --data manifest/slue-voxceleb --pred-data dataset/slue-voxceleb/preds/vc1/w2v2-large-lv60k-ft-slue-vc1-12h-lr1e-5-s1-mt800000-8gpu-update280000

python3 slue_toolkit/eval/eval_nlp_sentiment.py \
        --data manifest/slue-voxceleb \
        --subset test.asr-pred \
        --save-dir save/sentiment/nlp_topline_bert-base-cased \
        --use-gpu \
        --eval \
        
