#!/bin/bash

python slue_toolkit/eval/eval_nlp_da.py \
       --manifest-dir manifest/slue-hvb \
       --split test \
       --save-dir save/da/nlp_topline_deberta-base \
       --use-gpu \
       --eval \
       --modelname w2v2-base-hvb &
