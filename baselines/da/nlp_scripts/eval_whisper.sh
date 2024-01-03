#!/bin/bash
    
for split in dev test; do
for modelname in base.en tiny.en small.en medium.en ; do

  python slue_toolkit/eval/eval_nlp_da.py \
    --manifest-dir manifest/slue-hvb \
    --split ${split} \
    --save-dir save/da/nlp_topline_deberta-base \
    --use-gpu \
    --eval \
    --modelname whisper_${modelname} &


  python slue_toolkit/eval/eval_whisper_wer.py \
    --manifest-dir manifest/slue-hvb \
    --split ${split} \
    --save-dir save/da/nlp_topline_deberta-base \
    --modelname whisper_${modelname} &
    
done
done
