#!/bin/bash

mkdir -p log

for split in dev test; do
for modelname in base.en tiny.en small.en medium.en ; do
    python slue_toolkit/prepare/prepare_whisper_asr_transcription.py \
        --manifest-dir manifest/slue-hvb \
        --split ${split} \
        --modelname ${modelname} &
done
done
