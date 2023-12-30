#!/bin/bash

for modelname in stt_en_conformer_transducer_large \
  stt_en_conformer_transducer_large_ls \
  stt_en_conformer_transducer_medium \
  stt_en_conformer_transducer_small \
  stt_en_conformer_transducer_xlarge \
  stt_en_conformer_transducer_xxlarge \
  stt_en_contextnet_1024 \
  stt_en_contextnet_1024_mls \
  stt_en_contextnet_256 \
  stt_en_contextnet_256_mls \
  stt_en_contextnet_512 \
  stt_en_contextnet_512_mls \
  stt_en_jasper10x5dr \
  stt_en_quartznet15x5 \
  stt_en_squeezeformer_ctc_large_ls \
  stt_en_squeezeformer_ctc_medium_large_ls \
  stt_en_squeezeformer_ctc_medium_ls \
  stt_en_squeezeformer_ctc_small_ls \
  stt_en_squeezeformer_ctc_small_medium_ls \
  stt_en_squeezeformer_ctc_xsmall_ls \
; do

    python slue_toolkit/prepare/prepare_nemo_asr_transcription.py \
        --manifest-dir manifest/slue-hvb \
        --split dev \
        --modelname ${modelname} &

    python slue_toolkit/prepare/prepare_nemo_asr_transcription.py \
        --manifest-dir manifest/slue-hvb \
        --split test \
        --modelname ${modelname} &

done
