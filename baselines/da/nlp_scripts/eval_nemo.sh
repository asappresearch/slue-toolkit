#!/bin/bash

for modelname in QuartzNet15x5Base-En \
  stt_en_citrinet_1024 \
  stt_en_citrinet_1024_gamma_0_25 \
  stt_en_citrinet_256 \
  stt_en_citrinet_256_gamma_0_25 \
  stt_en_citrinet_512 \
  stt_en_citrinet_512_gamma_0_25 \
  stt_en_conformer_ctc_large \
  stt_en_conformer_ctc_large_ls \
  stt_en_conformer_ctc_medium \
  stt_en_conformer_ctc_medium_ls \
  stt_en_conformer_ctc_small \
  stt_en_conformer_ctc_small_ls \
  stt_en_conformer_ctc_xlarge \
  stt_en_conformer_transducer_large \
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
    split=test
    python slue_toolkit/eval/eval_nlp_da.py \
        --manifest-dir manifest/slue-hvb \
        --split ${split} \
        --save-dir save/da/nlp_topline_deberta-base \
        --eval \
        --use-gpu \
        --modelname nemo_${modelname} &
done

