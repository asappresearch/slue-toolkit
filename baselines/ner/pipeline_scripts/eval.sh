asr_model_type=$1
ner_model_type=$2
eval_set=$3
eval_label=$4
lm=$5

# model_ckpt=`realpath save/asr/${asr_model_type}-vp`
# python slue_toolkit/eval/eval_w2v.py eval_ctc_model \
# --model $model_ckpt \
# --data manifest/slue-voxpopuli \
# --subset ${eval_set} \
# --lm ${lm}

# Add post processing script
# python slue_toolkit/text_ner/reformat_pipeline.py prep_data \
# --model_type ${asr_model_type} \
# --asr_data_dir manifest/slue-voxpopuli \
# --asr_model_dir save/asr/${asr_model_type}-vp \
# --out_data_dir manifest/slue-voxpopuli/nlp_ner \
# --eval_set $eval_set \
# --lm $lm

python slue_toolkit/text_ner/ner_deberta.py eval \
--data_dir manifest/slue-voxpopuli/nlp_ner \
--model_dir save/nlp_ner/${ner_model_type} \
--model_type $ner_model_type \
--eval_asr True \
--eval_subset $eval_set \
--eval_label $eval_label \
--lm $lm \
--asr_model_type $asr_model_type