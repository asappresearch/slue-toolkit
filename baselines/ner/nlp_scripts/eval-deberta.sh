model_type=$1
eval_set=$2
eval_label=$3

python slue_toolkit/text_ner/ner_deberta.py eval \
--data_dir manifest/slue-voxpopuli/nlp_ner \
--model_dir save/nlp_ner/${model_type} \
--model_type $model_type \
--eval_asr False \
--eval_subset $eval_set \
--eval_label $eval_label