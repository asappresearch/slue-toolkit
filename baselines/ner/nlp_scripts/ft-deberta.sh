model_type=$1
label_type=$2

python slue_toolkit/text_ner/ner_deberta.py train \
--data_dir manifest/slue-voxpopuli/nlp_ner \
--model_dir save/nlp_ner/${model_type}_${label_type} \
--model_type $model_type \
--label_type $label_type