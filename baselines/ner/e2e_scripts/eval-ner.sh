pretrain_model=$1
subset=$2
eval_label=$3 # raw or combined
lm=$4 # nolm or vp_ner/4

beam=500
lm_wt=2
ws=1

# This saves the decoded text at save/e2e_ner/${pretrain_model}/decode
python slue_toolkit/eval/eval_w2v.py eval_ctc_model \
--model save/e2e_ner/${pretrain_model} \
--data manifest/slue-voxpopuli/e2e_ner \
--subset ${subset} \
--lm $lm \
--beam_size $beam \
--lm_weight $lm_wt \
--word_score $ws

# The evaluates the decoded utterances and saves it at save/e2e_ner/${pretrain_model}/metrics
python slue_toolkit/eval/eval_w2v_ner.py eval_ner \
--model_dir save/e2e_ner/${pretrain_model} \
--eval_set ${subset} \
--eval_label ${eval_label} \
--lm ${lm} \
--lm_sfx b${beam}-lw${lm_wt}-ws${ws}
