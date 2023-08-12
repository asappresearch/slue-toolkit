task=$1 # asr or e2e_ner
split=$2  # dev or test

pretrain_model=w2v2-base

model_dir=save/$task/$pretrain_model
decoder=argmax
results_dir=$model_dir/decode/$split/nolm-$decoder
if [ "$task" = "e2e_ner" ]; then
    manifest_dir=manifest/slue-voxpopuli/$task 
else
    manifest_dir=manifest/slue-voxpopuli/
fi

# Save CTC emission probabilities
python -m slue_toolkit.eval.infer_asr $manifest_dir \
--user-dir slue_toolkit/fairseq_addon \
--task audio_finetuning \
--nbest 1 \
--path ${model_dir}/checkpoints/checkpoint_best.pt \
--gen-subset $split \
--sil-weight 0 \
--max-tokens 4000000 \
--criterion ctc \
--eval-upsample 1.0 \
--results-path $results_dir \
--labels ltr \
--post-process letter \
--w2l-decoder $decoder \
--dump-emissions $results_dir/../emissions.npy

# Save the decoded text
python -m slue_toolkit.eval.infer_asr $manifest_dir \
--user-dir slue_toolkit/fairseq_addon \
--task audio_finetuning \
--nbest 1 \
--path ${model_dir}/checkpoints/checkpoint_best.pt \
--gen-subset $split \
--sil-weight 0 \
--max-tokens 4000000 \
--criterion ctc \
--eval-upsample 1.0 \
--results-path $results_dir \
--labels ltr \
--post-process letter \
--w2l-decoder $decoder 

# Save character-level time stamps from the emissions
python slue_toolkit/prepare/reformat_ctc_output.py \
--split $split \
--model_name $pretrain_model \
--lm "nolm-argmax" \
--task $task  
