
task=$1 # ppl | e2e | oracle_ppl

model_name="w2v2-base"

if [[ $task == "oracle_ppl" ]]; then
    python slue_toolkit/eval/eval_nel.py evaluate \
        --split dev \
        --task $task \
        --model_name $model_name 
else
    blankarray=( True False )
    echo "Evaluating for different hyper-parameters on dev set"
    for offset in $(seq -0.3 0.02 0.3); do 
        for blank in "${blankarray[@]}"; do
            python slue_toolkit/eval/eval_nel.py evaluate \
            --split dev \
            --task $task \
            --model_name $model_name \
            --offset $offset \
            --blank $blank
        done
    done
    echo "Choosing best hyper parameters"
    python slue_toolkit/eval/eval_nel.py choose_best --task $task
fi

# echo "Evaluating best hyper parameters on test set"
# python slue_toolkit/eval/eval_nel.py evaluate \
# --split test \
# --task $task \
# --model_name $model_name 