#!/bin/bash

#install transfomers
if ! [ -d transformers ]; then
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .
    cd -
fi

#prepare data in csv if no exist
if ! [ -f manifest/slue-hvb/fine-tune.huggingface.csv ]; then
    python slue_toolkit/prepare/prepare_hvb_huggingface.py --data manifest/slue-hvb
fi

#fine-tuning
# nlp_modelname=bert-base-cased
nlp_modelname=microsoft/deberta-base
python3 baselines/da/nlp_scripts/run_multi_label_trainer.py \
    --train_file manifest/slue-hvb/fine-tune.huggingface.csv \
    --validation_file manifest/slue-hvb/dev.huggingface.csv \
    --model_name_or_path ${nlp_modelname} \
    --output_dir save/da/nlp_topline_$(basename $nlp_modelname) \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 50 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type constant_with_warmup \
    --num_warmup_steps 800 \
    --checkpointing_steps best_epoch \
    --with_tracking \
    --report_to tensorboard \
    --seed 7

