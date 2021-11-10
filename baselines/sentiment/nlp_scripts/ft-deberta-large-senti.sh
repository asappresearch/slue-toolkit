#!/bin/bash

#install transfomers
if ! [ -d transformers ]; then
    git clone https://github.com/huggingface/transformers.git
    cd transformers
    pip install -e .
    cd -
fi

#prepare data in csv if no exist
if ! [ -f manifest/slue-voxceleb/fine-tune.huggingface.csv ]; then
    python slue_toolkit/prepare/prepare_voxceleb_huggingface.py --data manifest/slue-voxceleb
fi

#fine-tuning
nlp_modelname=deberta-large
python3 transformers/examples/pytorch/text-classification/run_glue_no_trainer.py \
    --train_file manifest/slue-voxceleb/fine-tune.huggingface.csv \
    --validation_file manifest/slue-voxceleb/dev.huggingface.csv \
    --model_name_or_path microsoft/${nlp_modelname} \
    --output_dir save/sentiment/nlp_topline_${nlp_modelname} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-6 \
    --weight_decay 0.1 \
    --num_train_epochs 50 \
    --gradient_accumulation_steps 4 \
    --seed 7
