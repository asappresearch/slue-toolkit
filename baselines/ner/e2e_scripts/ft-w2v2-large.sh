export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"

ngpu=8
seed=1

manifest_dir=$1
save=$2

pretrained_ckpt="save/pretrained/wav2vec_vox_new.pt"
if ! [ -f $pretrained_ckpt ]; then
    mkdir -p save/pretrained
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt -O $pretrained_ckpt
fi

mkdir -p $save
tb_save=${save}/tb_logs


data=`realpath $manifest_dir`
pretrained_ckpt=`realpath $pretrained_ckpt`

config_dir=baselines/ner/configs
config=w2v2_ner_1gpu
label_type=raw
train_subset=fine-tune_$label_type
valid_subset=dev_$label_type
python slue_toolkit/prepare/create_dict.py manifest/slue-voxpopuli/e2e_ner/fine-tune_$label_type.ltr manifest/slue-voxpopuli/e2e_ner/dict.ltr.txt
python slue_toolkit/prepare/create_dict.py manifest/slue-voxpopuli/e2e_ner/fine-tune_$label_type.wrd manifest/slue-voxpopuli/e2e_ner/dict.wrd.txt

normalize=true
lr=1e-5
max_tokens=800000
max_update=280000

fairseq-hydra-train \
    hydra.run.dir=$save \
    hydra.output_subdir=$save \
    common.tensorboard_logdir=$tb_save \
    task.data=$data \
    dataset.train_subset=$train_subset \
    dataset.valid_subset=$valid_subset \
    distributed_training.distributed_world_size=$ngpu \
    common.seed=$seed \
    model.w2v_path="$pretrained_ckpt" \
    optimization.max_update=$max_update \
    optimization.update_freq=1 \
    dataset.max_tokens=$max_tokens \
    task.normalize=$normalize \
    --config-dir $config_dir \
    --config-name $config
