export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"

ngpu=1
seed=1

manifest_dir=$1
save=$2

pretrained_ckpt="save/pretrained/wav2vec_small.pt"
if ! [ -f $pretrained_ckpt ]; then
    mkdir -p save/pretrained
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -O $pretrained_ckpt
fi

mkdir -p $save
tb_save=${save}/tb_logs


data=`realpath $manifest_dir`
pretrained_ckpt=`realpath $pretrained_ckpt`

config_dir=baselines/ner/configs
config=w2v2_ner_1gpu
train_subset=fine-tune_raw
valid_subset=dev_raw

normalize=false
lr=5e-5
max_tokens=3200000
max_update=20000

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
    dataset.max_tokens=$max_tokens \
    task.normalize=$normalize \
    --config-dir $config_dir \
    --config-name $config
