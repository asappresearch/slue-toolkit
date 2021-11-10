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

config_dir=baselines/asr/configs
config=w2v2_asr_1gpu
train_subset=fine-tune
valid_subset=dev

normalize=true
lr=1e-5
max_tokens=700000
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
    dataset.max_tokens=$max_tokens \
    task.normalize=$normalize \
    --config-dir $config_dir \
    --config-name $config