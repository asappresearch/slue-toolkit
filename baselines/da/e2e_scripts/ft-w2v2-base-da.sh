export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"
ngpu=1
seed=1

manifest_dir=$1
save=$2
max_update=$3
warm_up=$4
lr=$5
final_lr_scale=$6

pretrained_ckpt="save/pretrained/wav2vec_small.pt"
if ! [ -f $pretrained_ckpt ]; then
    mkdir -p save/pretrained
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt -O $pretrained_ckpt
fi

user_dir=`realpath ./slue_toolkit/fairseq_addon`
pretrained_ckpt=`realpath $pretrained_ckpt`
config_dir=baselines/da/configs
config=w2v2_da_1gpu
data=`realpath $manifest_dir`
train_subset=fine-tune
valid_subset=dev

mkdir -p $save
tb_save=${save}/tb_logs

# lr=2e-5
update_freq=1
pool_method=self_attn
classifier_dropout=0.2
max_tokens=2800000


fairseq-hydra-train \
  hydra.run.dir=$save \
  hydra.output_subdir=. \
  common.tensorboard_logdir=$tb_save \
  task.data=$data \
  task.normalize=false \
  dataset.train_subset=$train_subset \
  dataset.valid_subset=$valid_subset \
  dataset.max_tokens=$max_tokens \
  distributed_training.distributed_world_size=$ngpu \
  common.user_dir=$user_dir \
  common.seed=$seed \
  optimization.lr="[$lr]" \
  optimization.update_freq="[$update_freq]" \
  optimization.max_update=$max_update \
  model.w2v_path="$pretrained_ckpt" \
  model.pool_method="$pool_method" \
  model.classifier_dropout="$classifier_dropout" \
  model.freeze_finetune_updates=$warm_up \
  lr_scheduler.final_lr_scale=$final_lr_scale \
  --config-dir $config_dir \
  --config-name $config

# python slue_toolkit/prepare/prepare_hvb.py create_manifest
# bash baselines/da/e2e_scripts/ft-w2v2-base-da.sh manifest/slue-hvb save/da/w2v2-base