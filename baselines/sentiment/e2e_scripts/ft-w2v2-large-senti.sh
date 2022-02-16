export MKL_THREADING_LAYER=GNU
export PYTHONWARNINGS="ignore"
ngpu=1
seed=1

manifest_dir=$1
save=$2

pretrained_ckpt="save/pretrained/wav2vec_vox_new.pt"
if ! [ -f $pretrained_ckpt ]; then
    mkdir -p save/pretrained
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt -O $pretrained_ckpt
fi

user_dir=`realpath ./slue_toolkit/fairseq_addon`
pretrained_ckpt=`realpath $pretrained_ckpt`
config_dir=baselines/sentiment/configs
config=w2v2_sentiment_1gpu
data=`realpath $manifest_dir`
train_subset=fine-tune
valid_subset=dev

mkdir -p $save
tb_save=${save}/tb_logs

lr=2e-5
update_freq=1
pool_method=self_attn
classifier_dropout=0.2
max_tokens=1400000

fairseq-hydra-train \
  hydra.run.dir=$save \
  hydra.output_subdir=$save \
  common.tensorboard_logdir=$tb_save \
  task.data=$data \
  task.normalize=true \
  dataset.train_subset=$train_subset \
  dataset.valid_subset=$valid_subset \
  dataset.max_tokens=$max_tokens \
  distributed_training.distributed_world_size=$ngpu \
  common.user_dir=$user_dir \
  common.seed=$seed \
  optimization.lr="[$lr]" \
  optimization.update_freq="[$update_freq]" \
  model.w2v_path="$pretrained_ckpt" \
  model.pool_method="$pool_method" \
  model.classifier_dropout="$classifier_dropout" \
  --config-dir $config_dir \
  --config-name $config
