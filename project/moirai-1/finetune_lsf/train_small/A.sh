#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, change if needed

model=moirai_1.1_R_small_ohlcv  # Your OHLCV config
cp=conf/finetune
data=A
cl=512      # Context length (adjust based on your data)
ps=64       # Patch size (must divide cl evenly: 512/64=8 patches)
mode=M      # Multivariate mode
ft_pattern=full  # Full fine-tuning (all parameters)
exp_name=ohlcv_finetune_A_train_dis96_learning_rate_5e-6

# Fine-tune for different prediction lengths
for pl in 96 192 288; do
  /opt/uni2ts/venv/bin/python -m cli.train \
    -cp $cp \
  exp_name=$exp_name \
  run_name=cl${cl}_pl${pl} \
  model=$model \
  model.patch_size=${ps} \
  model.context_length=$cl \
  model.prediction_length=$pl \
  model.finetune_pattern=$ft_pattern \
  data=${data} \
  data.patch_size=${ps} \
  data.context_length=$cl \
  data.prediction_length=$pl \
  data.mode=${mode} \
  val_data=${data} \
  val_data.patch_size=${ps} \
  val_data.context_length=$cl \
  val_data.prediction_length=$pl \
  val_data.mode=${mode} \
  train_dataloader.num_workers=4 \
  val_dataloader.num_workers=4 \
  trainer.max_epochs=5 \
  model.lr=5e-6
done