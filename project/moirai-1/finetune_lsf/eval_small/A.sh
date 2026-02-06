#!/bin/bash

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

# Path to your best checkpoint
ckpt_path="outputs/finetune/ohlcv_finetune/moirai_1.1_R_small_ohlcv/full/A/cl512_pl96/checkpoints/epoch=XX-step=YYYY.ckpt"

python -m cli.eval \
  model=moirai_lightning_ckpt \
  model.ckpt_path=$ckpt_path \
  model.patch_size=64 \
  model.context_length=512 \
  run_name=A_eval \
  data=lsf \
  data.dataset_name=A \
  data.prediction_length=96 \
  data.context_length=512 \
  data.patch_size=64 \
  data.mode=M
