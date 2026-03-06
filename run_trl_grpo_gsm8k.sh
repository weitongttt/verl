#!/bin/bash
# TRL GRPO：GSM8K + Qwen2.5-0.5B-Instruct
# 需先安装：pip install trl datasets accelerate（或 pip install -r requirements_trl.txt）
set -e

# SwanLab API Key（与 verl_test.sh 一致，如需覆盖可在外面手动 export）
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-hlo16D6KKxblfDAgvGxVQ}"
export GRPO_MODEL_PATH="${GRPO_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
export GRPO_TRAIN_PATH="${GRPO_TRAIN_PATH:-data/gsm8k/train.parquet}"
export GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-outputs/trl_grpo_gsm8k}"
export GRPO_TRAIN_BATCH_SIZE=256
export USE_SWANLAB_TRL=1

# 单机多卡用 accelerate
if command -v accelerate &>/dev/null; then
  accelerate launch trl_grpo_gsm8k.py "$@"
else
  python3 trl_grpo_gsm8k.py "$@"
fi
