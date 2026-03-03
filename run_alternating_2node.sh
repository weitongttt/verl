#!/bin/bash
# 两节点交替训练脚本：节点 A 做 rollouter 时，节点 B 做 trainer，然后轮换
# 基于 run_fully_async.sh，但配置为 2 节点、1 GPU/节点，并实现角色轮换机制

set -x

export VERL_USE_MODELSCOPE=True
export HYDRA_CONFIG_PATH="/home/weitong/verl_alter/verl/verl/trainer/config"
export VERL_FORCE_DEVICE=npu
export SWANLAB_API_KEY="hlo16D6KKxblfDAgvGxVQ"
# 确保使用本地源码版本的 verl_alter
export PYTHONPATH=/home/weitong/verl_alter/verl:$PYTHONPATH

# NPU 环境变量设置
NODE_RANK=${NODE_RANK:-0}  # 通过环境变量传入，0 或 1
if [ "$NODE_RANK" = "0" ]; then
    export ASCEND_RT_VISIBLE_DEVICES=0
else
    export ASCEND_RT_VISIBLE_DEVICES=1
fi

export PATH=/usr/local/Ascend/driver/tools:/usr/local/bin:$PATH
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ASCEND_ENABLE_NZ="0"

rollout_mode="async"
rollout_name="vllm"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# 训练参数
adv_estimator="grpo"
train_files="../data/gsm8k/train.parquet"
val_files="../data/gsm8k/test.parquet"
model_path="Qwen/Qwen2.5-0.5B-Instruct"

train_prompt_bsz=0
gen_prompt_bsz=1
max_model_len=32768
max_response_length=4096
max_num_batched_tokens=$((max_response_length * 4))
n_resp_per_prompt=4
use_dynamic_bsz=true
total_rollout_steps=$(((250*32)))
mini_batch_size=32
require_batches=1
test_freq=1000
# 交替训练：每个阶段训练多少步后切换角色
training_steps_per_phase=5
partial_rollout=false

# 实验名
project_name="ascend_alternating_2node"
EXP_PREFIX=${EXP_PREFIX:-"alternating_$(date +%Y%m%d_%H%M%S)"}
experiment_name="${EXP_PREFIX}_node${NODE_RANK}"

# 两节点配置：每个节点 1 个 GPU，总共 2 个节点
# 注意：这里需要根据实际 Ray 集群配置调整
# 如果使用 SLURM 或其他调度器，可能需要通过环境变量获取节点信息

PYTHONUNBUFFERED=1 python -m verl.experimental.fully_async_policy.alternating_main \
    data.train_files=${train_files} \
    data.val_files=${val_files} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.return_raw_chat=${return_raw_chat} \
    data.shuffle=False \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.model.path=${model_path} \
    algorithm.adv_estimator=${adv_estimator} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger='[console,swanlab]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.device=npu \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node=1 \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    rollout.test_freq="${test_freq}" \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout="${partial_rollout}" \
    async_training.use_trainer_do_validate=false \
    async_training.use_alternating_training=true \
    async_training.max_alternations=50 \
    async_training.training_steps_per_phase="${training_steps_per_phase}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_batch_size} \

