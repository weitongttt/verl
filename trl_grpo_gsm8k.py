#!/usr/bin/env python3
"""
TRL GRPO 训练：GSM8K + Qwen2.5-0.5B-Instruct。
与 verl_test.sh 同任务（GRPO + GSM8K + 0.5B），改用 HuggingFace TRL 的 GRPOTrainer。

依赖：pip install trl datasets accelerate
运行：bash run_trl_grpo_gsm8k.sh  或  accelerate launch trl_grpo_gsm8k.py
无需安装 verl，GSM8K 判分（#### strict）已内联实现。
"""

import os
from typing import List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

import swanlab
from trl import GRPOTrainer, GRPOConfig


# ---------- 配置（与 verl 对齐：0.5B + GSM8K）-----------
MODEL_PATH = os.getenv("GRPO_MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
TRAIN_PATH = os.getenv("GRPO_TRAIN_PATH", "data/gsm8k/train.parquet")
OUTPUT_DIR = os.getenv("GRPO_OUTPUT_DIR", "outputs/trl_grpo_gsm8k")
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 256
PER_DEVICE_TRAIN_BATCH_SIZE = 1
NUM_GENERATIONS = 4
LEARNING_RATE = 1e-6
NUM_EPOCHS = 3


class SwanLabCallback(TrainerCallback):
    """把 TRL / HF Trainer 的 log 同步到 SwanLab。"""

    def __init__(self, project: str = "TRL", experiment_name: str | None = None) -> None:
        super().__init__()
        self.project = project
        self.experiment_name = experiment_name or os.getenv("SWANLAB_EXPERIMENT_NAME", "trl_grpo_gsm8k")
        self._initialized = False

    def _maybe_init(self, state) -> None:
        # transformers/trl 在不同版本里不一定会调用 setup()，因此用懒初始化保证 log 前已 init。
        if self._initialized:
            return
        if getattr(state, "is_world_process_zero", True) is False:
            return
        try:
            swanlab.init(project=self.project, experiment_name=self.experiment_name)
            self._initialized = True
        except Exception as e:
            # SwanLab 不可用时不要中断训练
            print(f"[swanlab] init failed: {e}")

    def on_train_begin(self, args, state, control, **kwargs):
        self._maybe_init(state)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        self._maybe_init(state)
        if not self._initialized:
            return
        if getattr(state, "is_world_process_zero", True) is False:
            return
        # logs 是一个 dict，例如 {"loss": ..., "reward": ...}
        try:
            swanlab.log(logs, step=state.global_step)
        except Exception as e:
            print(f"[swanlab] log failed: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        if self._initialized:
            try:
                swanlab.finish()
            except Exception as e:
                print(f"[swanlab] finish failed: {e}")


def _extract_answer_strict(solution_str: str):
    """提取 #### 后的数字，与 verl gsm8k 一致。返回 (answer_str or None, score=0/1 用的比较值)。"""
    import re
    s = solution_str.strip()
    if len(s) > 300:
        s = s[-300:]
    matches = re.findall(r"####\s*(\-?[0-9\.,]+)", s)
    if not matches:
        return None
    return matches[-1].replace(",", "").replace("$", "").strip()


def _gsm8k_compute_score(solution_str: str, ground_truth: str) -> float:
    """GSM8K strict 判分（#### 数字），与 verl 的 gsm8k.compute_score 一致。"""
    answer = _extract_answer_strict(solution_str)
    if answer is None:
        return 0.0
    return 1.0 if answer == ground_truth.strip() else 0.0


def _completion_to_text(completion) -> str:
    """TRL 可能传 standard（字符串）或 conversational（list of {role,content}），统一成字符串。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, (list, tuple)) and len(completion) > 0:
        msg = completion[0]
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"]) or ""
        if isinstance(msg, str):
            return msg
    return str(completion)


def gsm8k_reward_func(completions: List[str], ground_truth: List[str], **kwargs):
    """TRL 要求的 reward 函数：返回与 completions 等长的 float 列表。"""
    # 统一 completion 为字符串（TRL 可能传 conversational 格式）
    solution_strs = [_completion_to_text(c) for c in completions]
    rewards = [
        _gsm8k_compute_score(sol, gt)
        for sol, gt in zip(solution_strs, ground_truth)
    ]
    return rewards


def _extract_content_from_repr(s: str) -> str:
    """若 s 是 list/dict 的 repr（如 \"[{'content': '...', 'role': 'user'}]\"), 解析并取第一条 content。"""
    s = (s or "").strip()
    if not s or (s[0] != "[" and s[0] != "{"):
        return s
    try:
        import ast
        obj = ast.literal_eval(s)
        if isinstance(obj, list) and len(obj) > 0:
            first = obj[0]
            if isinstance(first, dict) and "content" in first:
                return str(first["content"]).strip() or s
        if isinstance(obj, dict) and "content" in obj:
            return str(obj["content"]).strip() or s
    except Exception:
        pass
    return s


def _normalize_chat(chat) -> List[dict]:
    """把 parquet 的 prompt（可能为 numpy 数组 / 嵌套结构）转成纯 Python list[dict]，保证 content 为题目原文。"""
    if not isinstance(chat, (list, tuple)):
        return [{"role": "user", "content": _extract_content_from_repr(str(chat))}]
    out = []
    for m in list(chat):
        if not isinstance(m, dict):
            out.append({"role": "user", "content": _extract_content_from_repr(str(m))})
            continue
        role = str(m.get("role", "user"))
        raw = m.get("content", "")
        content = str(raw) if raw is not None else ""
        # 若 content 是整条消息列表的 repr，解析出真正的题目文本
        content = _extract_content_from_repr(content)
        out.append({"role": role, "content": content})
    return out


def build_gsm8k_dataset(train_path: str, tokenizer, max_samples: int = -1) -> Dataset:
    """
    从 parquet 构建 TRL 标准格式数据集：prompt（字符串）、ground_truth。
    prompt 使用与 verl 一致的 apply_chat_template(..., add_generation_prompt=True)。
    """
    df = pd.read_parquet(train_path)
    if max_samples > 0:
        df = df.head(max_samples)

    prompts: List[str] = []
    ground_truths: List[str] = []

    for _, row in df.iterrows():
        chat = _normalize_chat(row["prompt"])
        prompt_str = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
        # 与 verl data.max_prompt_length 对齐：强制截断 prompt token 长度（从左侧截断）
        if MAX_PROMPT_LENGTH and MAX_PROMPT_LENGTH > 0:
            ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
            if len(ids) > MAX_PROMPT_LENGTH:
                ids = ids[-MAX_PROMPT_LENGTH:]
                prompt_str = tokenizer.decode(ids, skip_special_tokens=False)
        prompts.append(prompt_str)

        rm = row.get("reward_model", {}) or {}
        gt = rm.get("ground_truth", "")
        if isinstance(gt, (list, tuple)):
            gt = gt[0] if len(gt) > 0 else ""
        ground_truths.append(str(gt).strip())

    ds = Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})
    # 数据处理 debug：打印前 2 条 prompt 尾部与 ground_truth
    print("[data debug] dataset size:", len(prompts))
    for i in range(min(2, len(prompts))):
        p = prompts[i]
        tail = p[-400:] if len(p) > 400 else p
        print(f"[data debug] sample {i} ground_truth={ground_truths[i]!r} prompt_tail={repr(tail)[:200]}...")
    return ds


def main():
    print("=== TRL GRPO on GSM8K ===")
    print("model:", MODEL_PATH)
    print("data:", TRAIN_PATH)
    print("output_dir:", OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"

    train_dataset = build_gsm8k_dataset(TRAIN_PATH, tokenizer)
    print("train samples:", len(train_dataset))

    # 与 verl data.train_batch_size 对齐：设 GRPO_TRAIN_BATCH_SIZE=256 则总 batch=256，per_device=256//卡数
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    total_batch_env = os.environ.get("GRPO_TRAIN_BATCH_SIZE", "")
    if total_batch_env:
        per_device = max(1, int(total_batch_env) // world_size)
        print(f"GRPO_TRAIN_BATCH_SIZE={total_batch_env} → per_device_train_batch_size={per_device} (world_size={world_size})")
    else:
        per_device = PER_DEVICE_TRAIN_BATCH_SIZE

    # 不改你的超参（batch/generations/长度），只做省显存优化：
    # - 关 use_cache（训练/长序列更省显存）
    # - 开 gradient checkpointing（会更慢，但显著省显存）
    # - 尽量用更省显存的注意力实现（默认 sdpa；若环境支持可切换 flash_attention_2）
    attn_impl = os.getenv("GRPO_ATTN_IMPL", "sdpa")  # 可选：sdpa / flash_attention_2 / eager
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype="auto",
            attn_implementation=attn_impl,
        )
    except TypeError:
        # 旧 transformers 不支持 attn_implementation
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto")

    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # TRL GRPOConfig：不同版本字段略有差异，只传入当前版本支持的字段
    config_fields = getattr(GRPOConfig, "__dataclass_fields__", {}) or {}
    config_kwargs = dict(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=per_device,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_generations=NUM_GENERATIONS,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        remove_unused_columns=False,
    )
    # 某些 TRL 版本支持这些开关；存在则打开（不改变超参，只是节省显存）
    if "gradient_checkpointing" in config_fields:
        config_kwargs["gradient_checkpointing"] = True
    if "use_cache" in config_fields:
        config_kwargs["use_cache"] = False
    if "max_prompt_length" in config_fields:
        config_kwargs["max_prompt_length"] = MAX_PROMPT_LENGTH

    training_args = GRPOConfig(**{k: v for k, v in config_kwargs.items() if k in config_fields})

    # 默认先不启用 SwanLab，后面需要时可设置环境变量 USE_SWANLAB_TRL=1 再打开
    callbacks = []
    if os.getenv("USE_SWANLAB_TRL", "0") == "1":
        callbacks.append(SwanLabCallback(project="TRL"))

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=gsm8k_reward_func,
        callbacks=callbacks or None,
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(OUTPUT_DIR)
    tok = getattr(trainer, "tokenizer", None) or tokenizer
    if tok is not None:
        tok.save_pretrained(OUTPUT_DIR)
    print("Done. Saved to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
