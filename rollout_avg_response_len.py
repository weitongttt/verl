#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 vLLM 对指定 parquet 数据集逐条生成响应，并统计响应 token 的平均长度。
默认配置复用 run_fully_async.sh 中的模型与数据路径，可通过命令行覆盖。
"""
import argparse
import math
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Rollout 并计算响应平均长度")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF 模型名或本地权重路径")
    parser.add_argument("--data", default="data/gsm8k/train.parquet", help="parquet 数据路径")
    parser.add_argument("--prompt-column", default="prompt", help="用于生成的列名")
    parser.add_argument("--max-tokens", type=int, default=512, help="生成最大新 token 数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8, help="推理 batch size")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM 张量并行大小")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件: {data_path}")

    df = pd.read_parquet(data_path)
    if args.prompt_column not in df.columns:
        raise ValueError(f"列 {args.prompt_column} 不在数据中，可用列: {list(df.columns)}")
    prompts = df[args.prompt_column].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    total_tokens = 0
    total_samples = 0

    num_batches = math.ceil(len(prompts) / args.batch_size)
    for i in tqdm(range(num_batches), desc="rollout"):
        batch_prompts = prompts[i * args.batch_size : (i + 1) * args.batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)
        for out in outputs:
            # vLLM 的 token_ids 不包含 prompt，只是生成的新 token
            resp_ids = out.outputs[0].token_ids
            total_tokens += len(resp_ids)
            total_samples += 1

    avg_len = total_tokens / total_samples if total_samples else 0
    print(f"响应平均长度（tokens）: {avg_len:.2f}，样本数: {total_samples}")


if __name__ == "__main__":
    main()

