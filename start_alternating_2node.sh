#!/bin/bash
# 启动两节点交替训练
# 用法: ./start_alternating_2node.sh [实验名前缀]
EXP_PREFIX=${1:-"alternating_$(date +%Y%m%d_%H%M%S)"}
# 先启动节点 0（创建 coordinator）
NODE_RANK=0 EXP_PREFIX="${EXP_PREFIX}" ../run_alternating_2node.sh > ../logs/node0.log 2>&1 &
sleep 3  # 等待节点 0 创建 coordinator
# 再启动节点 1
NODE_RANK=1 EXP_PREFIX="${EXP_PREFIX}" ../run_alternating_2node.sh > ../logs/node1.log 2>&1 &
echo "已启动两个节点，实验名: ${EXP_PREFIX}_node0 和 ${EXP_PREFIX}_node1"
echo "日志: logs/node0.log 和 logs/node1.log"
