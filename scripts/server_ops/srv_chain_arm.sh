#!/bin/bash
# **通用：跑一个臂**（切开关 → 起批次 → 按 pid 等结束 → 出该臂结论）。
#
# 为什么要有它：链路里"起批次 + 等结束 + 评测"这三步以前是复制粘贴的，
# 于是 `while pgrep -f 'mas.py batch'` 这种自匹配隐患被抄到了每个链路里（坑 49）。
# 现在只此一份，等待统一走 `srv_wait_batch.sh`（按 pid）。
#
# 用法: bash srv_chain_arm.sh <tag> <config> <期望样本数> <held|kbself> [开关: "on 1.5 0.7"|"off"]
set -u
TAG="${1:?用法: srv_chain_arm.sh <tag> <config> <expect> <held|kbself> [switch]}"
CFG="${2:?缺 config}"
EXPECT="${3:-30}"
MODE="${4:?缺 mode（held|kbself）}"
SWITCH="${5:-}"
cd /root/autodl-tmp/MAS || exit 1
OUT="${ARM_CHAIN_LOG:-/root/autodl-tmp/chain_arm_${TAG}.log}"

say () { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

# 磁盘保险：低于 1.5G 不起跑（写满盘会让 Weaviate 转只读、正在跑的批次当场失败）
free_mb=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
if [ "$free_mb" -lt 1500 ]; then
  say "!! 磁盘不足（剩 ${free_mb}MB < 1500MB）⇒ 跳过臂 $TAG"; exit 9
fi

say "===== 臂 $TAG 准备（config=$CFG 期望 $EXPECT 样本，口径 $MODE）====="
if [ -n "$SWITCH" ]; then
  say "  切开关: $SWITCH"
  bash /root/autodl-tmp/srv_set_gate_fusion.sh $SWITCH 2>&1 | grep -E 'gate_fusion|sha256' | tee -a "$OUT"
fi

say "===== 臂 $TAG 起跑 ====="
bash /root/autodl-tmp/srv_start_batch.sh "$CFG" "/root/autodl-tmp/batch_${TAG}.log" 2>&1 | tail -4 | tee -a "$OUT"
bash /root/autodl-tmp/srv_wait_batch.sh 180 2>&1 | tee -a "$OUT"
say "===== 臂 $TAG 批次结束，开始出结论 ====="

if [ "$MODE" = "held" ]; then
  bash /root/autodl-tmp/srv_wait_then_eval_arm.sh "$TAG" "/root/autodl-tmp/batch_${TAG}.log" "$EXPECT" 2>&1 | tee -a "$OUT"
else
  bash /root/autodl-tmp/srv_wait_then_compare_arm.sh "$TAG" "/root/autodl-tmp/batch_${TAG}.log" "$EXPECT" 2>&1 | tee -a "$OUT"
fi

df -h /root/autodl-tmp | tail -1 | tee -a "$OUT"
say "ARM_DONE tag=${TAG}"
