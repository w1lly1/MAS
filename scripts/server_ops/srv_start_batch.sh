#!/bin/bash
# 启动一个批次（nohup，立刻返回；长跑靠巡检脚本看进度）
# 用法: bash srv_start_batch.sh <config> <logfile>
#
# ⚠️ **必须显式导出 HF_HOME**（2026-10-03 查出的重大缺陷）：
#   模型在 `/root/autodl-tmp/hf-cache`，该路径靠 `~/.bashrc` 里的 `HF_HOME` 指路；
#   而批次是**非交互 shell** 起的（ssh 执行命令），`.bashrc` 开头有
#   `case $- in *i*) ;; *) return;; esac` 守卫会直接 return ⇒ `HF_HOME` 没设 ⇒
#   `local_files_only=True` 找不到模型 ⇒ **embedder 静默退回"校验和兜底向量"** ⇒
#   查询向量与文本无关 ⇒ **整个向量/语义通道是死的**（实测：三臂 81/81 次查询命中同一批条目）。
#   体检脚本：utils/experiments/check_embedder_health.py
set -u
CFG="${1:?用法: srv_start_batch.sh <config> <logfile>}"
LOG="${2:-/root/autodl-tmp/batch.log}"
cd /root/autodl-tmp/MAS || exit 1

export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf-cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
: > "$LOG"
echo "HF_HOME=$HF_HOME  HF_HUB_OFFLINE=$HF_HUB_OFFLINE  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE" >> "$LOG"

# **起飞前断言**：模型必须能离线加载，否则拒绝起批次
# （宁可不起，也不要拿兜底向量跑一整天 —— 这正是之前悄悄发生的事）
if ! ./venv/bin/python - <<'PY' >> "$LOG" 2>&1
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
AutoModel.from_pretrained("distilbert-base-uncased", local_files_only=True)
print("[preflight] distilbert 离线加载 ✓")
PY
then
  echo "❌ preflight 失败：distilbert 离线加载不了 —— 拒绝起批次（否则会静默用兜底向量）" | tee -a "$LOG"
  exit 3
fi

nohup env HF_HOME="$HF_HOME" HF_HUB_OFFLINE="$HF_HUB_OFFLINE" TRANSFORMERS_OFFLINE="$TRANSFORMERS_OFFLINE" \
  ./venv/bin/python mas.py batch -c "$CFG" >> "$LOG" 2>&1 &
echo "started pid=$! log=$LOG"
echo "config=$CFG items=$(./venv/bin/python -c "import json;print(len(json.load(open('$CFG'))['items']))")"
