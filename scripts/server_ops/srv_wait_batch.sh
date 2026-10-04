#!/bin/bash
# **等批次结束**（给链路/巡检脚本用）。
#
# 为什么单独一个脚本、而且**按 pid 等**（2026-10-04，坑 49）：
# 以前各处写的是 `while pgrep -f 'mas.py batch' >/dev/null; do sleep 60; done`。
# 这条命令会**匹配到启动它的那条命令自己** —— 因为那条 cmdline 里就含 "mas.py batch"
# 这几个字（它是 pgrep 的参数字面量）。于是循环永远为真：批次 09:27 就跑完了，
# 链路却从 09:28 一直死等到 10:22，日志里只是"没动静"（不报错，最难发现的那种）。
#
# 用法: bash srv_wait_batch.sh [最长等待分钟数，默认 120]
#   exit 0 = 批次已结束；exit 2 = 等超上限
set -u
MAXMIN="${1:-120}"
PIDF=/root/autodl-tmp/batch.pid

if [ ! -f "$PIDF" ]; then
  echo "  ⚠️ 没有 $PIDF（旧批次或不是用 srv_start_batch.sh 起的）⇒ 用严格兜底模式"
  # 兜底模式的模式串带方括号：即使它被写进某个 wrapper 的 cmdline，也不会匹配到那个 wrapper 自己
  for i in $(seq 1 "$MAXMIN"); do
    if ! pgrep -f 'venv/bin/python mas[.]py batch' >/dev/null 2>&1; then
      echo "  批次已结束（等了约 $((i - 1)) 分钟，兜底模式）"; exit 0
    fi
    sleep 60
  done
  echo "  ⚠️ 等满 $MAXMIN 分钟仍未结束（兜底模式）"; exit 2
fi

PID=$(cat "$PIDF" | tr -d ' \r')
[ -z "$PID" ] && { echo "  ⚠️ $PIDF 是空的"; exit 2; }
if ! kill -0 "$PID" 2>/dev/null; then
  echo "  批次 pid=$PID 已经结束（无需等待）"; exit 0
fi
for i in $(seq 1 "$MAXMIN"); do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "  批次 pid=$PID 已结束（等了约 $((i - 1)) 分钟）"; exit 0
  fi
  sleep 60
done
echo "  ⚠️ 等满 $MAXMIN 分钟，pid=$PID 仍在跑"; exit 2
