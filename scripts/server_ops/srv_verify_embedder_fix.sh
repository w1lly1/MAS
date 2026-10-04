#!/usr/bin/env bash
# 验证 embedder 修复：等冒烟跑完 → 查日志有没有"加载失败" → 用体检脚本看命中集合是否随文本变化。
# 用法: bash srv_verify_embedder_fix.sh <批次日志> <CVE>
set -u
LOG="${1:?用法: srv_verify_embedder_fix.sh <批次日志> <CVE>}"
CVE="${2:?缺 CVE}"
cd /root/autodl-tmp/MAS || exit 1

echo "=== 1) 等批次结束（最多 15 分钟）==="
for i in $(seq 1 45); do
  if ! pgrep -f 'venv/bin/python mas[.]py batch' >/dev/null 2>&1; then echo "  已结束（等了约 $((i*20)) 秒）"; break; fi
  sleep 20
done

echo
echo "=== 2) 日志里是否还有 embedder 失效（模式：加载失败）==="
N=$(grep -c 加载失败 "$LOG" 2>/dev/null || echo 0)
echo "  '加载失败' 出现次数: $N （期望 0）"
grep -n 加载失败 "$LOG" 2>/dev/null | head -3 | cut -c1-140 || true

echo
echo "=== 3) 该样本的 weaviate 命中是否随查询变化 ==="
D=$(ls -dt "reports/analysis/$CVE"/*/ 2>/dev/null | head -1)
echo "  run: $D"
if [ -z "$D" ]; then echo "  找不到 run 目录"; exit 2; fi
printf '%s\n' "${D#reports/analysis/}" | sed 's:/$::' > /tmp/one_run_list.txt
venv/bin/python -X utf8 utils/experiments/check_embedder_health.py --runs /tmp/one_run_list.txt --limit 1 2>&1 | tail -12
