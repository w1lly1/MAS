#!/bin/bash
# **第一步**：库外两臂 —— 干净层 30 + overlap15，**门控融合开 λ=3.0 / θ=0.7**。
#
# 为什么是这两臂（预登记，跑前写下来）：
#   · λ 是"语义的话语权"，它真正可能伤人的地方是**库外样本**（那里没有 own，每条放行都是误报）；
#   · 库外层有现成的配对基线：干净层今天跑过"关/开→都是 0 放行"；overlap15 历史是"11 条全同文件、跨文件 0"。 
#
# 预测：
#   ① 干净层：**放行仍为 0**（否决会挡住跨文件候选）；
#   ② overlap15：**放行 ≤11 条、跨文件仍为 0**；
#   ③ 若 overlap15 出现**任何跨文件新放行** ⇒ λ=3.0 不可用，**立刻停**并把开关退回 1.5。
#
# 用 nohup 起（SSH 断了也照跑）；两臂串跑，中间无空档。
set -u
cd /root/autodl-tmp/MAS || exit 1
OUT=/root/autodl-tmp/chain_lam3.log

echo "===== 第一步开始 $(date '+%F %T') =====" | tee -a "$OUT"

echo "--- 打开开关并核对 ---" | tee -a "$OUT"
bash /root/autodl-tmp/srv_set_gate_fusion.sh on 3.0 0.7 2>&1 | grep -E 'gate_fusion|weaviate_top_k|配置 sha' | tee -a "$OUT"

ENABLED=$(venv/bin/python - <<'PY'
import json
c = json.load(open("infrastructure/config/ai_agent_config.json", encoding="utf-8"))
gf = c["second_pass_analysis_agent"].get("gate_fusion") or {}
print("1" if gf.get("enabled") else "0")
PY
)
if [ "$ENABLED" != "1" ]; then
  echo "❌ 开关没打开，拒绝继续" | tee -a "$OUT"; exit 3
fi

run_arm () {
  local tag="$1" cfg="$2" expect="$3"
  echo | tee -a "$OUT"
  echo "===== 臂 $tag 起跑 $(date '+%T') =====" | tee -a "$OUT"
  bash /root/autodl-tmp/srv_start_batch.sh "$cfg" "/root/autodl-tmp/batch_${tag}.log" 2>&1 | tee -a "$OUT"
  while pgrep -f 'mas.py batch' >/dev/null 2>&1; do sleep 60; done
  echo "臂 $tag 批次结束 $(date '+%T')" | tee -a "$OUT"
  bash /root/autodl-tmp/srv_wait_then_eval_arm.sh "$tag" "/root/autodl-tmp/batch_${tag}.log" "$expect" 2>&1 | tee -a "$OUT"
  echo "臂 $tag 评测结束 $(date '+%T')" | tee -a "$OUT"
}

run_arm clean30_lam3 utils/experiments/held_clean30.json 30
run_arm overlap15_lam3 utils/experiments/held_overlap15.json 15

echo | tee -a "$OUT"
echo "--- 两臂结束后的硬盘 ---" | tee -a "$OUT"
df -h /root/autodl-tmp | tail -1 | tee -a "$OUT"
echo "===== 第一步完成 $(date '+%F %T') =====" | tee -a "$OUT"
echo "CHAIN_LAM3_DONE"
