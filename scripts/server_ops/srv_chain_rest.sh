#!/bin/bash
# **剩余计划一条龙**（clean30_live 已经跑完，这里只做它没做完的部分）：
#   ① 补 clean30_live 的结论（批次 09:27 就跑完了，但链路因"自匹配死等"卡住没出结论）
#   ② overlap15_live（库外 overlap 15 样本，融合关）
#   ③ λ 真臂 λ=1.5（库内 30 样本）
#   ④ λ 真臂 λ=3.0
#   ⑤ 恢复开关为 off（生产默认），并打印三臂漏放样本对照
#
# 结束标记: CHAIN_REST_DONE
set -u
cd /root/autodl-tmp/MAS || exit 1
OUT=/root/autodl-tmp/chain_rest.log
say () { echo "[$(date '+%F %T')] $*" | tee -a "$OUT"; }

say "===== 剩余计划开始（clean30_live 批次早已结束，先补它的结论）====="
bash /root/autodl-tmp/srv_wait_then_eval_arm.sh clean30_live \
     /root/autodl-tmp/batch_clean30_live.log 30 2>&1 | tee -a "$OUT"

say "===== ② overlap15_live（库外，融合关）====="
ARM_CHAIN_LOG="$OUT" bash /root/autodl-tmp/srv_chain_arm.sh \
  overlap15_live utils/experiments/held_overlap15.json 15 held "off"

say "===== ③ λ 真臂 λ=1.5（库内 30 样本）====="
ARM_CHAIN_LOG="$OUT" bash /root/autodl-tmp/srv_chain_arm.sh \
  lam15_live utils/experiments/smoke_kb30.json 30 kbself "on 1.5 0.7"

say "===== ④ λ 真臂 λ=3.0 ====="
ARM_CHAIN_LOG="$OUT" bash /root/autodl-tmp/srv_chain_arm.sh \
  lam30_live utils/experiments/smoke_kb30.json 30 kbself "on 3.0 0.7"

say "===== ⑤ 恢复开关为 off（生产默认）====="
bash /root/autodl-tmp/srv_set_gate_fusion.sh off 2>&1 | grep -E 'gate_fusion|sha256' | tee -a "$OUT"

say "===== 三个库内臂的 own 漏放样本对照（看融合救回了谁）====="
for t in baseline_fix lam15_live lam30_live; do
  f="reports/${t}_compare.txt"
  if [ -f "$f" ]; then
    say "  [$t] 漏放 $(grep -c '放行无' "$f") 个 -> $(grep '放行无' "$f" | tr -s ' ' | cut -d' ' -f2 | tr '\n' ' ')"
  else
    say "  [$t] 缺 compare 输出"
  fi
done
say "===== 完成 ====="
echo "CHAIN_REST_DONE"
