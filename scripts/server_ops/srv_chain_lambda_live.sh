#!/bin/bash
# **λ 真臂**（库内 30 样本，活 embedder）：用**生产代码**回答"融合的语义项能不能补位"。
#
# 为什么用真臂而不是离线重算器：重算器校验未通过（放行数一致 14 vs 14，但逐条语义项
# 134 条里 14 条对不上，反例是"同块同 sid 的两条候选记录值不同"）⇒ 复现不了生产那一步。
# 生产代码是唯一权威口径（《02》§33、《03》坑 46）。
#
# 对照臂（已跑完，融合 **关**、同一批样本 = `utils/experiments/smoke_kb30.json`）：
#   reports/baseline_fix_runs.txt / _compare.txt
#   own 召回 30/30、**own 放行 28/30**、放行总 **65**、**非 own 0**
#   被放行候选通道：curated_issue 455 / weaviate 280 / sqlite 0
#   漏掉的 2 个（有候选但没放行）：**CVE-2018-6057、CVE-2002-2443**
#
# ===== 预登记预测（跑之前写下来，跑完逐条对照，错了也要写成"预测错了"）=====
# P1 λ=1.5 只新增放行 **0~1** 条；P2 λ=3.0 新增 **1~2** 条；
# P3 被救回的样本**如果**有，最可能是 CVE-2002-2443（6057 在死 embedder 时代语义项只有
#    0.064，是"语义上根本不像"的那一类）；
# P4 非 own 放行：λ=1.5 保持 **0**；λ=3.0 **≤2**（veto=same_file 只让同文件候选吃加分，
#    而库内样本存在"同文件但属别的 CVE"的候选，这是最可能冒出来的误报）。
#
# 用法: bash srv_chain_lambda_live.sh
# 结束标记: CHAIN_LAMBDA_LIVE_DONE
set -u
cd /root/autodl-tmp/MAS || exit 1
OUT=/root/autodl-tmp/chain_lambda_live.log

echo "===== λ 真臂开始 $(date '+%F %T') =====" | tee -a "$OUT"
echo "预登记：P1 λ=1.5 新增 0~1；P2 λ=3.0 新增 1~2；P3 被救回最可能是 CVE-2002-2443；P4 非 own λ=1.5 保持 0、λ=3.0 ≤2" | tee -a "$OUT"

run_arm () {
  local tag="$1" lam="$2" theta="$3"
  local free_mb
  free_mb=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
  if [ "$free_mb" -lt 1500 ]; then
    echo "!! 磁盘不足（剩 ${free_mb}MB），跳过臂 $tag" | tee -a "$OUT"
    return 9
  fi
  echo | tee -a "$OUT"
  echo "===== 臂 $tag 起跑（λ=$lam θ=$theta）$(date '+%T') =====" | tee -a "$OUT"
  bash /root/autodl-tmp/srv_set_gate_fusion.sh on "$lam" "$theta" 2>&1 | grep -E 'gate_fusion|sha256' | tee -a "$OUT"
  bash /root/autodl-tmp/srv_start_batch.sh utils/experiments/smoke_kb30.json \
       "/root/autodl-tmp/batch_${tag}.log" 2>&1 | tail -6 | tee -a "$OUT"
  # 按 pid 等（坑 49）；链路的"等批次"绝不能用会匹配到自己的 pgrep
  bash /root/autodl-tmp/srv_wait_batch.sh 120 2>&1 | tee -a "$OUT"
  echo "臂 $tag 批次结束 $(date '+%T')" | tee -a "$OUT"
  bash /root/autodl-tmp/srv_wait_then_compare_arm.sh "$tag" \
       "/root/autodl-tmp/batch_${tag}.log" 30 2>&1 | tee -a "$OUT"
  df -h /root/autodl-tmp | tail -1 | tee -a "$OUT"
}

run_arm lam15_live 1.5 0.7
run_arm lam30_live 3.0 0.7

# 跑完把开关恢复成生产默认（关）—— 不恢复会让"配置漂移"混进后续实验
bash /root/autodl-tmp/srv_set_gate_fusion.sh off 2>&1 | grep -E 'gate_fusion|sha256' | tee -a "$OUT"

echo | tee -a "$OUT"
echo "===== 三个臂的 own 漏放样本对照（应能直接看出融合救回了谁）=====" | tee -a "$OUT"
for t in baseline_fix lam15_live lam30_live; do
  f="reports/${t}_compare.txt"
  if [ -f "$f" ]; then
    echo "  [$t] 放行无的样本: $(grep -c '放行无' "$f") 个 -> $(grep '放行无' "$f" | tr -s ' ' | cut -d' ' -f2 | tr '\n' ' ')" | tee -a "$OUT"
  else
    echo "  [$t] 缺 compare 输出（$f）" | tee -a "$OUT"
  fi
done
echo | tee -a "$OUT"
echo "===== 完成 $(date '+%F %T') =====" | tee -a "$OUT"
echo "CHAIN_LAMBDA_LIVE_DONE"
