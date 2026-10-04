#!/bin/bash
# **活 embedder 下的库外两臂**（融合关 = 生产默认配置）：查"修好 embedder 后，误报面有没有变大"。
#
# 为什么这件事优先于 λ 调参：
#   · 今天修好 embedder 后，语义通道**第一次真的在放行**（库内臂：weaviate 280 条被放行、占 38.1%）；
#   · 而库外样本上**每一条放行都是误报面** ⇒ 这是现在最大的风险点；
#   · 而 λ 的作用是"语义补位"，属**次级问题**（收益远小于"通道从死到活"）。
#
# 旧基线（死 embedder，**结论已作废**，此处仅作参照）：
#   干净层 30 → 放行 0；overlap15 → 放行 11（全同文件、跨文件 0）。
#
# 预测（跑前写下）：
#   ① 干净层：放行**仍然很少**（≤3 条），且**跨文件为主**（该层与库中无同文件条目）；
#   ② overlap15：放行**可能多于 11**（语义通道活了），**但跨文件应当仍然很少**；
#   ③ 若出现大量跨文件放行（相对 45 个样本而言），说明"活语义通道 + 现有守卫"不够安全 ⇒ 需要收紧。
set -u
cd /root/autodl-tmp/MAS || exit 1
OUT=/root/autodl-tmp/chain_held_live.log

echo "===== 活 embedder 库外两臂开始 $(date '+%F %T') =====" | tee -a "$OUT"
bash /root/autodl-tmp/srv_set_gate_fusion.sh off 2>&1 | grep gate_fusion | tee -a "$OUT"

run_arm () {
  local tag="$1" cfg="$2" expect="$3"
  # 磁盘保险：低于 1.5G 就不起跑（避免写满盘把库/报告搞坏）
  local free_mb
  free_mb=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
  if [ "$free_mb" -lt 1500 ]; then
    echo "!! 磁盘不足（剩 ${free_mb}MB < 1500MB），跳过臂 $tag" | tee -a "$OUT"
    return 9
  fi
  echo | tee -a "$OUT"
  echo "===== 臂 $tag 起跑 $(date '+%T') =====" | tee -a "$OUT"
  bash /root/autodl-tmp/srv_start_batch.sh "$cfg" "/root/autodl-tmp/batch_${tag}.log" 2>&1 | tee -a "$OUT"
  # 等批次结束：**按 pid 等**（坑 49：`pgrep -f 'mas.py batch'` 会匹配到启动命令自己 ⇒ 死等）
  bash /root/autodl-tmp/srv_wait_batch.sh 120 2>&1 | tee -a "$OUT"
  echo "臂 $tag 批次结束 $(date '+%T')" | tee -a "$OUT"
  bash /root/autodl-tmp/srv_wait_then_eval_arm.sh "$tag" "/root/autodl-tmp/batch_${tag}.log" "$expect" 2>&1 | tee -a "$OUT"
  echo "臂 $tag 评测结束 $(date '+%T')" | tee -a "$OUT"
  df -h /root/autodl-tmp | tail -1 | tee -a "$OUT"
}

run_arm clean30_live utils/experiments/held_clean30.json 30
run_arm overlap15_live utils/experiments/held_overlap15.json 15

echo | tee -a "$OUT"
df -h /root/autodl-tmp | tail -1 | tee -a "$OUT"
echo "===== 完成 $(date '+%F %T') =====" | tee -a "$OUT"
echo "CHAIN_HELD_LIVE_DONE"
