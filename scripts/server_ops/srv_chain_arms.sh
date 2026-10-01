#!/bin/bash
# 自动链：等 arm2 跑完 → 补跑它的 partial → 打包 → 翻开关 → 跑 arm3 → 补跑 → 打包 → 完成标记。
#
# 设计要点
#  * 每步都写 chain.log，事后可复盘（哪一步何时开始、等了多久、补跑几个）。
#  * 每个臂跑完**先打包**（日志 + CSV + run 列表 + 每个 run 的 r2/run_summary/debug），
#    这样即使后面出问题，前面臂的数据也已经落地成一个小包。
#  * 补跑会覆盖失败那次（make_run_list 顺序覆盖），所以打包时把两个日志都传进去。
set -u
cd /root/autodl-tmp/MAS || exit 1
CHAIN_LOG=/root/autodl-tmp/chain.log
CFG=utils/experiments/smoke_kb30.json

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$CHAIN_LOG"; }

wait_batch() {
  local minutes="${1:-240}"
  for i in $(seq 1 "$minutes"); do
    if ! pgrep -f "venv/bin/python mas.py batch" >/dev/null 2>&1; then
      say "  批次进程已结束（等待 ${i} 分钟）"
      return 0
    fi
    sleep 60
  done
  say "  *** 等待超过 ${minutes} 分钟，继续下一步（可能仍在跑）"
  return 1
}

retry_if_needed() {
  local name="$1" logf="$2"
  local retry="utils/experiments/${name}_retry.json"
  rm -f "$retry"
  ./venv/bin/python utils/experiments/make_retry_batch.py \
      --log "$logf" --config "$CFG" --out "$retry" >> "$CHAIN_LOG" 2>&1 || true
  local n=0
  if [ -f "$retry" ]; then
    n=$(./venv/bin/python -c "import json;print(len(json.load(open('$retry'))['items']))" 2>/dev/null || echo 0)
  fi
  if [ "${n:-0}" -gt 0 ]; then
    say "  $name 有 $n 个 partial → 补跑（新上限）"
    bash /root/autodl-tmp/_srv_start_batch.sh "$retry" "${logf%.log}_retry.log" >> "$CHAIN_LOG" 2>&1
    wait_batch 240
  else
    say "  $name 没有 partial"
    rm -f "$retry"
  fi
}

say "===== 链启动 ====="

# ---------- Arm 2（已在跑，等它） ----------
say "等 Arm 2（基线：特性关 + 旧库）跑完"
wait_batch 300
retry_if_needed arm2_base_old /root/autodl-tmp/arm2_base_old.log
say "打包 Arm 2"
bash /root/autodl-tmp/_srv_pack_arm.sh arm2_baseline_oldkb \
     /root/autodl-tmp/arm2_base_old.log /root/autodl-tmp/arm2_base_old_retry.log >> "$CHAIN_LOG" 2>&1
say "  $(tail -2 "$CHAIN_LOG" | head -1)"

# ---------- 翻开关（代码维打开，库仍是旧的） ----------
say "打开 gap_chunk_semantic_lookup"
bash /root/autodl-tmp/_srv_set_switch.sh on >> "$CHAIN_LOG" 2>&1

# ---------- Arm 3（消融：特性开 + 旧库） ----------
say "启动 Arm 3（消融：特性开 + 旧库）"
bash /root/autodl-tmp/_srv_start_batch.sh "$CFG" /root/autodl-tmp/arm3_ablate_old.log >> "$CHAIN_LOG" 2>&1
wait_batch 300
retry_if_needed arm3_ablate_old /root/autodl-tmp/arm3_ablate_old.log
say "打包 Arm 3"
bash /root/autodl-tmp/_srv_pack_arm.sh arm3_ablate_oldkb \
     /root/autodl-tmp/arm3_ablate_old.log /root/autodl-tmp/arm3_ablate_old_retry.log >> "$CHAIN_LOG" 2>&1
say "  $(tail -2 "$CHAIN_LOG" | head -1)"

# ---------- 收尾 ----------
say "全部完成。产物："
ls -la /root/autodl-tmp/artifacts_*.tgz | awk '{printf "  %8.1f MB  %s\n", $5/1048576, $9}' | tee -a "$CHAIN_LOG"
df -h /root/autodl-tmp | tail -1 | tee -a "$CHAIN_LOG"
touch /root/autodl-tmp/CHAIN_DONE
say "===== 链结束 ====="
