#!/bin/bash
# 今晚剩下的 GPU 活，一条龙跑完（nohup 起，抗 ssh 断线）。
#
#   0) 等正在跑的 T2 批次结束
#   1) T2 后处理：run 列表 / 完成度 / 生效性检查 / 与昨晚 Arm1 对照 / 未放行样本拒因
#   2) T5 三个库外批次（fp4 → overlap15 → kb30，先跑最有检验力的）
#   3) 每批之后：磁盘守卫 + 评测（库外：每条放行都是误报面）
#   4) 打印汇总，最后写 TONIGHT_DONE 标记
#
# 用法（服务器上）: nohup bash /root/autodl-tmp/srv_tonight.sh > /root/autodl-tmp/tonight.log 2>&1 &
set -u
cd /root/autodl-tmp/MAS || exit 1

LOG=/root/autodl-tmp/tonight.log
say () { echo "[$(date +%H:%M:%S)] $*"; }

say "===== 0) 等正在跑的 T2 批次结束 ====="
while pgrep -f 'mas.py batch' >/dev/null 2>&1; do sleep 30; done
say "T2 批次已结束"

say "===== 1) T2 后处理 ====="
bash /root/autodl-tmp/srv_t2_after.sh 2>&1 | tee /root/autodl-tmp/t2_after_report.txt | tail -60

# 磁盘守卫：低于 3G 就按 run 目录精确清理（保留正在评测的那几批）
guard () {
  local free_mb
  free_mb=$(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')
  say "磁盘可用 ${free_mb}MB"
  if [ "$free_mb" -lt 3000 ]; then
    say "空间偏紧，执行精确清理（保留 t2/held 各批 run 清单）"
    bash /root/autodl-tmp/srv_prune_run_dirs.sh --older-than-min 20 \
      --keep /root/autodl-tmp/t2_runs.txt \
      --keep /root/autodl-tmp/held_fp4_runs.txt \
      --keep /root/autodl-tmp/held_overlap15_runs.txt \
      --keep /root/autodl-tmp/held_kb30_runs.txt \
      --apply 2>&1 | tail -12
  fi
}

run_one () {
  local name="$1" cfg="$2"
  # ⚠️ 必须分成两条 local：bash 先展开整行的词、再执行赋值，所以
  # `local name="$1" log="...${name}..."` 在 set -u 下会 "unbound variable"
  # （这正是第一次跑本脚本时挂掉的原因，见《03》坑 35）。
  local log="/root/autodl-tmp/${name}_log.txt"
  guard
  say "==================== $name 开始 ===================="
  ./venv/bin/python mas.py batch -c "$cfg" > "$log" 2>&1
  say "$name 结束 rc=$? partial=$(grep -c '记为 partial' "$log" || true)"
  grep -E "成功|未确认完成|失败" "$log" | tail -3
  ./venv/bin/python utils/experiments/make_run_list.py --logs "$log" \
      --out "/root/autodl-tmp/${name}_runs.txt" 2>/dev/null | tail -3
}

say "===== 2) T5：三个库外批次 ====="
run_one held_fp4       utils/experiments/held_fp4.json
run_one held_overlap15 utils/experiments/held_overlap15.json
run_one held_kb30      utils/experiments/held_kb30.json

say "===== 3) 库外评测（每条放行都是误报面）====="
for name in held_fp4 held_overlap15 held_kb30; do
  f="/root/autodl-tmp/${name}_runs.txt"
  [ -f "$f" ] || { say "$name 没有 run 列表，跳过"; continue; }
  say "-------- $name --------"
  ./venv/bin/python utils/experiments/eval_held_runs.py \
      --arms "$name=$f" --db infrastructure/database/mas.db 2>/dev/null \
      | tee "/root/autodl-tmp/${name}_eval.txt" | head -16
done

say "===== 4) 收尾：状态与体积 ====="
df -h /root/autodl-tmp | tail -1
du -sh reports/analysis 2>/dev/null
say "TONIGHT_DONE"
