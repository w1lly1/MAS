#!/bin/bash
# 某一批跑完就**立刻**评测它（不等整条链跑完）。
#
# 为什么：编排脚本把评测放在三段批次的最后，意味着"最有信息量的那一批"也要等到最后才有数字。
# 而 held_fp4 那 4 个样本是**历史上真的误报过**的，它一跑完就能直接回答断言①。
#
# 用法: nohup bash srv_eval_one_batch.sh held_fp4 > /root/autodl-tmp/eval_fp4.log 2>&1 &
set -u
NAME="${1:?用法: srv_eval_one_batch.sh <批次名>}"
cd /root/autodl-tmp/MAS || exit 1

LIST="/root/autodl-tmp/${NAME}_runs.txt"
OUT="/root/autodl-tmp/${NAME}_eval.txt"

say () { echo "[$(date +%H:%M:%S)] $*"; }

say "等 $NAME 的 run 清单出现（$LIST）"
for i in $(seq 1 240); do
  [ -f "$LIST" ] && break
  sleep 30
done
if [ ! -f "$LIST" ]; then
  say "*** 等不到 run 清单，放弃"
  echo "NO_RUNLIST" > "$OUT"
  exit 1
fi
# 清单出现后，再等最后那个 run 的 r2 落盘（避免读到半截文件）
sleep 60
say "$NAME 评测开始（样本 $(wc -l < "$LIST") 个）"
./venv/bin/python -X utf8 utils/experiments/eval_held_runs.py \
    --arms "${NAME}=${LIST}" --db infrastructure/database/mas.db > "$OUT" 2>&1
say "$NAME 评测结束 rc=$? → $OUT"
grep -E "放行总数|同文件撞车|跨文件|通道" "$OUT" | head -6
echo "EVAL_DONE" >> "$OUT"
