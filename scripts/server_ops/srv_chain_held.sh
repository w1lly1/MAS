#!/bin/bash
# 依次跑 T5 的三个库外(held)批次，然后自动收 run 列表并评测"误报面"。
#
# ## 顺序是有讲究的
# 先跑**最有检验力**的两个小批次，最后才是"干净"的 30 个：
#   1) held_fp4        4 个   —— 历史上真的误报过的样本（回归测试，最有信息量）
#   2) held_overlap15 15 个   —— 与库里文件同末两级路径相撞的那一层
#   3) held_kb30      30 个   —— 与库零重合的干净层（预期放行≈0，检验力最弱）
# 万一时间/磁盘不够，前面的结果已经落袋。
#
# 用法: bash /root/autodl-tmp/srv_chain_held.sh
set -u
cd /root/autodl-tmp/MAS || exit 1

run_one () {
  local name="$1" cfg="$2" log="/root/autodl-tmp/${name}_log.txt"
  echo "==================== $name 开始 $(date +%H:%M:%S) ===================="
  echo "配置: $cfg  样本: $(./venv/bin/python -c "import json;print(len(json.load(open('$cfg'))['items']))")"
  ./venv/bin/python mas.py batch -c "$cfg" > "$log" 2>&1
  local rc=$?
  echo "$name 结束 rc=$rc：partial $(grep -c '记为 partial' "$log" || true) 个"
  grep -E "成功|未确认完成|失败" "$log" | tail -3
  ./venv/bin/python utils/experiments/make_run_list.py --logs "$log" \
      --out "/root/autodl-tmp/${name}_runs.txt" 2>/dev/null | tail -3
  df -h /root/autodl-tmp | tail -1
  echo
}

run_one held_fp4        utils/experiments/held_fp4.json
run_one held_overlap15  utils/experiments/held_overlap15.json
run_one held_kb30       utils/experiments/held_kb30.json

echo "==================== 全部批次结束，开始评测 $(date +%H:%M:%S) ===================="
for name in held_fp4 held_overlap15 held_kb30; do
  f="/root/autodl-tmp/${name}_runs.txt"
  [ -f "$f" ] || { echo "$name 没有 run 列表，跳过"; continue; }
  echo "-------- $name（库外：每一条放行都是误报面）--------"
  ./venv/bin/python utils/experiments/eval_held_runs.py \
      --arms "$name=$f" --db infrastructure/database/mas.db 2>/dev/null | head -22
done
echo "CHAIN_HELD_DONE"
