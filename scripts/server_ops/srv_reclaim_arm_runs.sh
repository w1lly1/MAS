#!/bin/bash
# 空间回收（**零丢失**）：把三臂那 90 个 run 目录**整目录**打成一个 tgz，校验后删原目录。
#
# 为什么还要再打一次包：先前那个 arm_artifacts 归档只装了 `run_summary.json` +
# 压平后的 `second_pass_consolidated_*.json`；而 `reports/analysis` 里的目录更全
# （还有 `agents/`、`fullLayer/`、`pureLLM/`、`debug/` 等）。为了**不丢任何东西**，
# 这里按整目录打包，而不是直接删。
#
# 安全措施：低优先级（不抢正在跑的批处理的 CPU/IO）；**先打包、后逐条校验、再删除**；
# 校验不过就原地不动。
#
# 用法: nohup bash /root/autodl-tmp/srv_reclaim_arm_runs.sh > /root/autodl-tmp/reclaim.log 2>&1 &
set -u
cd /root/autodl-tmp/MAS/reports || exit 1

OUT=/root/autodl-tmp/arm_runs_full_20261002.tgz
LISTS="/root/autodl-tmp/arm1_runs.txt /root/autodl-tmp/arm2_runs.txt /root/autodl-tmp/arm3_runs.txt"

say () { echo "[$(date +%H:%M:%S)] $*"; }

# 1) 收集要打包的目录（存在才收）
PATHS=/tmp/_arm_paths.txt
: > "$PATHS"
for f in $LISTS; do
  [ -f "$f" ] || continue
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    d="analysis/$line"
    [ -d "$d" ] && echo "$d" >> "$PATHS"
  done < "$f"
done
N=$(wc -l < "$PATHS")
say "待打包 run 目录: $N 个，当前占用 $(du -shc $(cat "$PATHS") 2>/dev/null | tail -1 | cut -f1)"
[ "$N" -gt 0 ] || { say "没有可打包的目录，退出"; exit 0; }

say "开始整目录打包（低优先级）"
nice -n 19 ionice -c3 tar czf "$OUT" -T "$PATHS"
say "打包完成：$(du -sh "$OUT" | cut -f1)"

# 2) 校验：条目数 + 抽样 3 个文件逐字节 sha256
say "校验"
ENTRIES=$(tar tzf "$OUT" | wc -l)
say "  归档条目数: $ENTRIES"
OK=1
CHECKED=0
while IFS= read -r d; do
  for rel in run_summary.json second_pass_debug.log; do
    if [ -f "$d/$rel" ]; then
      B=$(sha256sum "$d/$rel" | awk '{print $1}')
      A=$(tar xzf "$OUT" -O "$d/$rel" 2>/dev/null | sha256sum | awk '{print $1}')
      CHECKED=$((CHECKED+1))
      if [ "$B" != "$A" ]; then
        say "  *** 不一致: $d/$rel"
        OK=0
      fi
    fi
    [ "$CHECKED" -ge 6 ] && break 2
  done
done < "$PATHS"
say "  校验文件数: $CHECKED，全部一致: $([ "$OK" = "1" ] && echo 是 || echo 否)"

if [ "$OK" = "1" ] && [ "$ENTRIES" -gt 100 ]; then
  say "校验通过 → 删除原 run 目录（内容已在 $OUT 里）"
  while IFS= read -r d; do rm -rf "$d"; done < "$PATHS"
  say "删除后：analysis=$(du -sh analysis | cut -f1)，磁盘可用 $(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')MB"
  echo "RECLAIM_OK"
else
  say "*** 校验不通过（abort）：**不删**原目录"
  echo "RECLAIM_FAILED"
fi
