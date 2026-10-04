#!/bin/bash
# 磁盘归因：把 reports/analysis 的体积按"臂"拆开，区分
#   ① 死 embedder 的臂（**结论已作废**，向量侧证据无效）——可清理的对象
#   ② 活 embedder 的臂（有效证据）——要保留
#   ③ 未被任何 run 列表覆盖的孤儿目录（多半是被覆盖掉的旧臂）
#
# 为什么需要它：400 样本预计 ~20 MB/样本 ≈ 8 GB，而盘只剩 5.7 GB。
# 清理必须有依据（哪些能删、哪些是证据），不能靠"看着旧就删"。
set -u
cd /root/autodl-tmp/MAS || exit 1

echo "=== 盘况 ==="
df -h /root/autodl-tmp | tail -1
echo

# 死 embedder 的臂（2026-10-03 及之前跑的批次，向量通道从未真正运行）
DEAD_TAGS="ab_g0 ab_g1 ab_smoke_kb8 ab_smoke_v2 held_clean30 held_clean30_clean30_off held_clean30_clean30_on clean30_lam3 kbself_on kbself_on3 kbself_fix overlap15_lam3"

echo "=== 按 run 列表归因（MB）==="
printf '%-28s %6s %10s  %s\n' "run列表" "样本" "体积MB" "判定"
total_all=0
for f in reports/*_runs.txt; do
  [ -f "$f" ] || continue
  tag=$(basename "$f" _runs.txt)
  n=0; mb=0
  while IFS= read -r rid; do
    rid=$(echo "$rid" | tr -d ' \r')
    [ -z "$rid" ] && continue
    n=$((n+1))
    # run 列表里是相对路径（CVE-xxxx/<uuid>），不是裸 uuid —— 第一版按裸 uuid 拼路径，
    # 结果所有臂的体积都算成 0（工具静默给出错数字，比报错更坏）
    sz=$(du -sm -- "reports/analysis/$rid" 2>/dev/null | head -1 | cut -f1)
    [ -n "${sz:-}" ] && mb=$((mb+sz))
  done < "$f"
  verdict="✅有效证据"
  for t in $DEAD_TAGS; do [ "$tag" = "$t" ] && verdict="🗑 死 embedder（可清理）"; done
  printf '%-28s %6s %10s  %s\n' "$tag" "$n" "$mb" "$verdict"
  total_all=$((total_all+mb))
done
echo "  上述臂合计: ${total_all} MB"
echo

echo "=== 未覆盖的孤儿 run 目录（top 15，按体积）==="
# 收集所有清单里的 run id（相对路径 CVE-x/<uuid> 形式，与目录相对路径同构）
cat reports/*_runs.txt 2>/dev/null | tr -d ' \r' | grep -v '^$' | sort -u > /tmp/_known_runs.txt
find reports/analysis -mindepth 2 -maxdepth 2 -type d -printf '%P\n' 2>/dev/null | sort -u > /tmp/_all_runs.txt
comm -23 /tmp/_all_runs.txt /tmp/_known_runs.txt > /tmp/_orphan_runs.txt
echo "  孤儿 run 数: $(wc -l < /tmp/_orphan_runs.txt)"
while IFS= read -r rid; do
  [ -z "$rid" ] && continue
  du -sm -- "reports/analysis/$rid" 2>/dev/null | head -1
done < /tmp/_orphan_runs.txt | sort -rn | head -15
echo "  孤儿合计 MB: $(while IFS= read -r rid; do [ -z \"$rid\" ] && continue; du -sm -- \"reports/analysis/$rid\" 2>/dev/null | cut -f1; done < /tmp/_orphan_runs.txt | paste -sd+ | bc)"
echo
echo "=== 总账 ==="
du -sh reports/analysis 2>/dev/null
echo "DISK_INVENTORY_DONE"
