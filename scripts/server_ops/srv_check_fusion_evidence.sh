#!/bin/bash
# 生效性检查（可复用）：判断"门控融合开关到底有没有接上"。
#
# 用法: bash srv_check_fusion_evidence.sh <run 目录>
#
# ## 判据为什么这样定（第一版判错了一次，记在这里）
#
# 第一版要求产物里同时出现 `fusion_score` 和 `gate_branch`，结果在干净层上误判"未生效"。
# 真实原因是**代码结构**：`gate_branch` 只在"已经过了 F(x) 全部硬守卫、进入 DNF 分支"的
# 候选上才写；而干净层样本与库里**没有同文件条目**，绝大多数候选在守卫处就提前 return 了
# （`cross_file_mismatch` / `code_already_fixed`）⇒ 没有 `gate_branch` 是**正常现象**。
#
# 正确判据：
#   ① `fusion_score` / `fusion_semantic_term` 出现（说明融合代码跑了）；
#   ② `gate_formula` 里带融合分支（说明是这一版代码、开关打开）；
#   ③ 额外信息（不参与判定）：有多少候选真的走到 DNF、`gate_branch` 分布、融合分取值分布。
set -u
RUN="${1:?用法: srv_check_fusion_evidence.sh <run 目录>}"
[ -d "$RUN" ] || { echo "不是目录: $RUN"; exit 2; }

F_SCORE=$(grep -rl 'fusion_score' "$RUN" 2>/dev/null | wc -l)
F_TERM=$(grep -rl 'fusion_semantic_term' "$RUN" 2>/dev/null | wc -l)
F_BRANCH=$(grep -rl 'gate_branch' "$RUN" 2>/dev/null | wc -l)
FORMULA=$(grep -rho 'admit = F(x) & ( s(x)>=theta_s[^"]*' "$RUN" 2>/dev/null | head -1)

echo "  含 fusion_score         : $F_SCORE 个文件"
echo "  含 fusion_semantic_term : $F_TERM 个文件"
echo "  含 gate_branch          : $F_BRANCH 个文件（只有过了硬守卫的候选才有，可为 0）"
echo "  gate_formula            : ${FORMULA:-（未出现）}"
echo "  fusion_score 出现次数   : $(grep -rho '"fusion_score"' "$RUN" 2>/dev/null | wc -l)"
echo "  fusion_score 取值分布   :"
grep -rho '"fusion_score": [0-9.]*' "$RUN" 2>/dev/null | sort | uniq -c | sort -rn | head -6
echo "  候选判定分布            :"
grep -rho '"gating_decision": "[a-z_]*"' "$RUN" 2>/dev/null | sort | uniq -c | sort -rn | head -6
echo "  拒绝理由分布（前 4）    :"
grep -rho '"rejection_reason": "[a-z_]*"' "$RUN" 2>/dev/null | sort | uniq -c | sort -rn | head -4

OK=1
[ "$F_SCORE" -gt 0 ] || OK=0
case "$FORMULA" in *lambda*s_sem*) ;; *) OK=0 ;; esac
echo
if [ "$OK" = "1" ]; then
  echo "✅ 生效性检查通过：融合代码已运行，且公式含融合分支"
  exit 0
fi
echo "❌ 生效性检查不通过：融合代码没跑（开关没接上），A/B 不能往下做"
exit 2
