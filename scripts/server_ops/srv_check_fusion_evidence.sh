#!/bin/bash
# 生效性检查（可复用）：判断"门控融合开关到底有没有接上"。
#
# 用法: bash srv_check_fusion_evidence.sh <run 目录>
#
# ## 判据为什么这样定（前两版都判错过，记在这里）
#
# 第一版要求同时出现 `fusion_score` 与 `gate_branch` → 在干净层误判"未生效"：
#   `gate_branch` 只在"已过 F(x) 全部硬守卫、进入 DNF 分支"的候选上才写，
#   干净层绝大多数候选在守卫处就 return 了 ⇒ 没有它是**正常现象**。
# 第二版改成"公式文本里必须含 lambda" → 又在库里臂误判：
#   用 `grep -o` 抓公式**会抓到旧公式**（同一份产物里两种写法可能都在），
#   拿命令行文本当判据本身就很脆。
#
# 现在改成**按候选数**判（与 `srv_diag_fusion_terms.sh` 同一套读法）：
#   ① 有候选带 `fusion_score`；② 至少一条候选的 `gate_formula` 含融合分支；
#   ③ 另外报告有没有真正走新分支放行（`gate_branch=fused_semantic`）—— 这才是有收益的证据。
set -u
RUN="${1:?用法: srv_check_fusion_evidence.sh <run 目录>}"
[ -d "$RUN" ] || { echo "不是目录: $RUN"; exit 2; }
cd /root/autodl-tmp/MAS || exit 1
RUNREL="${RUN#reports/analysis/}"
RUNREL="${RUNREL%/}"
echo "$RUNREL" > /tmp/one_run_check.txt

echo "  文件层面：含 fusion_score $(grep -rl 'fusion_score' "$RUN" 2>/dev/null | wc -l) 个文件"
echo "  候选层面（权威判据）："
venv/bin/python - /tmp/one_run_check.txt <<'PY'
import json, sys, glob, os
from collections import Counter
rel = open(sys.argv[1], encoding="utf-8").read().strip()
n_fs = n_new = n_old = 0
branch = Counter()
terms = Counter()
for f in glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*_r2.json"),
                  recursive=True):
    try:
        j = json.loads(open(f, encoding="utf-8").read())
    except Exception:
        continue
    for key in ("retrieval_evidence", "gap_retrieval_evidence"):
        for b in (j.get(key) or []):
            for c in (b.get("candidates") or []):
                if not isinstance(c, dict):
                    continue
                if "fusion_score" in c:
                    n_fs += 1
                    terms["语义项>0" if float(c.get("fusion_semantic_term") or 0) > 0
                          else "语义项=0"] += 1
                    if c.get("gate_branch"):
                        branch[str(c.get("gate_branch"))] += 1
                gf = c.get("gate_formula")
                if gf:
                    n_new += 1 if "lambda" in gf else 0
                    n_old += 0 if "lambda" in gf else 1
print("    有 fusion_score 的候选       : %d" % n_fs)
print("    gate_formula 含融合分支的候选: %d（旧公式 %d）" % (n_new, n_old))
print("    gate_branch 分布             : %s" % (dict(branch) or "（无）"))
print("    语义项取值                   : %s" % (dict(terms) or "（无）"))
ok = (n_fs > 0) and (n_new > 0)
print("    ⇒ %s" % ("✅ 生效（融合代码在跑）" if ok else "❌ 未生效（开关没接上）"))
sys.exit(0 if ok else 2)
PY
