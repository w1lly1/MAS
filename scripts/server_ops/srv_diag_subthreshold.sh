#!/usr/bin/env bash
# 查"为什么未达阈值的候选一条都拿不到语义分"：按【有没有同文件字段】×【有没有层】拆开。
#
# 怀疑（要证）：curated 通道的候选只有在命中 `error_code_clone` 时才被赋予 `vector_layer`
# （见 `_build_candidate_from_curated_issue`：view_layer = code_pattern if error_code_clone else None）。
# 而"同文件但没针"的候选正是**既满足否决、又需要语义补位**的那一批 —— 可它们 vector_layer 为 None，
# 查不到整层余弦表 ⇒ 语义项仍为 0。
set -u
cd /root/autodl-tmp/MAS || exit 1
RUNS="${1:?用法: srv_diag_subthreshold.sh <runs 列表>}"

venv/bin/python - "$RUNS" <<'PY'
import json, sys, glob, os
from collections import Counter

THETA_S = 0.65
SAME = {"file_basename_anchor", "basename_match"}
runs = [ln.strip() for ln in open(sys.argv[1], encoding="utf-8") if ln.strip()]
grid = Counter()
layers = Counter()
stats_ok = Counter()
for rel in runs:
    for f in glob.glob(os.path.join("reports/analysis", rel, "second_pass", "**", "*_r2.json"),
                       recursive=True):
        try:
            j = json.loads(open(f, encoding="utf-8").read())
        except Exception:
            continue
        for key in ("retrieval_evidence", "gap_retrieval_evidence"):
            for b in (j.get(key) or []):
                for c in (b.get("candidates") or []):
                    if not isinstance(c, dict) or "fusion_score" not in c:
                        continue
                    if float(c.get("unified_structured_score") or 0) >= THETA_S:
                        continue                      # 只看"老规则不放行"的候选
                    mf = set(c.get("matched_fields") or [])
                    same = bool(mf & SAME)
                    layer = c.get("vector_layer")
                    term = float(c.get("fusion_semantic_term") or 0)
                    grid[("同文件" if same else "跨文件",
                          "有层" if layer else "无层",
                          "语义项>0" if term > 0 else "语义项=0")] += 1
                    if same:
                        layers[str(layer)] += 1
                        stats_ok[int(c.get("fusion_stats_n") or 0) > 0] += 1

print("未达 θ_s 的候选，按【同/跨文件 × 有无层 × 语义项】拆开：")
for k in sorted(grid, key=lambda x: -grid[x]):
    print("   %-6s %-4s %-8s : %d" % (k[0], k[1], k[2], grid[k]))
print()
print("其中【同文件】候选的 vector_layer 分布:", dict(layers))
print("其中【同文件】候选拿到整层分布的比例: 有 %d / 无 %d"
      % (stats_ok.get(True, 0), stats_ok.get(False, 0)))
print()
print("读法：若『同文件 × 无层』占绝大多数，说明**融合缺的不是语义原料，而是候选根本没被指派到某一层** ——")
print("      这类候选查不到整层余弦表，语义项只能记 0（修法与影响需单独评估）。")
PY
