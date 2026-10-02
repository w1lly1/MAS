#!/bin/bash
# T2 跑完之后的**一次性**后处理：收 run 列表 → 完成度体检 → 生效性检查 → 与昨晚 Arm1 对照 → 未放行样本拒因。
#
# 为什么单独一个脚本：这几步必须**按顺序**做，且"下结论前先证明处理真的生效"
# 是本项目踩过坑之后定下的纪律（见《03》坑 7）。写成脚本能避免手抄命令时漏步。
#
# 用法: bash /root/autodl-tmp/srv_t2_after.sh
set -u
cd /root/autodl-tmp/MAS || exit 1

LOG=/root/autodl-tmp/arm1_t2_log.txt
NEW=/root/autodl-tmp/t2_runs.txt
OLD=/root/autodl-tmp/arm1_runs.txt
DB=infrastructure/database/mas.db

echo "############ 0) 先确认这一批是不是真的跑完了 ############"
echo "日志里的 partial 次数: $(grep -c '记为 partial' "$LOG" || true)"
echo "--- 批处理收尾几行 ---"
grep -E "成功|未确认完成|失败|批量分析" "$LOG" | tail -6

echo
echo "############ 1) 收集 run 列表（每个样本最终用哪个 run）############"
./venv/bin/python utils/experiments/make_run_list.py --logs "$LOG" --out "$NEW" 2>/dev/null | tail -6
echo "样本数: $(wc -l < "$NEW")"

echo
echo "############ 2) 生效性检查：补漏通道到底有没有用上语义 ############"
./venv/bin/python utils/experiments/check_treatment_applied.py \
  --pairs "T2_新库v2=$NEW" "Arm1_昨晚=$OLD" 2>/dev/null | tail -14

echo
echo "############ 3) 与昨晚 Arm1 对照（检索段 / 门控段分开量）############"
./venv/bin/python utils/experiments/compare_arms.py \
  --arms "Arm1_昨晚=$OLD" "T2_今晚=$NEW" --db "$DB" 2>/dev/null | head -12

echo
echo "############ 4) 未放行样本的拒因（重点看 CVE-2018-20854）############"
./venv/bin/python utils/experiments/explain_missed_admissions.py \
  --runs "$NEW" --db "$DB" --cves CVE-2018-20854 CVE-2017-17053 CVE-2018-6057 CVE-2002-2443 2>/dev/null \
  | head -60

echo
echo "############ 5) 关键单样本：CVE-2018-20854 的门控证据 ############"
./venv/bin/python - <<'PY' 2>/dev/null
import json, sqlite3
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS")
runs = {}
for line in Path("/root/autodl-tmp/t2_runs.txt").read_text(encoding="utf-8").splitlines():
    if "/" in line:
        cve, run = line.strip().split("/", 1)
        runs[cve] = run

con = sqlite3.connect("file:%s?mode=ro" % (ROOT / "infrastructure/database/mas.db"), uri=True)
own = {int(i) for i, t in con.execute("select id, title from issue_patterns")
       if (t or "").strip().upper() == "CVE-2018-20854"}
ci = {int(i): int(p) for i, p in con.execute("select id, pattern_id from curated_issues")}
con.close()

cve = "CVE-2018-20854"
run = runs.get(cve)
print("样本 %s -> run %s（自己的条目 id=%s）" % (cve, run, own))
d = ROOT / "reports/analysis" / cve / run / "second_pass/consolidated"
for f in sorted(d.glob("*_r2.json")):
    j = json.loads(f.read_text(encoding="utf-8"))
    print(" 文件:", j.get("file"))
    print(" new_findings 条数:", len(j.get("new_findings") or []))
    keys = ("retrieval_evidence", "gap_retrieval_evidence")
    for key in keys:
        for ev in (j.get(key) or []):
            for c in (ev.get("candidates") or []):
                sid = c.get("sqlite_id")
                resolved = ci.get(int(sid)) if str(c.get("channel")) == "curated_issue" else sid
                if resolved in own or (sid is not None and int(sid) in own):
                    print("   [%s] 候选 channel=%s sqlite_id=%s 决策=%s 拒因=%s matched=%s s(x)=%s"
                          % (key, c.get("channel"), sid, c.get("gating_decision"),
                             c.get("rejection_reason"), c.get("matched_fields"),
                             c.get("unified_structured_score")))
PY
echo
echo "############ 完成 ############"
