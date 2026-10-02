#!/bin/bash
# 把 **v2 候选库**（= v1 + curated_issues 重排）落到线上，并逐项验证。
#
# ## 为什么这一步不需要重写向量
# `curated_issues` **不进向量索引**（索引里 800 条对象 = `issue_patterns` 200 × 4 层）。
# 而 v2 相对 v1 **只改了 `curated_issues.solution`**（167 行），`issue_patterns` 语义逐行相同
# ——这一点已在本地用 `utils/experiments/verify_rebuild_bundle.py` 证明过
# （198 行差异**全部**只差 `updated_at` 时间戳；向量包 535 条的 (条目,层) 与 layer_text 逐条对得上）。
# 因此本步只替换 SQLite 文件，索引保持上一轮切好的状态即可。
#
# 用法: bash /root/autodl-tmp/srv_apply_v2.sh
set -u
cd /root/autodl-tmp/MAS || exit 1

V2=/root/autodl-tmp/mas_rebuild_candidate_v2.db
LIVE=infrastructure/database/mas.db
STAMP=$(date +%Y%m%d_%H%M%S)

[ -f "$V2" ] || { echo "*** 缺少 $V2"; exit 1; }

echo "===== 1) 备份当前线上库 ====="
cp -f "$LIVE" "/root/autodl-tmp/mas.db.bak_beforev2_$STAMP"
echo "  -> /root/autodl-tmp/mas.db.bak_beforev2_$STAMP ($(stat -c%s "$LIVE") 字节)"

echo "===== 2) 写入 v2 ====="
cp -f "$V2" "$LIVE"
echo "  线上库 sha256 = $(sha256sum "$LIVE" | cut -c1-16)"
echo "  v2 库   sha256 = $(sha256sum "$V2" | cut -c1-16)"

echo "===== 3) 验证 ====="
./venv/bin/python - "$LIVE" "$V2" <<'PY'
import hashlib, sqlite3, sys

live, v2 = sys.argv[1], sys.argv[2]
EXPECT_IP = "ec1fb79bc31f"      # issue_patterns.solution 总指纹（v1 == v2，本地算过）
EXPECT_CI = "e18046ae3931"      # curated_issues.solution 总指纹（v2）


def fp(t):
    return hashlib.sha256(t.encode("utf-8")).hexdigest()[:12]


def read(path):
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    ip = "\n".join(str(r[0] or "") for r in
                   con.execute("select solution from issue_patterns order by id"))
    ci = "\n".join(str(r[0] or "") for r in
                   con.execute("select solution from curated_issues order by id"))
    sem = con.execute("select count(*) from issue_patterns "
                      "where trim(coalesce(llm_semantic,''))<>''").fetchone()[0]
    row = con.execute("select solution from curated_issues where id=163").fetchone()[0] or ""
    con.close()
    return fp(ip), fp(ci), sem, row

ok = True
live_ip, live_ci, live_sem, live_163 = read(live)
v2_ip, v2_ci, v2_sem, v2_163 = read(v2)

print("  线上 issue_patterns.solution 指纹 = %s（期望 %s）" % (live_ip, EXPECT_IP))
print("  线上 curated_issues.solution 指纹 = %s（期望 %s）" % (live_ci, EXPECT_CI))
print("  线上 llm_semantic 非空 = %d（期望 193）" % live_sem)
print("  线上 curated id=163 = %s" % live_163[:120])
print("  v2   curated id=163 = %s" % v2_163[:120])

for name, got, want in (("issue_patterns 指纹", live_ip, EXPECT_IP),
                        ("curated 指纹", live_ci, EXPECT_CI),
                        ("llm_semantic 非空数", live_sem, 193)):
    good = got == want
    ok = ok and good
    print("  [%s] %s: %s" % ("OK" if good else "NG", name, got))
# 关键判据：线上 curated 必须等于 v2（说明 v2 真的落库了）
same_as_v2 = (live_ci == v2_ci and live_163 == v2_163)
ok = ok and same_as_v2
print("  [%s] 线上 curated == v2（v2 真的落库）" % ("OK" if same_as_v2 else "NG"))
# 针的形态：重排后那条针不再重复出现
needle_dup = live_163.count("for (i = 0; i <= SERDES_MAX; i++) {")
print("  [%s] id=163 的粘针重复次数 = %d（重排后应为 1）"
      % ("OK" if needle_dup == 1 else "NG", needle_dup))
ok = ok and needle_dup == 1
raise SystemExit(0 if ok else 1)
PY
RC=$?

echo "===== 4) 索引是否仍然配套 ====="
echo "  curated_issues 不进向量索引（索引 800 条 = issue_patterns×4 层），"
echo "  且 issue_patterns 与 v1 语义逐行相同 ⇒ 无需重写向量。"
echo "  下面核对索引对象数与语义属性是否仍是切库后的状态："
./venv/bin/python -u scripts/server_ops/srv_kb_verify.py 2>/dev/null | grep -E "对象总数|llm_semantic 非空的对象数|属性列含"

if [ "$RC" -ne 0 ]; then
  echo "*** 验证失败：请回滚 cp /root/autodl-tmp/mas.db.bak_beforev2_$STAMP $LIVE"
  exit 1
fi
echo "OK：v2 已落库并通过全部核对"
