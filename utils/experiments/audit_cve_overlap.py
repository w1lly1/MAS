# -*- coding: utf-8 -*-
"""量化"不同种子建库之间 CVE 重叠"的现状（纯标准库，本地可跑）。

输出：
  · 可用 CVE 池大小（metadata + before + after 三处齐全）
  · 每个批次/清单文件的 CVE 集合、kb/held 划分
  · 两两重叠矩阵，以及"kb×kb"的重叠（这才是真正会造成知识库重复的部分）
"""
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
META, BEFORE, AFTER = DS / "metadata", DS / "before", DS / "after"

FILES = {
    "batch:test_400_batch(2024?)": ROOT / "论文/test_400_batch.json",
    "batch:test_400_error_batch": ROOT / "论文/test_400_error_batch.json",
    "batch:test_400_error_batch_remaining": ROOT / "论文/test_400_error_batch_remaining.json",
    "batch:test_200_batch": ROOT / "论文/test_200_batch.json",
    "batch:test_50_codebert": ROOT / "论文/test_50_codebert_batch.json",
    "manifest:negative_exp_400": ROOT / "reports/negative_exp_manifest_400.json",
    "manifest:negative_exp_400_error": ROOT / "reports/negative_exp_manifest_400_error.json",
    "manifest:five_fold": ROOT / "reports/five_fold_manifest.json",
    "manifest:held_out": ROOT / "reports/held_out_manifest.json",
}


def has_files(p: Path) -> bool:
    return p.is_dir() and any(f.is_file() for f in p.rglob("*"))


def pool():
    out = []
    for d in sorted(META.glob("CVE-*")):
        try:
            m = json.loads((d / "cve_metadata.json").read_text(encoding="utf-8"))
            cid = str(m.get("cve_id", "")).strip()
            if not cid or not str(m.get("summary", "")).strip():
                continue
            if not has_files(BEFORE / cid) or not has_files(AFTER / cid):
                continue
            out.append(cid)
        except Exception:
            continue
    return out


def read_set(p: Path):
    """返回 (全部CVE集合, kb集合, held集合, seed)。兼容 dict / list 两种写法。"""
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, list):
        d = {"rows": d}
    allc, kb, held, seed = set(), set(), set(), d.get("seed")
    rows = d.get("rows") or d.get("items") or []
    for r in rows:
        if not isinstance(r, dict):
            continue
        c = str(r.get("cve") or "").strip()
        if not c:
            continue
        allc.add(c)
        if str(r.get("role") or "") == "kb":
            kb.add(c)
        elif str(r.get("role") or "") == "held":
            held.add(c)
    if not allc:
        for c in d.get("kb_cves") or []:
            allc.add(c); kb.add(c)
        for c in d.get("held_cves") or []:
            allc.add(c); held.add(c)
    return allc, kb, held, seed


def main():
    P = pool()
    print("可用 CVE 池: %d" % len(P))
    sets = {}
    print("\n%-42s %6s %6s %6s %6s" % ("文件", "总数", "kb", "held", "seed"))
    for name, p in FILES.items():
        r = read_set(p)
        if not r:
            print("%-42s  (缺失)" % name)
            continue
        allc, kb, held, seed = r
        sets[name] = (allc, kb, held)
        print("%-42s %6d %6d %6d %6s" % (name, len(allc), len(kb), len(held), seed))

    print("\n=== 两两重叠（全部样本）===")
    print("%-42s %-42s %7s %7s %7s" % ("A", "B", "|A∩B|", "占比A", "占比B"))
    for a, b in combinations(sets, 2):
        A, B = sets[a][0], sets[b][0]
        if not A or not B:
            continue
        inter = A & B
        print("%-42s %-42s %7d %6.1f%% %6.1f%%" % (
            a, b, len(inter), 100 * len(inter) / len(A), 100 * len(inter) / len(B)))

    print("\n=== 关键：kb × kb 重叠（直接导致两个知识库出现相同 CVE）===")
    for a, b in combinations(sets, 2):
        A, B = sets[a][1], sets[b][1]
        if not A or not B:
            continue
        inter = A & B
        if inter:
            print("  %s(kb %d) × %s(kb %d) → 重叠 %d 条  样例: %s" % (
                a.split(":")[1], len(A), b.split(":")[1], len(B), len(inter),
                sorted(inter)[:5]))

    print("\n=== 若要求两次 400 全不相交，池子够不够 ===")
    print("  需要 2×400=800，池子 %d → %s" % (len(P), "够" if len(P) >= 800 else "不够"))
    print("  需要 5×400=2000 做 5 折全不相交 → %s" % ("够" if len(P) >= 2000 else "不够（5 折需减小每折规模）"))


if __name__ == "__main__":
    sys.exit(main())
