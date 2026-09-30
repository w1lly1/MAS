# -*- coding: utf-8 -*-
"""评测分层口径（唯一权威实现）—— 把「库外组」按文件是否在库拆成两层，并可复现地给出子类。

## 为什么要固化这个口径

评测批次里每个样本有一个 `role`：

* `kb`   = 该 CVE 自己的知识条目在库里 → 检索**应该**捞回它 → 用来测**召回率**
* `held` = 该 CVE 自己的条目不在库里 → 检索**不该**捞到东西 → 用来测**误报率**

但 `held` 组内部其实混着两类性质完全不同的样本：

1. **文件根本不在库里**（`held-pure`）
   检索还能命中，只有一个解释：方法在**无关代码**上产生了命中 → 这才是**真误报**。
2. **文件在库里，但库里那条记录属于另一个 CVE**（`held-same-file`）
   检索命中的是"同一个文件的另一个 CVE"。这既不是干净的误报（文件确实是同一个、
   位置常常也是同一处），也不是干净的召回（编号对不上）。**它是一个第三类现象**，
   混在一起算会同时污染两个指标。

实测（seed=2024 主实验）：8 个"误报"**全部**落在第 2 类，第 1 类是 0 个。
也就是说，那个 4.0% 的误报率，其字面含义与它给人的印象并不一致。

所以本模块把这件事**做成代码**，而不是靠事后人工分类：

* 分层只依赖两个**可复现**的数据源 → 知识库 SQLite + 数据集 metadata（都是文本，无 GPU、无随机性）
* 评测脚本直接调用它，于是**每一轮评测天然带上分层字段**，不需要回头补算
* 子类判定也只用 metadata（CWE / 漏洞分类文本 / 摘要），不依赖"再跑一遍 ingest 流水线"，
  因而任何人都能复算

## 术语（尽量用大白话）

* **文件**：指样本要分析的那个源码文件，例如 `libavcodec/mpeg4videodec.c`。
* **末两级路径**：`libavcodec/mpeg4videodec.c` 就是"末两级"（目录 + 文件名）。
  为什么不用裸文件名？因为本项目数据集里 `inode.c` / `core.c` / `socket.c` 这类重名极多，
  只比文件名会把 `fs/udf/inode.c` 和 `fs/overlayfs/inode.c` 当成同一个文件（实测出现过这个假阳性）。
  为什么不用末三级？数据集里的路径深度与知识库不一定一致，末三级过严会漏判。
* **兄弟条目**：同一个文件、不同 CVE 的库内条目。

## 用法

    from utils.eval_strata import build_strata
    st = build_strata(["CVE-2018-16422", "CVE-2014-6229"])
    st["CVE-2014-6229"]["group"]        # -> "held-pure" / "held-same-file" / "kb-self" / "kb-shared-file"
    st["CVE-2014-6229"]["subclass"]     # -> "A" / "B" / "C" / ""（仅 held-same-file 有值）

命令行：

    python utils/eval_strata.py --manifest reports/negative_exp_manifest_400_error.json
    python utils/eval_strata.py --eval reports/eval_400_error_v4.csv --out reports/strata.json
    python utils/eval_strata.py --validate-pairs reports/held_same_file_cross_cve_analysis.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT, normalize_key  # noqa: E402

DB = ROOT / "infrastructure/database/mas.db"
DS_ROOT = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"

# 分层标签（冻结，不要在别处另起名字）
G_KB_SELF = "kb-self"                # 该 CVE 自己的条目在库 → 召回样本
G_KB_SHARED = "kb-shared-file"       # 在库，但同文件还有别的 CVE 也入过库 → 召回侧混淆样本
G_HELD_SAME = "held-same-file"       # 条目不在库，但同文件在库 → 第三类现象
G_HELD_PURE = "held-pure"            # 条目不在库，同文件也不在库 → 真误报样本

SUBCLASS_LABEL = {
    "A": "同漏洞异号（同一个漏洞被编了两个号）",
    "B": "同文件 + 同类型 + 不同修复",
    "C": "同文件 + 不同类型",
    "?": "无法判定（缺 metadata）",
}

# 子类 A 的阈值：摘要词面相似度。取 0.5 是因为实测 A 类（真同漏洞异号）两侧摘要
# 词面几乎重合（同一次提交的两个编号），而非同漏洞的样本摘要词面相似度普遍 < 0.35。
SUMMARY_SIM_A = 0.5


# --------------------------------------------------------------------------- #
# 数据源读取
# --------------------------------------------------------------------------- #
def _files_of(cve: str, dataset_root: Path) -> List[str]:
    """样本目录下的源码文件名（只取 basename，与门控里的 basename 尺子一致）。"""
    d = dataset_root / "before" / cve
    if not d.exists():
        return []
    out = []
    for f in d.rglob("*"):
        try:
            if f.is_file() and f.suffix.lower() in SOURCE_EXT:
                out.append(f.name)
        except OSError:
            continue
    return sorted(set(out))


def meta_of(cve: str, dataset_root: Path) -> Dict[str, str]:
    """读数据集 metadata（CWE 编号 / 漏洞分类 / 摘要）。缺失返回空字典。"""
    p = dataset_root / "metadata" / cve / "cve_metadata.json"
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "cwe": str(d.get("cwe_id") or "").strip().upper(),
        "classification": str(d.get("vulnerability_classification") or "").strip().lower(),
        "summary": str(d.get("summary") or ""),
        "project": str(d.get("project") or ""),
    }


def load_kb(db: Path) -> Dict[str, object]:
    """读知识库：CVE 集合 + 按『末两级路径』索引的条目。"""
    con = sqlite3.connect(str(db))
    rows = con.execute(
        "select id, title, file_pattern, error_type, class_pattern, problematic_pattern, solution "
        "from issue_patterns"
    ).fetchall()
    con.close()
    by_cve: Dict[str, Dict[str, object]] = {}
    by_key2: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        rec = {
            "id": r[0],
            "cve": str(r[1] or "").strip().upper(),
            "file_pattern": str(r[2] or ""),
            "error_type": str(r[3] or ""),
            "class_pattern": str(r[4] or ""),
            "problematic_pattern": str(r[5] or ""),
            "solution": str(r[6] or ""),
        }
        if rec["cve"]:
            by_cve[rec["cve"]] = rec
        k2 = normalize_key(rec["file_pattern"], 2)
        if k2:
            by_key2.setdefault(k2, []).append(rec)
    return {"by_cve": by_cve, "by_key2": by_key2, "n": len(rows)}


_WORD_RE = re.compile(r"[A-Za-z_]\w{2,}")


def _tokens(text: str) -> set:
    return {t.lower() for t in _WORD_RE.findall(str(text or ""))}


def summary_similarity(a: str, b: str) -> float:
    """摘要词面 Jaccard 相似度（用于判『是否同一个漏洞被编了两个号』）。"""
    A, B = _tokens(a), _tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# --------------------------------------------------------------------------- #
# 主体
# --------------------------------------------------------------------------- #
def classify_one(
    cve: str,
    kb: Dict[str, object],
    dataset_root: Path = DS_ROOT,
) -> Dict[str, object]:
    """给一个 CVE 定性：属于哪一层、库里有没有同文件兄弟、held-same-file 时属于哪个子类。

    返回字段（冻结；评测脚本只读这些键，改名视为口径变更）：
      files            样本自身的源码文件（basename 列表）
      file_keys        上面这些文件的『末两级路径』
      in_kb_own        该 CVE 自己的条目是否在库（决定 kb / held）
      kb_siblings      同文件的库内条目（另一个 CVE）→ 列表
      group            分层标签
      file_shared      库里是否存在同文件条目（与 in_kb_own 独立）
      subclass         A / B / C / ""（仅 group=held-same-file 时有值）
      subclass_why     子类判定的依据（可直接写进报告）
      signals          {same_cwe, same_classification, summary_similarity}
    """
    cve = str(cve or "").strip().upper()
    by_cve = kb["by_cve"]           # type: ignore[index]
    by_key2 = kb["by_key2"]         # type: ignore[index]

    files = _files_of(cve, dataset_root)
    keys = sorted({normalize_key(f, 2) for f in files if normalize_key(f, 2)})
    own = by_cve.get(cve)           # type: ignore[union-attr]
    in_kb_own = own is not None

    # 同文件的库内条目（排除它自己那条）
    sibs: List[Dict[str, object]] = []
    seen_ids = set()
    for k in keys:
        for rec in by_key2.get(k, []):   # type: ignore[union-attr]
            if rec["id"] in seen_ids:
                continue
            if in_kb_own and rec["id"] == own["id"]:   # type: ignore[index]
                continue
            seen_ids.add(rec["id"])
            sibs.append(rec)
    # 稳态排序：编号小的在前，便于对照
    sibs.sort(key=lambda r: (str(r["cve"]), int(r["id"])))

    if in_kb_own:
        group = G_KB_SHARED if sibs else G_KB_SELF
    else:
        group = G_HELD_SAME if sibs else G_HELD_PURE

    out: Dict[str, object] = {
        "cve": cve,
        "files": files,
        "file_keys": keys,
        "in_kb_own": in_kb_own,
        "file_shared": bool(sibs),
        "kb_siblings": [
            {"id": r["id"], "cve": r["cve"], "file_pattern": r["file_pattern"]} for r in sibs
        ],
        "group": group,
        "subclass": "",
        "subclass_why": "",
        "signals": {},
    }
    if group != G_HELD_SAME:
        return out

    mine = meta_of(cve, dataset_root)
    best, best_sim, best_signals = None, -1.0, {}
    for r in sibs:
        sm = meta_of(str(r["cve"]), dataset_root)
        same_cwe = bool(mine.get("cwe") and sm.get("cwe") and mine["cwe"] == sm["cwe"])
        same_cls = bool(
            mine.get("classification") and sm.get("classification")
            and mine["classification"] == sm["classification"]
        )
        sim = summary_similarity(mine.get("summary", ""), sm.get("summary", ""))
        if sim > best_sim:
            best, best_sim = r, sim
            best_signals = {"same_cwe": same_cwe, "same_classification": same_cls,
                            "summary_similarity": round(sim, 4)}
    if best is None:
        out["subclass"] = "?"
        out["subclass_why"] = "库中无同文件兄弟（数据不一致）"
        return out

    same_cwe = bool(best_signals.get("same_cwe"))
    same_cls = bool(best_signals.get("same_classification"))
    if not mine:
        sc, why = "?", "样本缺 metadata（CWE/分类/摘要都取不到），无法判类型"
    elif same_cwe and same_cls and best_sim >= SUMMARY_SIM_A:
        sc = "A"
        why = ("同 CWE(%s) + 同分类(%s) + 摘要词面相似 %.2f → 判为同一个漏洞的两个编号"
               % (mine.get("cwe"), mine.get("classification"), best_sim))
    elif same_cwe or same_cls:
        sc = "B"
        why = ("同类型（CWE 同=%s / 分类同=%s）但摘要词面相似仅 %.2f → 判为同类型不同修复"
               % (same_cwe, same_cls, best_sim))
    else:
        sc = "C"
        why = ("CWE(%s vs %s) 与分类(%s vs %s) 都不同 → 判为同文件不同类型"
               % (mine.get("cwe"), meta_of(str(best["cve"]), dataset_root).get("cwe"),
                  mine.get("classification"),
                  meta_of(str(best["cve"]), dataset_root).get("classification")))
    out["subclass"] = sc
    out["subclass_why"] = why
    out["subclass_sibling"] = {"id": best["id"], "cve": best["cve"],
                               "file_pattern": best["file_pattern"]}
    out["signals"] = dict(best_signals)
    return out


def build_strata(
    cves: Iterable[str],
    db: Path = DB,
    dataset_root: Path = DS_ROOT,
) -> Dict[str, Dict[str, object]]:
    """批量定性。返回 {CVE: 定性字典}。"""
    kb = load_kb(Path(db))
    return {str(c).strip().upper(): classify_one(str(c), kb, Path(dataset_root))
            for c in cves if str(c).strip()}


def counts(strata: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """汇总：分层计数 + 子类计数。"""
    g: Dict[str, int] = {}
    sub: Dict[str, int] = {}
    for v in strata.values():
        g[v["group"]] = g.get(v["group"], 0) + 1
        if v["group"] == G_HELD_SAME:
            s = str(v["subclass"] or "?")
            sub[s] = sub.get(s, 0) + 1
    return {"groups": g, "subclasses": sub, "total": len(strata)}


def check_role_consistency(
    strata: Dict[str, Dict[str, object]],
    role_by_cve: Dict[str, str],
) -> Dict[str, object]:
    """**防误用守卫**：CSV/manifest 里写的 role，是否与"按当前知识库算出来的分类"一致。

    为什么必须有这道守卫：分层的分母是"按**当前**知识库算出来的 kb/held"，
    而 CSV 里的 role 是**那一轮运行当时**的批次标签。如果那轮跑的是**另一份知识库**
    （例如换了个随机种子重建的库），两者就会整体错位——此时算出来的分层数字毫无意义，
    而且会**静默地**看起来很像正常结果（每一层都有人、都有比例）。

    实测就踩到了：`eval_400_seed2025.csv` 的 role 是 200 kb / 200 held，
    但按当前库算出来是 69 kb / 331 held → 说明它对应的是另一份知识库，分层结果不可用。

    返回 {"n", "mismatch", "rate", "ok", "examples"}；`ok=False` 时调用方应当**拒绝出数**。
    """
    n = mismatch = 0
    examples: List[str] = []
    for cve, st in strata.items():
        role = str(role_by_cve.get(cve) or "").strip().lower()
        if not role:
            continue
        n += 1
        implied = "kb" if st["group"] in (G_KB_SELF, G_KB_SHARED) else "held"
        if role != implied:
            mismatch += 1
            if len(examples) < 6:
                examples.append("%s(标签=%s, 当前库=%s)" % (cve, role, implied))
    rate = mismatch / n if n else 0.0
    return {"n": n, "mismatch": mismatch, "rate": rate, "ok": rate <= 0.05,
            "examples": examples}


def warn_if_inconsistent(info: Dict[str, object], source: str = "") -> bool:
    """打印一致性结论。返回 True 表示口径可用，False 表示应拒绝出数。"""
    if not info["n"]:
        return True
    if info["ok"]:
        print("  [口径自检] 标签与当前知识库一致（%d/%d 不符，阈值 5%%）" % (
            info["mismatch"], info["n"]))
        return True
    print("  " + "!" * 72)
    print("  [口径自检失败] %s 的 role 标签与当前知识库**整体错位**：" % (source or "该批次"))
    print("      不符 %d / %d = %.1f%%" % (info["mismatch"], info["n"], 100 * info["rate"]))
    print("      样例: %s" % "; ".join(info["examples"]))
    print("    含义：那一轮跑的是**另一份知识库**，本模块按当前库算的分层结果不可用。")
    print("    做法：换用那一轮对应的知识库（--db）后重算，不要直接引用下面这些数字。")
    print("  " + "!" * 72)
    return False


# --------------------------------------------------------------------------- #
# 与「重跑 ingest 流水线」的旧口径互校
# --------------------------------------------------------------------------- #
def validate_pairs(pairs_json: Path, db: Path = DB, dataset_root: Path = DS_ROOT) -> None:
    """把本模块的 metadata 口径，和旧脚分析_held_same_file.py 的流水线口径做交叉表。

    旧口径的 A 类判据是"修复文本几乎相同（Jaccard > 0.95）"，代价是要为每个样本重跑一遍
    ingest 流水线；本模块改用 metadata，代价是零。若两者一致性高，就可以放心用便宜的那个。
    """
    data = json.loads(Path(pairs_json).read_text(encoding="utf-8"))
    pairs = data.get("pairs") or []
    if not pairs:
        print("对照文件里没有 pairs，跳过")
        return
    best: Dict[str, Dict[str, object]] = {}
    for p in pairs:   # 每个 held CVE 可能对多个兄弟，取修复相似度最高的那个（与主脚本一致）
        c = p["held_cve"]
        if c not in best or p["solution_similarity"] > best[c]["solution_similarity"]:
            best[c] = p
    st = build_strata(best.keys(), db, dataset_root)
    print("=" * 78)
    print("口径互校：metadata 口径（便宜） × 流水线口径（贵）")
    print("=" * 78)
    print("  %-4s %-14s %-10s %-10s %s" % ("CVE", "旧口径子类", "新口径", "修复相似度", "摘要相似度"))
    tab: Dict[str, Dict[str, int]] = {}
    for cve, p in sorted(best.items()):
        old = ("A" if p["solution_similarity"] > 0.95
               else "B" if (p["same_cwe"] or p["same_classification"] or p["same_error_type"])
               else "C")
        new = str(st[cve]["subclass"] or "?")
        sim = float(st[cve].get("signals", {}).get("summary_similarity") or 0.0)
        tab.setdefault(old, {})
        tab[old][new] = tab[old].get(new, 0) + 1
        print("  %-4s %-14s %-10s %-10.3f %.3f" % (cve, old, new, p["solution_similarity"], sim))
    print("\n  交叉表（行=旧口径，列=新口径）:")
    for old, row in sorted(tab.items()):
        print("    旧 %s: %s" % (old, ", ".join("%s×%d" % (k, v) for k, v in sorted(row.items()))))
    agree = sum(v for old, row in tab.items() for k, v in row.items() if old == k)
    print("  完全一致 %d / %d = %.1f%%" % (agree, len(best), 100 * agree / max(1, len(best))))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cves_from(args: argparse.Namespace) -> List[str]:
    if args.cves:
        return [c.strip().upper() for c in args.cves.split(",") if c.strip()]
    if args.manifest:
        d = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        rows = d.get("rows") or d.get("items") or []
        return [str(r.get("cve") or "").strip().upper() for r in rows if r.get("cve")]
    if args.eval:
        with open(args.eval, encoding="utf-8") as fh:
            return [str(r.get("cve") or "").strip().upper() for r in csv.DictReader(fh)]
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description="评测分层口径（唯一权威实现）")
    ap.add_argument("--cves", help="逗号分隔的 CVE 列表")
    ap.add_argument("--manifest", type=Path, help="批次/结果 manifest JSON（读 rows/items 里的 cve）")
    ap.add_argument("--eval", type=Path, help="逐 CVE 评测 CSV")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--dataset-root", type=Path, default=DS_ROOT)
    ap.add_argument("--out", type=Path, default=None, help="分层明细 JSON")
    ap.add_argument("--validate-pairs", type=Path, default=None)
    ap.add_argument("--show", type=int, default=20, help="打印前 N 条明细（0=不打印）")
    args = ap.parse_args()

    if args.validate_pairs:
        validate_pairs(args.validate_pairs, args.db, args.dataset_root)
        return

    cves = _cves_from(args)
    if not cves:
        raise SystemExit("请用 --cves / --manifest / --eval 指定样本")
    st = build_strata(cves, args.db, args.dataset_root)
    c = counts(st)

    print("=" * 92)
    print("评测分层口径  样本 %d 个   知识库 %s" % (len(st), args.db))
    print("=" * 92)
    order = [G_KB_SELF, G_KB_SHARED, G_HELD_PURE, G_HELD_SAME]
    desc = {
        G_KB_SELF: "条目在库、同文件无别的条目 → 干净的召回样本",
        G_KB_SHARED: "条目在库、但同文件还有别的 CVE → 召回侧混淆样本",
        G_HELD_PURE: "条目不在库、同文件也不在库 → 真误报样本",
        G_HELD_SAME: "条目不在库、但同文件在库（属别的 CVE）→ 第三类现象",
    }
    for k in order:
        n = c["groups"].get(k, 0)
        if n or k in (G_HELD_PURE, G_HELD_SAME):
            print("  %-16s %4d   %s" % (k, n, desc[k]))
    if c["subclasses"]:
        print("\n  held-same-file 子类:")
        for k in ("A", "B", "C", "?"):
            n = c["subclasses"].get(k, 0)
            if n:
                print("    %s  %4d   %s" % (k, n, SUBCLASS_LABEL[k]))

    if args.show:
        print("\n  %-16s %-16s %-3s %-28s %s" % ("CVE", "分层", "子", "文件（末两级）", "依据"))
        for cve, v in list(st.items())[: args.show]:
            print("  %-16s %-16s %-3s %-28s %s" % (
                cve, v["group"], v["subclass"] or "-",
                (v["file_keys"][0] if v["file_keys"] else "(无源文件)")[:28],
                (v["subclass_why"] or ("兄弟: " + ",".join(
                    s["cve"] for s in v["kb_siblings"][:2]) if v["kb_siblings"] else "库中无同文件"[:40]))[:60]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "db": str(args.db), "dataset_root": str(args.dataset_root),
            "criterion": {
                "group": "kb-self/kb-shared-file 由『该 CVE 自己的条目是否在库(title 列)』定；"
                         "held-pure/held-same-file 再按『末两级路径是否在库』细分",
                "path_granularity": "末两级路径（目录+文件名），避免同名文件假阳性",
                "subclass": "A=同CWE+同分类+摘要词面相似>=%.1f；B=同CWE或同分类；C=都不相同"
                            % SUMMARY_SIM_A,
                "subclass_source": "数据集 metadata（无需重跑 ingest 流水线）",
            },
            "counts": c, "items": st,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n明细已写出: %s" % args.out)


if __name__ == "__main__":
    main()
