# -*- coding: utf-8 -*-
"""造一个**可以在真机上直接跑**的 held（库外）批次，并**在本地证明这批样本确实不在 KB 里**。

## 为什么需要它

论文里两种样本必须分清：

* **kb-self**：被分析样本**自己的条目**就在被测系统查询的那个知识库里 → 检索理应能捞回它自己
  → 用来测**召回**（"自己捞自己"）。
* **held**：样本**不在** KB 里 → 检索捞不到自己的记录 → 用来测**泛化 / 误报是否增加**。

`role` 是手写标签，而 KB 是另一个脚本用某个随机种子建的，两者没有一致性校验；
本项目实测出现过"标成 held 的样本其实在库里"（见 `utils/kb_coverage.py` 的说明）。
所以 held 批次不能只靠标签，必须**对 KB 做缺席证明**。

## 本脚本做什么

1. 从 `reports/batch_summary_seed2025.csv` 取 `role=held` 的 CVE（应为 200 个），
   并与 `reports/held_out_manifest.json`、`reports/held_seed2025_v5_shard*.jsonl` 交叉核对，
   说明三者关系与交集（**不假设它们一致**）。
2. **KB 缺席证明**（核心产出），对每个 held CVE 跑两族**相互独立**的证据：

   * **A 族：SQLite（`issue_patterns` + `curated_issues`，先打印 `PRAGMA table_info` 看真实列名）**
     - `A1` 权威口径：`issue_patterns.title == CVE`（本项目的 kb/held 判定口径，见 `kb_coverage.cve_in_kb`）
     - `A2` 提及口径：CVE 号出现在两表**任意文本列**里（弱证据：可能只是别的条目描述里提到了它）
     - `A3` 文件口径：样本源文件归一化后的 **basename / 末两级路径** 落在 `issue_patterns.file_pattern`
       与 `curated_issues.file_path` 集合里（**可能是库里别的 CVE 的同名文件**，见下）
   * **B 族：`reports/weaviate_kb_dump_today.jsonl`（先打印第一条对象的 key 列表看真实结构）**
     - `B1` CVE 号出现在 dump 对象的**任意字符串字段**里（dump 只有 800 条向量、没有 title 字段，
       所以这是向量侧唯一能独立于 SQLite 的"号码"证据）
     - `B2` 文件口径：样本文件的 basename / 末两级路径落在 dump 对象自带的 `file_pattern` 集合里
     - `B3` 一致性链路：`dump.sqlite_id → issue_patterns.title == CVE`
       （**注意：这不是独立证据**，它的号码来自 SQLite，只用来验证"向量索引确实覆盖了这个条目"）

   **口径说明（很重要）**：`A3` / `B2` 命中的是"库里存在同路径文件"，**不等于**"这个样本自己的条目在库里"。
   本项目里重名文件极常见（`inode.c` / `fork.c` / `scm.c` …），`kb_coverage.py` 明确警告过 basename 级别的假阳性。
   所以"held 是否成立"由 **A1/A2 + B1** 决定，`A3/B2` 单独列出，作为**混淆风险**（默认还会把有同文件风险的样本
   从批次里剔除，见 `--overlap-rule`）。

   `in_kb_db = A1 or A2` 是**保守口径**：只要 KB 里出现过这个号码（哪怕只是别条目的描述里提到它），
   就记成"可能入库"，宁可误报也不漏报 —— 不过要记住 A2 命中 ≠ 条目真的入库，
   所以清单里 A1/A2 分开记录，`A1` 才是真正的"自己捞自己"依据。
   本机实测：held 200 个的 A1/A2/B1 **全为 0**，反向对照的 kb-self 则 A1 命中。

3. 从 200 个 held 里**确定性地**选一批（默认 30 个）适合跑批的：目录齐全（before/after/metadata 都在）、
   有源文件、源文件总字节落在区间内、按字节升序（与 `smoke_kb30.json` 同一套挑法，便于两批对照），
   生成：

   * `utils/experiments/held_kb30.json` —— 批处理配置（schema 与 `smoke_kb30.json` 一致，`target_dir`
     默认用真机（GPU 服务器）上的路径约定 `/root/autodl-tmp/MAS/...`，与 `smoke_kb30.json` 一样）
   * `reports/held_batch_manifest.json` —— 200 个 held 的逐样本清单（含两条证据、是否入批）

4. **自验**（脚本内建，跑一次就有）：
   * 断言入批样本的 `A1/A2/B1` 命中数 = 0；
   * **反向对照**：拿 `reports/arm1_runs.txt` 里 3 个已知 kb-self 的 CVE 跑**同一套匹配逻辑**，
     必须**命中** KB。若 kb-self 都测不出命中，说明匹配逻辑坏了 → 直接以非零码退出。

## 用法

    # 生成（默认 30 个、seed=2025，输出 utils/experiments/held_kb30.json）
    python -X utf8 utils/experiments/make_held_batch.py

    # 换规模/种子/输出
    python -X utf8 utils/experiments/make_held_batch.py --n 50 --seed 2025 \
        --out utils/experiments/held_kb50.json

    # 想用本地 Windows 路径写 target_dir（本机 CPU 复刻跑批用）
    python -X utf8 utils/experiments/make_held_batch.py --local-paths

    # 真机上跑这批（GPU 服务器，MAS 根目录下）
    python mas.py batch --config utils/experiments/held_kb30.json

输出是**确定的**：固定 seed、排序稳定、不写时间戳，重复运行结果逐字节一致。

本脚本**只读**既有文件，只新增上面两个产物，不修改任何既有文件。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT, normalize_key  # noqa: E402

REPORTS = ROOT / "reports"
EXPERIMENTS = ROOT / "utils" / "experiments"

DS_REL = "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"
DS = ROOT / DS_REL
SERVER_DS = "/root/autodl-tmp/MAS/" + DS_REL            # 真机（GPU 服务器）上的同一目录

CSV = REPORTS / "batch_summary_seed2025.csv"
HELD_MANIFEST = REPORTS / "held_out_manifest.json"
SHARD_GLOB = "held_seed2025_v5_shard*.jsonl"
KB_DUMP = REPORTS / "weaviate_kb_dump_today.jsonl"
DEFAULT_DB = REPORTS / "mas_live.db"
CONTROL_RUNS = REPORTS / "arm1_runs.txt"

CVE_RE = re.compile(r"CVE-\d{4}-\d{3,7}", re.I)
EVIDENCE_CAP = 4            # 每条证据最多保留几条样例，避免产物膨胀


# --------------------------------------------------------------------------
# 读输入
# --------------------------------------------------------------------------
def load_held_roles(csv_path: Path) -> tuple[list[str], list[str]]:
    """从批次汇总 CSV 取 role=held / role=kb 的 CVE（去重、排序稳定）。"""
    text = csv_path.read_text(encoding="utf-8-sig")
    rows = list(csv.DictReader(text.splitlines()))
    held = sorted({(r["cve"] or "").strip().upper() for r in rows if r.get("role") == "held"})
    kb = sorted({(r["cve"] or "").strip().upper() for r in rows if r.get("role") == "kb"})
    held = [c for c in held if c]
    kb = [c for c in kb if c]
    return held, kb


def load_v5_shards() -> list[str]:
    """读 held_seed2025_v5_shard*.jsonl，返回里面出现过的 CVE（去重排序）。"""
    out: set[str] = set()
    for p in sorted(REPORTS.glob(SHARD_GLOB)):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.add(str(json.loads(line).get("cve") or "").strip().upper())
    return sorted(c for c in out if c)


def load_v5_results() -> dict[str, dict]:
    """上一轮 held 跑批（v5 门控）的逐 CVE 结果，作为参考写进清单（不影响选择）。"""
    out: dict[str, dict] = {}
    for p in sorted(REPORTS.glob(SHARD_GLOB)):
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            cve = str(o.get("cve") or "").strip().upper()
            if cve:
                out[cve] = {"new_findings": o.get("new_findings"), "fp": o.get("fp")}
    return out


# --------------------------------------------------------------------------
# A 族证据：SQLite
# --------------------------------------------------------------------------
def _columns(con: sqlite3.Connection, table: str) -> list[tuple]:
    return con.execute("pragma table_info(%s)" % table).fetchall()


def build_sqlite_evidence(db: Path, verbose: bool = True) -> dict:
    """读 KB SQLite，建三套索引：CVE 身份 / CVE 文本提及 / 文件身份。"""
    con = sqlite3.connect(str(db))
    tables = ("issue_patterns", "curated_issues")

    cols: dict[str, list[str]] = {}
    for t in tables:
        info = _columns(con, t)
        cols[t] = [c[1] for c in info]
        if verbose:
            n = con.execute("select count(*) from %s" % t).fetchone()[0]
            print("  [PRAGMA] %s（%d 行）:" % (t, n))
            for c in info:
                print("      cid=%-2s %-20s %-14s pk=%s" % (c[0], c[1], c[2], c[5]))

    # 文本列 = 声明类型含 TEXT / VARCHAR / CHAR
    text_cols: dict[str, list[str]] = {}
    for t in tables:
        info = _columns(con, t)
        text_cols[t] = [c[1] for c in info
                        if any(k in str(c[2]).upper() for k in ("TEXT", "CHAR", "CLOB"))]

    # 文件身份列：只认真正表示"文件"的列名（不臆测其它列）
    file_cols = {"issue_patterns": ["file_pattern"], "curated_issues": ["file_path"]}
    for t, cs in file_cols.items():
        missing = [c for c in cs if c not in cols[t]]
        if missing:
            raise SystemExit("表 %s 缺少预期的文件列 %s（列名变了？先看 PRAGMA 输出）" % (t, missing))

    title_index: dict[str, list] = {}          # CVE -> [issue_patterns.id]
    mention_index: dict[str, list] = {}        # CVE -> [(table, rowid, col)]
    file_index: dict[str, dict[str, list]] = {"k1": {}, "k2": {}}
    n_rows = 0

    for t in tables:
        sel = ", ".join(cols[t])
        for row in con.execute("select %s from %s" % (sel, t)):
            n_rows += 1
            rec = dict(zip(cols[t], row))
            rid = rec.get("id")
            # A1：权威身份（title 就是 CVE 编号）
            if t == "issue_patterns":
                tv = str(rec.get("title") or "").strip().upper()
                if tv:
                    title_index.setdefault(tv, []).append(rid)
            # A2：任意文本列里的 CVE 号
            for c in text_cols[t]:
                v = rec.get(c)
                if not isinstance(v, str) or "CVE-" not in v.upper():
                    continue
                for tok in {m.group(0).upper() for m in CVE_RE.finditer(v)}:
                    bucket = mention_index.setdefault(tok, [])
                    if len(bucket) < EVIDENCE_CAP:
                        bucket.append({"table": t, "row_id": rid, "column": c,
                                       "value": v.strip()[:120]})
            # A3：文件身份（basename / 末两级路径）
            for c in file_cols[t]:
                v = rec.get(c)
                if not isinstance(v, str) or not v.strip():
                    continue
                for depth, bucket in ((1, "k1"), (2, "k2")):
                    k = normalize_key(v, depth)
                    if not k:
                        continue
                    file_index[bucket].setdefault(k, [])
                    if len(file_index[bucket][k]) < EVIDENCE_CAP:
                        file_index[bucket][k].append(
                            {"table": t, "row_id": rid, "column": c, "value": v.strip()})
    con.close()
    return {"db": str(db), "rows": n_rows, "title_index": title_index,
            "mention_index": mention_index, "file_index": file_index}


# --------------------------------------------------------------------------
# B 族证据：Weaviate 索引 dump
# --------------------------------------------------------------------------
def build_vector_evidence(dump: Path, verbose: bool = True) -> dict:
    """读向量库 dump（每行一个对象的 properties），建 CVE 文本提及 / 文件身份 / sqlite_id 链路。"""
    mention_index: dict[str, list] = {}
    file_index: dict[str, dict[str, list]] = {"k1": {}, "k2": {}}
    sqlite_ids: dict[int, int] = {}
    n = 0
    first_keys: list[str] = []
    empty_file_pattern = 0

    with dump.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if not first_keys:
                first_keys = sorted(o.keys())
                if verbose:
                    print("  [dump 第一条对象的 key] %s" % first_keys)
            n += 1
            sid = o.get("sqlite_id")
            if isinstance(sid, int):
                sqlite_ids[sid] = sqlite_ids.get(sid, 0) + 1
            for k, v in o.items():
                if not isinstance(v, str):
                    continue
                if "CVE-" in v.upper():
                    for tok in {m.group(0).upper() for m in CVE_RE.finditer(v)}:
                        bucket = mention_index.setdefault(tok, [])
                        if len(bucket) < EVIDENCE_CAP:
                            bucket.append({"field": k, "sqlite_id": sid,
                                           "value": v.strip()[:120]})
            fp = o.get("file_pattern")
            if isinstance(fp, str) and fp.strip():
                for depth, bucket in ((1, "k1"), (2, "k2")):
                    kk = normalize_key(fp, depth)
                    if not kk:
                        continue
                    file_index[bucket].setdefault(kk, [])
                    if len(file_index[bucket][kk]) < EVIDENCE_CAP:
                        file_index[bucket][kk].append({"field": "file_pattern",
                                                       "sqlite_id": sid, "value": fp.strip()})
            else:
                empty_file_pattern += 1
    if verbose:
        print("  [dump] 对象 %d 个，覆盖 sqlite_id %d 个（min=%s max=%s），"
              "file_pattern 为空的 %d 个"
              % (n, len(sqlite_ids), min(sqlite_ids) if sqlite_ids else None,
                 max(sqlite_ids) if sqlite_ids else None, empty_file_pattern))
    return {"dump": str(dump), "objects": n, "first_keys": first_keys,
            "mention_index": mention_index, "file_index": file_index,
            "sqlite_ids": sqlite_ids, "empty_file_pattern": empty_file_pattern}


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------
def _rel(p: Path) -> str:
    """尽量写成仓库内相对路径（打印/落盘都用它，换机器也看得懂）。"""
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p.resolve())


def sample_files(cve: str) -> list[str]:
    """样本 before 目录下的源文件名（数据集转义名，如 `fs__overlayfs__inode.c`）。"""
    d = DS / "before" / cve
    if not d.is_dir():
        return []
    return sorted(f.name for f in d.rglob("*")
                  if f.is_file() and f.suffix.lower() in SOURCE_EXT)


def src_bytes(cve: str) -> int:
    d = DS / "before" / cve
    if not d.is_dir():
        return 0
    return sum(f.stat().st_size for f in d.rglob("*")
               if f.is_file() and f.suffix.lower() in SOURCE_EXT)


def _sample_keys(files: list[str]) -> dict[str, set[str]]:
    return {"k1": {normalize_key(f, 1) for f in files} - {""},
            "k2": {normalize_key(f, 2) for f in files} - {""}}


def judge(cve: str, files: list[str], sq: dict, vec: dict,
          title_to_ids: dict[str, list] | None = None) -> dict:
    """对单个 CVE 跑 A/B 两族证据，返回布尔 + 命中证据。"""
    cve = cve.strip().upper()
    keys = _sample_keys(files)

    # --- A 族：SQLite
    a1_ids = sq["title_index"].get(cve, [])
    a2 = sq["mention_index"].get(cve, [])
    a3 = {g: sorted({e["value"] for k in keys[g]
                     for e in sq["file_index"][g].get(k, [])})
          for g in ("k1", "k2")}

    # --- B 族：Weaviate dump
    b1 = vec["mention_index"].get(cve, [])
    b2 = {g: sorted({e["value"] for k in keys[g]
                     for e in vec["file_index"][g].get(k, [])})
          for g in ("k1", "k2")}
    link_ids = sorted(i for i in (title_to_ids or {}).get(cve, []) if i in vec["sqlite_ids"])

    return {
        "cve": cve,
        "files": files,
        "n_files": len(files),
        # 权威：该样本自己的条目在不在库里
        "A1_cve_is_title_in_db": bool(a1_ids),
        "A1_issue_pattern_ids": sorted(a1_ids),
        # 弱：CVE 号出现在库文本里（可能只是别的条目提到了它）
        "A2_cve_mentioned_in_db_text": bool(a2),
        "A2_evidence": a2,
        # 文件级：库里有没有同路径文件（可能是别的 CVE 的）
        "A3_same_basename_in_db": sorted(a3["k1"]),
        "A3_same_relpath2_in_db": sorted(a3["k2"]),
        # 向量侧独立的"号码"证据
        "B1_cve_in_vector_text": bool(b1),
        "B1_evidence": b1,
        # 向量侧文件级
        "B2_same_basename_in_vectors": sorted(b2["k1"]),
        "B2_same_relpath2_in_vectors": sorted(b2["k2"]),
        # 链路一致性（号码来自 SQLite，不算独立证据）
        "B3_vector_link_to_title": bool(link_ids),
        "B3_matched_sqlite_ids": link_ids,
    }


def in_kb_db(j: dict) -> bool:
    """A 族判定：KB 里出现过这个样本的号码（权威口径 A1 + 提及口径 A2）。"""
    return bool(j["A1_cve_is_title_in_db"] or j["A2_cve_mentioned_in_db_text"])


def in_kb_vectors(j: dict) -> bool:
    """B 族判定：向量索引里出现过这个样本的号码（B1，独立于 SQLite）。"""
    return bool(j["B1_cve_in_vector_text"])


def overlap_keys(j: dict, rule: str) -> list[str]:
    """按 rule 给出"库里有同路径文件"的命中项（用于混合风险过滤）。"""
    if rule == "none":
        return []
    field = "k1" if rule == "basename" else "k2"
    return sorted(set(j["A3_same_%s_in_db" % ("basename" if field == "k1" else "relpath2")])
                  | set(j["B2_same_%s_in_vectors" % ("basename" if field == "k1" else "relpath2")]))


def load_project(cve: str) -> str:
    """从 samples 的 metadata/<CVE>/cve_metadata.json 取 project（本地就有，无需联网）。"""
    p = DS / "metadata" / cve / "cve_metadata.json"
    if not p.is_file():
        return ""
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("project") or "")
    except Exception:
        return ""


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="造 held 批次并做 KB 缺席证明")
    ap.add_argument("--csv", type=Path, default=CSV, help="批次汇总 CSV（role 列）")
    ap.add_argument("--held-manifest", type=Path, default=HELD_MANIFEST)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="被测系统实际查询的 KB SQLite")
    ap.add_argument("--dump", type=Path, default=KB_DUMP, help="向量索引 dump jsonl")
    ap.add_argument("--n", type=int, default=30, help="入批样本数（默认 30）")
    ap.add_argument("--seed", type=int, default=2025, help="随机种子（random 模式下用它抽样）")
    ap.add_argument("--select", choices=["smallest", "random"], default="smallest",
                    help="smallest=源文件总字节升序取前 n（默认，与 smoke_kb30 同口径）；"
                         "random=在候选池里用 seed 随机抽 n 个")
    ap.add_argument("--min-kb", type=int, default=4, help="源文件大小下限（KB），与 smoke_kb30 一致")
    ap.add_argument("--max-kb", type=int, default=400, help="源文件大小上限（KB），与 smoke_kb30 一致")
    ap.add_argument("--overlap-rule", choices=["relpath2", "basename", "none"], default="relpath2",
                    help="库里存在同路径文件时的处理：relpath2=按末两级剔除（默认）；"
                         "basename=按裸文件名剔除（更严）；none=不剔除（会把混淆风险带进批次）")
    ap.add_argument("--local-paths", action="store_true",
                    help="target_dir 写本地绝对路径（本机 CPU 复刻跑批用）；"
                         "默认写真机路径 /root/autodl-tmp/MAS/...（与 smoke_kb30.json 一致）")
    ap.add_argument("--out", type=Path, default=None,
                    help="批处理配置输出（默认 utils/experiments/held_kb<n>.json）")
    ap.add_argument("--manifest-out", type=Path,
                    default=REPORTS / "held_batch_manifest.json")
    ap.add_argument("--control-from", type=Path, default=CONTROL_RUNS,
                    help="反向对照来源：每行 CVE/run_id，取前 --control-n 个（它们是 kb-self）")
    ap.add_argument("--control-n", type=int, default=3)
    args = ap.parse_args()

    # 路径统一解析成绝对路径（否则 --out 给相对路径时 relative_to(ROOT) 会炸）
    for name in ("csv", "held_manifest", "db", "dump", "out", "manifest_out", "control_from"):
        v = getattr(args, name)
        if v is not None:
            setattr(args, name, Path(v).resolve())
    out_cfg = args.out or (EXPERIMENTS / ("held_kb%d.json" % args.n))

    print("=" * 78)
    print("MAS held 批次生成 + KB 缺席证明")
    print("  仓库根     : %s" % ROOT)
    print("  角色来源   : %s" % args.csv)
    print("  KB (SQLite): %s" % args.db)
    print("  KB (向量)  : %s" % args.dump)
    print("  样本目录   : %s" % DS)
    print("=" * 78)

    # ---------- 1. 角色与三方交叉核对 ----------
    held, kb_self_csv = load_held_roles(args.csv)
    shard_cves = load_v5_shards()
    v5_results = load_v5_results()
    man = json.loads(args.held_manifest.read_text(encoding="utf-8"))
    man_cves = sorted({str(x["cve"]).strip().upper() for x in man})

    print("\n[1] 角色来源与三方交叉核对")
    print("  CSV role=held : %d 个（去重后）" % len(held))
    print("  CSV role=kb   : %d 个（去重后）" % len(kb_self_csv))
    print("  held ∩ kb     : %d 个 %s" % (len(set(held) & set(kb_self_csv)),
                                          sorted(set(held) & set(kb_self_csv))[:5]))
    print("  v5 shards     : %d 个 CVE（%s）" % (len(shard_cves), SHARD_GLOB))
    print("    · shards 与 CSV held 完全相同 ? %s（差集 %d）"
          % (set(shard_cves) == set(held), len(set(shard_cves) ^ set(held))))
    print("  held_out_manifest: %d 个 CVE（%s）" % (
        len(man_cves), {s: sum(1 for x in man if x["split"] == s)
                        for s in sorted({x["split"] for x in man})}))
    print("    · manifest ∩ CSV held : %d 个 %s"
          % (len(set(man_cves) & set(held)), sorted(set(man_cves) & set(held))[:8]))
    print("    · manifest ∩ CSV kb   : %d 个 %s"
          % (len(set(man_cves) & set(kb_self_csv)), sorted(set(man_cves) & set(kb_self_csv))[:8]))
    print("  ⇒ 三者关系：CSV 的 200 个 held 与 v5 shards 是**同一批**（逐 CVE 一致）；")
    print("     held_out_manifest.json 是**另一套（更早的）50 个划分**（KB40+HELD10），")
    print("     与 seed2025 的 200/200 划分只有 %d 个交集，且其 KB/HELD 标签与当前 KB 不一致"
          % len(set(man_cves) & set(held)))
    print("     （下面用 title 列核对；本脚本的角色权威来源 = %s 的 role 列 + KB 实际内容）"
          % args.csv.name)

    # ---------- 2. KB 缺席证明 ----------
    print("\n[2] 建 KB 证据索引")
    print("  A 族 SQLite：")
    sq = build_sqlite_evidence(args.db, verbose=True)
    print("    · issue_patterns.title 唯一 CVE 号 %d 个；两表文本列里出现的 CVE 号 %d 个；"
          "文件身份索引 basename %d / 末两级 %d"
          % (len(sq["title_index"]), len(sq["mention_index"]),
             len(sq["file_index"]["k1"]), len(sq["file_index"]["k2"])))
    print("  B 族 Weaviate dump：")
    vec = build_vector_evidence(args.dump, verbose=True)
    print("    · dump 文本里出现的 CVE 号 %d 个；文件身份索引 basename %d / 末两级 %d"
          % (len(vec["mention_index"]), len(vec["file_index"]["k1"]), len(vec["file_index"]["k2"])))

    title_to_ids = sq["title_index"]
    print("\n  对 %d 个 held CVE 逐个判定…" % len(held))
    judged: dict[str, dict] = {}
    for cve in held:
        judged[cve] = judge(cve, sample_files(cve), sq, vec, title_to_ids)

    hits_a1 = [c for c in held if judged[c]["A1_cve_is_title_in_db"]]
    hits_a2 = [c for c in held if judged[c]["A2_cve_mentioned_in_db_text"]]
    hits_b1 = [c for c in held if judged[c]["B1_cve_in_vector_text"]]
    hits_b3 = [c for c in held if judged[c]["B3_vector_link_to_title"]]
    hits_a3 = [c for c in held if judged[c]["A3_same_relpath2_in_db"]]
    hits_b2 = [c for c in held if judged[c]["B2_same_relpath2_in_vectors"]]
    hits_a3b = [c for c in held if judged[c]["A3_same_basename_in_db"]]
    hits_b2b = [c for c in held if judged[c]["B2_same_basename_in_vectors"]]

    print("\n  KB 缺席证明（held %d 个）：" % len(held))
    print("    A1 CVE 号 == issue_patterns.title .......... 命中 %d %s"
          % (len(hits_a1), hits_a1[:5]))
    print("    A2 CVE 号出现在两表文本列 .................. 命中 %d %s"
          % (len(hits_a2), hits_a2[:5]))
    print("    B1 CVE 号出现在 dump 对象字段 .............. 命中 %d %s"
          % (len(hits_b1), hits_b1[:5]))
    print("    B3 dump.sqlite_id→title 链路一致 ........... 命中 %d %s"
          % (len(hits_b3), hits_b3[:5]))
    print("    —— 上面 4 条是『号码』证据；下面 2 条只是『库里有没有同路径文件』——")
    print("    A3 库中存在同末两级路径文件 ................ %d 个 %s"
          % (len(hits_a3), hits_a3[:6]))
    print("    B2 dump 中存在同末两级路径文件 ............. %d 个 %s"
          % (len(hits_b2), hits_b2[:6]))
    print("       （仅裸文件名相同：A3 %d 个 / B2 %d 个 —— 重名文件假阳性，见 kb_coverage 说明）"
          % (len(hits_a3b), len(hits_b2b)))

    # ---------- 3. 选批次 ----------
    print("\n[3] 选入批样本")
    lo, hi = args.min_kb * 1024, args.max_kb * 1024
    cand_rows = []
    stat = {"no_before": 0, "no_after": 0, "no_metadata": 0, "no_src": 0,
            "out_of_size": 0, "overlap": 0}
    for cve in held:
        b, a, m = DS / "before" / cve, DS / "after" / cve, DS / "metadata" / cve
        if not b.is_dir():
            stat["no_before"] += 1
            continue
        if not a.is_dir():
            stat["no_after"] += 1
            continue
        if not m.is_dir():
            stat["no_metadata"] += 1
            continue
        sz = src_bytes(cve)
        if judged[cve]["n_files"] == 0 or sz == 0:
            stat["no_src"] += 1
            continue
        if not (lo <= sz <= hi):
            stat["out_of_size"] += 1
            continue
        ov = overlap_keys(judged[cve], args.overlap_rule)
        if ov:
            stat["overlap"] += 1
            continue
        cand_rows.append({"cve": cve, "src_bytes": sz})
    cand_rows.sort(key=lambda r: (r["src_bytes"], r["cve"]))
    print("  过滤统计（held 共 %d）：目录缺 before %d / after %d / metadata %d；无源文件 %d；"
          "尺寸不在 %d-%dKB %d；同路径文件风险剔除 %d"
          % (len(held), stat["no_before"], stat["no_after"], stat["no_metadata"],
             stat["no_src"], args.min_kb, args.max_kb, stat["out_of_size"], stat["overlap"]))
    print("  候选池 %d 个" % len(cand_rows))
    if len(cand_rows) < args.n:
        print("  [警告] 候选池只有 %d 个，少于要求的 %d 个" % (len(cand_rows), args.n))
    if args.select == "smallest":
        picked = cand_rows[: args.n]
        print("  选择模式 smallest：按源文件总字节升序取前 %d 个"
              "（本模式下 seed 不改变结果，仅记录存档）" % args.n)
    else:
        rng = random.Random(args.seed)
        picked = sorted(rng.sample(cand_rows, min(args.n, len(cand_rows))),
                        key=lambda r: (r["src_bytes"], r["cve"]))
        print("  选择模式 random：seed=%d 在候选池里随机抽 %d 个" % (args.seed, len(picked)))
    batch_cves = [r["cve"] for r in picked]
    picked_ids = {r["cve"]: i for i, r in enumerate(picked, 1)}

    # ---------- 4. 写出产物 ----------
    items = []
    for r in picked:
        cve, cve_j = r["cve"], judged[r["cve"]]
        rel = "before/%s" % cve
        items.append({
            "role": "held",
            "cve": cve,
            "project": load_project(cve),
            "target_dir": (str(DS / rel) if args.local_paths else "%s/%s" % (SERVER_DS, rel)),
            "output_dir": cve,
            "kb_entry_id": None,             # held 的定义就是"没有入库条目"
            "kb_file_pattern": None,
            "src_bytes": r["src_bytes"],
            "local_target_dir": str(DS / rel),
            "after_dir": (str(DS / "after" / cve) if args.local_paths
                          else "%s/after/%s" % (SERVER_DS, cve)),
            "kb_absent": {"in_kb_db": in_kb_db(cve_j),
                          "in_kb_vectors": in_kb_vectors(cve_j),
                          "same_file_in_kb_relpath2": overlap_keys(cve_j, "relpath2"),
                          "same_file_in_kb_basename": overlap_keys(cve_j, "basename")},
        })

    cfg = {
        "description": ("线上知识库冒烟：%d 个『库外』(held) 样本（各自条目**不在**线上库里），"
                        "已用 SQLite + 向量 dump 两条独立证据证明缺席，"
                        "按源文件从小到大挑；用来测库外样本上的误报/泛化" % len(items)),
        "kb": _rel(args.db),
        "why_kb": ("分层/标签必须按『被测系统实际查询的那个库』算；held 同样要对这个库做缺席证明，"
                   "否则会把库里样本当库外样本报误报（见 utils/kb_coverage.py）"),
        "selection": {
            "min_kb": args.min_kb,
            "max_kb": args.max_kb,
            "sort": "源文件总字节升序",
            "mode": args.select,
            "seed": args.seed,
            "require_dirs": ["before", "after", "metadata"],
            "exclude_same_file_in_kb": args.overlap_rule,
            "source_of_role": str(args.csv.name),
            "kb_absence_evidence": ["A1/A2 SQLite issue_patterns+curated_issues",
                                    "B1 weaviate_kb_dump_today.jsonl"],
        },
        "items": items,
    }
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    out_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest_items = []
    for cve in held:
        j = judged[cve]
        manifest_items.append({
            "cve": cve,
            "project": load_project(cve),
            "before": str(DS / "before" / cve),
            "after": str(DS / "after" / cve),
            "metadata": str(DS / "metadata" / cve),
            "src_bytes": src_bytes(cve),
            "n_source_files": j["n_files"],
            "kb_absent_evidence": {
                "A1_cve_is_title_in_db": j["A1_cve_is_title_in_db"],
                "A1_issue_pattern_ids": j["A1_issue_pattern_ids"],
                "A2_cve_mentioned_in_db_text": j["A2_cve_mentioned_in_db_text"],
                "A2_evidence": j["A2_evidence"],
                "B1_cve_in_vector_text": j["B1_cve_in_vector_text"],
                "B1_evidence": j["B1_evidence"],
                "B3_vector_link_to_title": j["B3_vector_link_to_title"],
                "B3_matched_sqlite_ids": j["B3_matched_sqlite_ids"],
                "A3_same_basename_in_db": j["A3_same_basename_in_db"],
                "A3_same_relpath2_in_db": j["A3_same_relpath2_in_db"],
                "B2_same_basename_in_vectors": j["B2_same_basename_in_vectors"],
                "B2_same_relpath2_in_vectors": j["B2_same_relpath2_in_vectors"],
            },
            "in_batch": cve in picked_ids,
            "batch_index": picked_ids.get(cve),
            "prev_v5_run_result": v5_results.get(cve),
        })

    manifest = {
        "generated_by": "utils/experiments/make_held_batch.py",
        "role_authority": {"csv": _rel(args.csv), "column": "role==held"},
        "kb": {"sqlite": _rel(args.db),
               "sqlite_tables": ["issue_patterns", "curated_issues"],
               "vector_dump": _rel(args.dump)},
        "cross_check": {
            "csv_held": len(held),
            "csv_kb_self": len(kb_self_csv),
            "v5_shards_total": len(shard_cves),
            "v5_shards_equals_csv_held": set(shard_cves) == set(held),
            "held_out_manifest_total": len(man_cves),
            "held_out_manifest_intersect_csv_held": sorted(set(man_cves) & set(held)),
            "held_out_manifest_intersect_csv_kb": sorted(set(man_cves) & set(kb_self_csv)),
            "note": ("held_out_manifest.json 是更早的 50 个划分（KB40+HELD10），"
                     "与 seed2025 的 200 held 只有少量交集；其 split 标签不能当 seed2025 的角色依据"),
        },
        "kb_absence_proof": {
            "held_total": len(held),
            "A1_cve_is_title_in_db_hits": len(hits_a1),
            "A2_cve_mentioned_in_db_text_hits": len(hits_a2),
            "B1_cve_in_vector_text_hits": len(hits_b1),
            "B3_vector_link_to_title_hits": len(hits_b3),
            "A3_same_relpath2_in_db_hits": len(hits_a3),
            "B2_same_relpath2_in_vectors_hits": len(hits_b2),
            "A3_same_basename_in_db_hits": len(hits_a3b),
            "B2_same_basename_in_vectors_hits": len(hits_b2b),
            "A1_A2_B1_all_zero": not (hits_a1 or hits_a2 or hits_b1),
            "hits_any_A1": hits_a1,
            "hits_any_A2": hits_a2,
            "hits_any_B1": hits_b1,
        },
        "selection": {
            "n_requested": args.n, "seed": args.seed, "mode": args.select,
            "min_kb": args.min_kb, "max_kb": args.max_kb,
            "overlap_rule": args.overlap_rule,
            "target_dir_style": "local" if args.local_paths else "server(/root/autodl-tmp/MAS)",
            "candidate_pool": len(cand_rows),
            "filter_stats": stat,
            "batch": batch_cves,
            "config": _rel(out_cfg),
        },
        "items": manifest_items,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")

    # ---------- 5. 自验：批次断言 + 反向对照 ----------
    print("\n[4] 自验")
    print("  批次配置  : %s（%d 项）" % (out_cfg, len(items)))
    print("  逐样本清单: %s（%d 项）" % (args.manifest_out, len(manifest_items)))

    bad = [c for c in batch_cves if in_kb_db(judged[c]) or in_kb_vectors(judged[c])]
    ov_in_batch = [c for c in batch_cves if overlap_keys(judged[c], "relpath2")]
    print("  断言①：入批 %d 个样本，A1/A2/B1 任一命中 = %d 个 %s"
          % (len(batch_cves), len(bad), bad))
    print("  断言②：入批样本里'库中有同末两级路径文件' = %d 个 %s（--overlap-rule=%s%s）"
          % (len(ov_in_batch), ov_in_batch, args.overlap_rule,
             "，不剔除故此处只作提示、不判失败" if args.overlap_rule == "none" else ""))

    control = []
    if args.control_from.is_file():
        lines = [l.strip() for l in args.control_from.read_text(encoding="utf-8").splitlines() if l.strip()]
        control = [l.split("/")[0].strip().upper() for l in lines[: max(args.control_n, 0)]]
    print("\n  反向对照（同一套匹配逻辑跑 kb-self，必须命中；来源 %s）：" % args.control_from.name)
    print("    %-16s %-6s %-8s %-8s %-8s %s" % ("CVE", "A1", "A2", "B3链路", "B2同文件", "判定"))
    ctrl_ok = 0
    for cve in control:
        j = judge(cve, sample_files(cve), sq, vec, title_to_ids)
        ok = in_kb_db(j)
        ctrl_ok += 1 if ok else 0
        print("    %-16s %-6s %-8s %-8s %-8s %s"
              % (cve, j["A1_cve_is_title_in_db"], j["A2_cve_mentioned_in_db_text"],
                 j["B3_vector_link_to_title"], bool(j["B2_same_relpath2_in_vectors"]),
                 "命中 KB ✓" if ok else "❌ 未命中（匹配逻辑坏了）"))
        if j["A1_cve_is_title_in_db"]:
            print("        A1 证据: issue_patterns.id=%s（title 就是该 CVE）"
                  % j["A1_issue_pattern_ids"])
        if j["B2_same_relpath2_in_vectors"]:
            print("        B2 证据: dump 里同末两级路径 %s" % j["B2_same_relpath2_in_vectors"][:2])
    if not control:
        print("    [警告] 没读到对照 CVE（%s 不存在？）→ 反向对照没做" % args.control_from)
    else:
        print("    反向对照命中 %d/%d" % (ctrl_ok, len(control)))

    # ---------- 6. 人看的汇总表 ----------
    print("\n" + "=" * 78)
    print("汇总")
    print("  held（CSV role=held）              : %d" % len(held))
    print("  与 v5 shards 逐 CVE 一致           : %s" % (set(shard_cves) == set(held)))
    print("  与 held_out_manifest 交集          : %d（manifest 是另一套 50 个划分）"
          % len(set(man_cves) & set(held)))
    print("  KB 缺席证明（号码证据，必须全 0）  : A1=%d  A2=%d  B1=%d  (B3一致性=%d)"
          % (len(hits_a1), len(hits_a2), len(hits_b1), len(hits_b3)))
    print("  同路径文件风险（非本样本条目）     : A3(末两级)=%d  B2(末两级)=%d"
          % (len(hits_a3), len(hits_b2)))
    print("  候选池 / 入批                      : %d / %d" % (len(cand_rows), len(items)))
    print("  反向对照（kb-self %d 个）           : 命中 %d/%d" % (len(control), ctrl_ok, len(control)))
    print("\n  批次内容（%s）：" % out_cfg.name)
    print("  %-4s %-16s %-12s %9s  %-6s %-6s  %s"
          % ("#", "CVE", "project", "src_KB", "kb_db", "kb_vec", "目标目录"))
    for i, it in enumerate(items, 1):
        print("  %-4d %-16s %-12s %9.1f  %-6s %-6s  %s"
              % (i, it["cve"], (it["project"] or "-")[:12],
                 it["src_bytes"] / 1024, it["kb_absent"]["in_kb_db"],
                 it["kb_absent"]["in_kb_vectors"], it["target_dir"]))
    print("\n  怎么跑（真机 / GPU 服务器，MAS 根目录下）：")
    print("      python mas.py batch --config %s"
          % _rel(out_cfg).replace("\\", "/"))
    print("      # 本机 CPU 复刻（先加 --local-paths 重新生成配置）：")
    print("      python -X utf8 utils/experiments/make_held_batch.py --local-paths")
    print("      # 跑完用 make_run_list.py + compare_arms.py 汇总：")
    print("      python utils/experiments/make_run_list.py --out reports/held_runs.txt \\")
    print("          --logs <本次批处理日志>")

    ok_all = ((not bad)
              and (args.overlap_rule == "none" or not ov_in_batch)
              and bool(control) and ctrl_ok == len(control))
    print("\n" + ("[OK] 全部自验通过：入批样本在 KB 里 0 命中，且反向对照能测出 kb-self 命中"
                  if ok_all else
                  "[FAIL] 自验未通过：见上面的断言①②与反向对照（不要拿这批去跑）"))
    print("=" * 78)
    if not ok_all:
        sys.exit(1)


if __name__ == "__main__":
    main()
