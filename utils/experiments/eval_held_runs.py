# -*- coding: utf-8 -*-
"""库外（held）批次的评测口径：**库外样本上，任何被放行的条目都是误报面**。

## 为什么库外要用另一套口径

库内（kb-self）样本的指标是"**自己的条目**有没有被捞到 / 被放行"——因为样本对应的知识
本来就在库里。库外样本**在库里没有自己的条目**，所以：

* 放行数**不可能是**"命中自己"，**每一条放行都落在"给库外样本推了库里的知识"这件事上**；
* 这正是预登记断言里"`held-pure` 误报不增加"要测的东西。

因此对库外样本量三件事：

1. **放行总数**（每条都是一个误报面）；
2. **同文件撞车**：被放行条目的 `file_pattern` 与样本自己在分析的文件同名/同相对路径
   —— 这是已知现象（不同 CVE 改同一个文件），**性质比跨文件轻**，要单独报；
3. **跨文件**：既不是自己的知识、文件也对不上——这才是"纯误报"。

## 一条逻辑只允许一个实现

候选取数、`new_findings` 解析、"curated 通道 id 要先映射到 issue_patterns.id"这些，
**全部复用 `compare_arms.py` 的 `load_arm`**，本脚本只加库外专属的口径。
所以它有一个硬自检：把三臂那批 **库内** run 列表喂进来时，**放行总数必须等于
`compare_arms.py` 报出的 35 / 36 / 37**（用 `--expect-total` 断言），
对不上说明两个实现在同一份数据上不一致，先修工具再谈结论。

用法：
    # 库外批次跑完后（run 列表由 make_run_list.py 生成）
    python -X utf8 utils/experiments/eval_held_runs.py \
        --arms "库外=reports/held_runs.txt" --db reports/mas_live.db

    # 自检（库内三臂，放行总数应当与 compare_arms.py 一致）
    python -X utf8 utils/experiments/eval_held_runs.py \
        --arms "新系统=reports/arm1_runs.txt" --expect-total 35
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.experiments.compare_arms import load_arm  # noqa: E402


def file_identity(path: str, depth: int = 2) -> str:
    """把两种路径写法归一化到同一个"文件身份"，用于判断"是不是同一个文件"。

    **为什么不能直接用 norm_key**：两侧写法不一样 ——
      * 知识库里的 `file_pattern` 是**真斜杠**：`include/freerdp/codec/nsc.h`
      * 运行产物里的文件路径是**压平的**（目录分隔被换成 `__`，前面还有一层哈希目录）：
        `.../before/CVE-2018-8788/d1112c27/include__freerdp__codec__nsc.h`
    直接按 `/` 取末两段，会把**同一个文件**判成"跨文件"——第一版就是这么错的：
    三臂 35/36/37 条放行全被判成 `cross_file`，而同文件比例实际已知是**很高的**。
    靠"与已知结果对不上"才发现（《03》坑 24 的同款纪律）。
    """
    p = str(path or "").strip().replace("\\", "/")
    if not p:
        return ""
    tail = p.rsplit("/", 1)[-1]
    if "__" in tail:                     # 压平写法：把最后一段按 __ 还原成目录
        p = p[: len(p) - len(tail)] + tail.replace("__", "/")
    parts = [x for x in p.split("/") if x]
    return "/".join(x.lower() for x in parts[-depth:]) if parts else ""


def classify(admitted, sample_files, file_by_id) -> dict:
    """把一条放行分到"同文件撞车 / 跨文件"。

    判据：知识条目的 `file_pattern` 与**本次被分析的文件**（或该条 new_finding 自己的 file）
    是不是同一个文件（用 `file_identity` 归一化后比较）。
    """
    kb_file = file_by_id.get(admitted.get("sqlite_id")) or ""
    kb_id = file_identity(kb_file)
    candidates = list(sample_files) + [admitted.get("file") or ""]
    hit = bool(kb_id) and any(file_identity(f) == kb_id for f in candidates if f)
    return {"kind": "same_file" if hit else "cross_file",
            "kb_file": kb_file,
            "sample_file": admitted.get("file") or ""}


def main() -> None:
    ap = argparse.ArgumentParser(description="库外(held)批次的误报面评测")
    ap.add_argument("--arms", nargs="+", required=True, help="形如 名称=run列表路径")
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_live.db")
    ap.add_argument("--expect-total", type=int, default=None,
                    help="自检：该臂放行总数必须等于这个数（用来和 compare_arms.py 对齐）")
    args = ap.parse_args()

    con = sqlite3.connect("file:%s?mode=ro" % args.db.as_posix(), uri=True)
    id_by_title = {(t or "").strip().upper(): int(i) for i, t in
                   con.execute("select id, title from issue_patterns")}
    file_by_id = {int(i): (fp or "") for i, fp in
                  con.execute("select id, file_pattern from issue_patterns")}
    ci_to_pattern = {int(i): int(p) for i, p in
                     con.execute("select id, pattern_id from curated_issues")}
    con.close()

    ok_all = True
    for item in args.arms:
        name, path = item.split("=", 1)
        runs_path = Path(path)
        if not runs_path.is_absolute():
            runs_path = ROOT / runs_path
        arm = load_arm(runs_path, id_by_title, ci_to_pattern)
        held_ids = {v for v in id_by_title.values()}
        total = same = cross = 0
        rows = []
        for cve, rec in arm.items():
            for a in rec["admitted"]:
                total += 1
                info = classify(a, rec["files"], file_by_id)
                same += info["kind"] == "same_file"
                cross += info["kind"] == "cross_file"
                rows.append((cve, a.get("channel"), a.get("sqlite_id"), info["kind"],
                             info["kb_file"], info["sample_file"]))
            # 库外样本：它在库里**不该**有自己的条目（这就是 held 的定义）
            if rec.get("own") is not None:
                held_ids.discard(int(rec["own"]))

        print("=" * 100)
        print("臂：%s    样本 %d 个    run 列表 %s" % (name, len(arm), runs_path.name))
        print("=" * 100)
        print("  被分析文件不在库里的样本数（own 为空）: %d / %d"
              % (sum(1 for r in arm.values() if r.get("own") is None), len(arm)))
        print("  **放行总数 = 误报面**: %d" % total)
        print("     同文件撞车（不同 CVE 同一文件，性质较轻）: %d" % same)
        print("     **跨文件（纯误报）**: %d" % cross)
        if total:
            print("     跨文件占比: %.1f%%" % (100.0 * cross / total))
        chans = {}
        for r in rows:
            chans[r[1]] = chans.get(r[1], 0) + 1
        print("  放行来源通道: %s" % (chans or "(无)"))
        if rows:
            print("  前 8 条放行明细（CVE / 通道 / 条目 / 类别 / 库里的文件 / 样本文件）:")
            for r in rows[:8]:
                print("     %-16s %-14s %-6s %-10s %-28s %s"
                      % (r[0], str(r[1]), str(r[2]), r[3], (r[4] or "")[-28:],
                         (r[5] or "")[-40:]))
        if args.expect_total is not None:
            good = total == args.expect_total
            ok_all &= good
            print("  [%s] 自检：放行总数 %d == 期望 %d（与 compare_arms.py 对齐）"
                  % ("OK" if good else "NG", total, args.expect_total))
        print()

    print("=" * 100)
    print("结论：%s" % ("自检通过" if ok_all else "**自检不通过 —— 两个实现不一致，先修工具**"))
    print("=" * 100)
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
