# -*- coding: utf-8 -*-
"""判断"某个被分析样本的文件是否真的在知识库里"—— 用于自动判定 kb / held 标签。

## 为什么需要它

批次配置里每个样本有一个 `role` 字段，取 `kb`（库内）或 `held`（库外）。
它的语义应该是：**这个样本的源码文件是否已经在知识库里**。

* `kb`  = 在库里 → 检索理应能捞回它自己的那条记录 → 用来测**召回**
* `held` = 不在库里 → 检索捞不到自己的记录 → 用来测**泛化 / 误报**

但 `role` 是**手写**在配置里的，而知识库是另一个脚本用某个随机种子建的，
两者没有做一致性校验。本项目实测就出现了**标签与事实相反**的情况：

    smoke8 里标成 held（库外）的 CVE-2014-6229，它的文件 hphp/runtime/ext/ext_hash.cpp
    恰恰是 9 个文件里**唯一**真正在库里的；而被标成 kb 的 6 个，文件全都不在库里。

这会造成**评测泄漏**：按标签报"库外样本上的表现"，其中却混着实际在库里的样本。

## 怎么用

    from utils.kb_coverage import load_kb_index, sample_in_kb
    kb = load_kb_index("infrastructure/database/mas.db")
    role = "kb" if sample_in_kb(target_dir, kb) else "held"

判定口径：把样本的源文件名和知识库的 `file_pattern` 都归一化成**裸文件名**后比较
（数据集里的文件名是 `a__b__c.c` 的转义写法，知识库里是 `a/b/c.c` 的真实路径）。
这正是门控里 `basename_match` / `file_basename_anchor` 用的同一把尺子。
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

SOURCE_EXT = (".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".py", ".java", ".S", ".js", ".ts")


def normalize_relpath(path: str) -> str:
    """把任意写法归一化成**相对路径**（小写，无前导目录）。

    · 数据集转义名：`hphp__runtime__ext__ext_hash.cpp` → `hphp/runtime/ext/ext_hash.cpp`
    · 知识库真实路径：同左
    · 绝对路径：剥掉前面的本地前缀，只留 `a/b/c.c` 形式
    """
    s = str(path or "").replace("\\", "/").strip().lower()
    b = os.path.basename(s)
    if "__" in b:
        # 转义名本身就是一个完整相对路径：a__b__c.c → a/b/c.c
        return b.replace("__", "/")
    # 真实路径：尽量剥掉 tests/... 之类的本地前缀，保留末几段
    parts = [p for p in s.split("/") if p]
    return "/".join(parts[-4:]) if len(parts) > 4 else "/".join(parts)


def normalize_key(path: str, depth: int = 2) -> str:
    """取"末 depth 级路径"作为比对键。默认 2 级 = 目录 + 文件名。

    为什么不用裸文件名：本数据集里重名极常见（inode.c / core.c / socket.c / map.c …），
    只比文件名会把 `fs/udf/inode.c` 和 `fs/overlayfs/inode.c` 误判成同一个文件
    —— 实测就出现过这个假阳性。
    末两级既能区分不同目录，又能容忍数据集与知识库的前缀差异。
    """
    rp = normalize_relpath(path)
    parts = [p for p in rp.split("/") if p]
    if not parts:
        return ""
    return "/".join(parts[-depth:]) if len(parts) >= depth else "/".join(parts)


def normalize_basename(path: str) -> str:
    """裸文件名（小写）。仅用于展示，不建议单独作为判定依据。"""
    return os.path.basename(normalize_relpath(path))


def normalize_project(path: str) -> str:
    """取"项目/顶层目录"：数据集转义名取第一个 `__` 之前的部分，真实路径取第一段。"""
    s = str(path or "").replace("\\", "/").strip().lower()
    b = os.path.basename(s)
    if "__" in b:
        return b.split("__")[0]
    return s.split("/")[0] if "/" in s else ""


def load_kb_index(db_path: str | Path, status: Optional[str] = None) -> Dict[str, Set[str]]:
    """读知识库，返回多个粒度的比对集合。

    关键区分（**两者语义不同，不要混用**）：
      · `cves`  = 知识库里存了哪些 CVE 的**条目**（来自 title 列，title 就是 CVE 编号）
                  → 这决定 `kb` / `held` 标签：样本**自身的条目**是否入库
      · `key1/2/3` = 知识库里出现过哪些**文件路径**
                  → 这只说明"库里有没有同文件的知识"（可能是**别的 CVE** 的同名文件）
                    它决定的是"门控的同文件锚点有没有机会触发"，不是"答案在不在库"
    """
    c = sqlite3.connect(str(db_path))
    sql = "select file_pattern, title from issue_patterns"
    if status:
        sql += " where status = ?"
        rows = c.execute(sql, (status,)).fetchall()
    else:
        rows = c.execute(sql).fetchall()
    c.close()
    keys1, keys2, keys3, projs, pats, cves = set(), set(), set(), set(), set(), set()
    for fp, title in rows:
        fp = str(fp or "").strip()
        t = str(title or "").strip()
        if t:
            cves.add(t.upper())
        if not fp:
            continue
        pats.add(fp.lower())
        for depth, bucket in ((1, keys1), (2, keys2), (3, keys3)):
            k = normalize_key(fp, depth)
            if k:
                bucket.add(k)
        p = normalize_project(fp)
        if p:
            projs.add(p)
    return {"cves": cves, "key1": keys1, "key2": keys2, "key3": keys3,
            "projects": projs, "patterns": pats,
            "basenames": keys1}          # 兼容旧字段名


def cve_in_kb(cve: str, kb: Dict[str, Set[str]]) -> bool:
    """**权威判定**：该 CVE 自身的条目是否在知识库里（比对 title 列里的 CVE 编号）。"""
    return str(cve or "").strip().upper() in kb["cves"]


def files_under(target_dir: str | Path) -> List[str]:
    """列出一个样本目录（可能含多级子目录）下的所有源文件名。"""
    out: List[str] = []
    root = Path(target_dir)
    if not root.exists():
        return out
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in SOURCE_EXT:
            out.append(os.path.basename(f.as_posix()))
    return out


def sample_in_kb(target_dir: str | Path, kb: Dict[str, Set[str]],
                 mode: str = "relpath2") -> bool:
    """判断该样本是否有文件落在知识库里。

    mode（默认 relpath2，最稳妥）：
      · "relpath2"：末两级路径（目录+文件名）命中 —— **推荐**
      · "relpath3"：末三级路径命中（更严格，要求父目录也一致）
      · "basename"：只比裸文件名（**有过假阳性，慎用**）
      · "project" ：只比项目/顶层目录（最宽松）
    """
    files = files_under(target_dir)
    if not files:
        return False
    if mode == "relpath2":
        return bool({normalize_key(f, 2) for f in files} & kb["key2"])
    if mode == "relpath3":
        return bool({normalize_key(f, 3) for f in files} & kb["key3"])
    if mode == "basename":
        return bool({normalize_key(f, 1) for f in files} & kb["key1"])
    if mode == "project":
        return bool({normalize_project(f) for f in files} & kb["projects"])
    raise ValueError("未知 mode: %s" % mode)


def explain(target_dir: str | Path, kb: Dict[str, Set[str]]) -> Dict[str, object]:
    """给出判定的详细依据，便于人工复核。"""
    files = files_under(target_dir)
    k1 = {normalize_key(f, 1) for f in files}
    k2 = {normalize_key(f, 2) for f in files}
    k3 = {normalize_key(f, 3) for f in files}
    return {
        "files": files,
        "hit_basenames": sorted(k1 & kb["key1"]),
        "hit_relpaths": sorted(k2 & kb["key2"]),
        "hit_projects": sorted({normalize_project(f) for f in files} & kb["projects"]),
        "in_kb": bool(k2 & kb["key2"]),
        "in_kb_strict": bool(k3 & kb["key3"]),
    }
