#!/usr/bin/env python
"""C3 前置验证：在真实记录的 gap chunk 上检查「功能意图骨架」的质量（不花 GPU）。

要回答：
  1. 样板噪声（license/copyright）是否被去掉？—— 原始 chunk 常整段是版权头
  2. 高风险 API / 函数名 / 控制流 / 类型 是否被捕获？（这些才是漏洞语义的载体）
  3. 骨架长度是否落在合理区间（太长会超出有效注意力，太短信息不足）
"""
import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from utils.code_intent import build_code_intent, _RISKY_APIS  # noqa: E402

QOFF2 = {"0fbe60f4-dea6-4cfd-864a-f789fd0194d0", "12dc120e-2c9d-49fe-8c2d-8e2a614c8f26",
         "6c817c11-76ae-41c9-ac66-ad027c9415a8", "76be00c4-52e3-41e8-afc5-afa29b9edf94",
         "8ca34dfe-f407-4a65-abad-38634eff4b31", "b63de181-9078-4c9e-9c34-e51ffe54c425",
         "ce3ecece-684a-433e-bf6f-0021e0bb6a6c", "df87d95a-b194-4694-8574-a77007d26706"}


def main():
    chunks = []
    for f in sorted(glob.glob("reports/analysis/*/*/second_pass/consolidated/*_r2.json")):
        if f.split("/")[3] not in QOFF2:
            continue
        d = json.load(open(f, encoding="utf-8"))
        for it in (d.get("gap_retrieval_evidence") or []):
            cc = it.get("code_chunk") or {}
            t = str(cc.get("text") or "")
            if t:
                chunks.append((os.path.basename(str(d.get("file") or "")), t))
    print("gap chunk 数: %d" % len(chunks))

    raw_len, int_len = [], []
    boiler_only = 0
    has_risky = 0
    no_signal = 0
    risky_counter = Counter()
    for fn, t in chunks:
        s = build_code_intent(t, file_path=fn)
        raw_len.append(len(t))
        int_len.append(len(s))
        is_boiler = "copyright" in t.lower() or "licen" in t.lower()
        if is_boiler and "[comments]" not in s:
            boiler_only += 1
        apis = []
        if "[api_risky] " in s:
            apis = s.split("[api_risky] ")[1].split("\n")[0].split(", ")
            has_risky += 1
            for a in apis:
                risky_counter[a] += 1
        if "[funcs]" not in s and "[api" not in s and "[control]" not in s:
            no_signal += 1

    import statistics as st
    print("\n=== 长度 ===")
    print("  原始 chunk : min=%d 中位=%d max=%d" % (min(raw_len), int(st.median(raw_len)), max(raw_len)))
    print("  意图骨架   : min=%d 中位=%d max=%d" % (min(int_len), int(st.median(int_len)), max(int_len)))
    print("  压缩比      : %.2f" % (st.median(int_len) / st.median(raw_len)))
    print("\n=== 质量 ===")
    print("  含 license/版权字样且骨架里已无 comments（样板被丢弃）: %d / %d" % (boiler_only, len(chunks)))
    print("  骨架捕获到高风险 API 的 chunk: %d / %d" % (has_risky, len(chunks)))
    print("  骨架完全没有 funcs/api/control 信号的 chunk: %d / %d" % (no_signal, len(chunks)))
    print("  出现最多的高风险 API:", dict(risky_counter.most_common(12)))

    print("\n=== 样例 1：license 密集的 chunk ===")
    for fn, t in chunks:
        if "copyright" in t.lower():
            print("  file=%s  原始前 150: %r" % (fn[:50], t[:150]))
            print("  骨架:\n    " + build_code_intent(t, file_path=fn).replace("\n", "\n    ")[:700])
            break
    print("\n=== 样例 2：含高风险 API 的 chunk ===")
    for fn, t in chunks:
        s = build_code_intent(t, file_path=fn)
        if "[api_risky]" in s:
            print("  file=%s" % fn[:60])
            print("  骨架:\n    " + s.replace("\n", "\n    ")[:900])
            break


if __name__ == "__main__":
    sys.exit(main())
