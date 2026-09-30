# -*- coding: utf-8 -*-
"""seed=2025 命中案例定点复算（论文"结果可复核"小表的数据来源）。

只读复算，不跑完整评测流水线、不改任何已有文件。逐样本/逐候选重放
gap 证据里的候选，用 v5 门控 DNF 重新判定，并抽出一个「词法独立命中」主案例
与一个「语义独立准入」次案例，逐字段打印核对依据。

复算口径（与 _regate_seed2025.py / param_sweep_v5.py 完全一致）：
  s(x) = 0.5·I_lex + 0.4·I_loc + 0.1·I_desc（封顶 1.0）
    I_lex  : "error_code_clone" ∈ matched_fields
    I_loc  : matched_fields ∩ LOCATE ≠ ∅（LOCATE = 文件名/类名/函数名锚点）
    I_desc : matched_fields ∩ WEAK ≠ ∅
  v(x)   : 候选 semantic_score（各视图近邻距离折算后取最大）
  k(x)   = [v(x) ≥ 0.65] ∧ [a(x) ≥ 0.35] ∧ [s(x) ≥ 0.20]
  admit  = F(x) ∧ [ s(x) ≥ 0.65 ∨ k(x) ]
  F(x)   : 跨文件拒绝 / 弱结构无锚拒绝（仅向量通道）+ 代码已修复拒绝（全通道）

运行：python utils/experiments/reproduce_hit_case_seed2025.py
"""
import sys
from pathlib import Path
ROOT = Path(r"E:\MyOwn\ProgramStudy\MAS")
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import asyncio, gzip, json, sqlite3  # noqa: E402
from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent as Agent  # noqa: E402
from infrastructure.database.sqlite.service import DatabaseService  # noqa: E402
from utils.experiments import param_sweep_fast as PSF  # noqa: E402
from utils.experiments.ablate_views_singlepass import _load_ci_to_pattern, _load_id_by_title  # noqa: E402
from utils.experiments.hard_filter_ablation import install_caches  # noqa: E402
from utils.experiments.local_numpy_weaviate import NumpyVectorService  # noqa: E402

DB = ROOT / "reports" / "mas_seed2025.db"
KB = ROOT / "reports" / "_from_gpu" / "weaviate_kb_dump.jsonl"
DUMP = ROOT / "reports" / "wsens_dump_seed2025_shard{}.jsonl.gz"

TS, TAU, THETA_A, THETA_W = 0.65, 0.65, 0.35, 0.2


def s_of(mf):
    s = 0.0
    if "error_code_clone" in mf:
        s += 0.5
    if mf & PSF.LOCATE:
        s += 0.4
    if mf & PSF.WEAK:
        s += 0.1
    return min(1.0, s)


def gate_paper(agent, p):
    """按 v5 公式重算门控，返回 (decision, s, v, a, cross, weak, fixed)。"""
    mf = set(p["mf"])
    s = s_of(mf)
    v = float(p["semantic"] or 0.0)
    a = float(p["anchor_s"] or 0.0)
    ch = p["ch"]
    anchor_ok = p["anchor"]
    cross = weak = fixed = False
    if ch == "weaviate":
        if p["cross"] and not anchor_ok:
            return "discarded", s, v, a, True, False, False
        if s < THETA_W and not anchor_ok:
            return "discarded", s, v, a, False, True, False
    if p["fixed"]:
        return "discarded", s, v, a, False, False, True
    if s >= TS:
        return "formal", s, v, a, False, False, False
    if v >= TAU and a >= THETA_A and s >= THETA_W:
        return "explanatory", s, v, a, False, False, False
    return "discarded", s, v, a, False, False, False


async def main():
    agent = Agent()
    agent.vector_service = NumpyVectorService(KB)
    agent.db_service = DatabaseService(database_url=f"sqlite:///{DB}")
    install_caches(agent)
    id_by_title = _load_id_by_title(str(DB))
    ci2p = _load_ci_to_pattern(str(DB))
    sp = await agent.db_service.get_issue_patterns(status="active")
    sp = sp[: getattr(agent, "max_sqlite_patterns", 2000)]

    con = sqlite3.connect(str(DB))
    cur = con.cursor()
    cur.execute("SELECT id, pattern_id, file_path, start_line, end_line, code_snippet, solution FROM curated_issues")
    curated_by_id = {int(r[0]): r for r in cur.fetchall()}
    cur.execute("SELECT id, title, error_type, severity, error_description, file_pattern, class_pattern FROM issue_patterns")
    ipinfo = {int(r[0]): r for r in cur.fetchall()}
    con.close()

    PRIMARY = "CVE-2014-5352"
    SECONDARY = "CVE-2017-1000252"
    targets = {PRIMARY, SECONDARY}

    hits = {}                  # cve -> 命中候选明细（s 最高者）
    own_weaviate_sim = {PRIMARY: 0.0, SECONDARY: 0.0}

    for sid_shard in range(4):
        with gzip.open(str(DUMP).format(sid_shard), "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cve = r.get("cve")
                if cve not in targets:
                    continue
                own = id_by_title.get(cve)
                for fe in r.get("files") or []:
                    fp = fe.get("file") or ""
                    for g in fe.get("gap") or []:
                        for h in g.get("weaviate_hits") or []:
                            if h.get("sqlite_id") == own:
                                own_weaviate_sim[cve] = max(own_weaviate_sim[cve], float(h.get("similarity") or 0.0))
                    ev = PSF.prepare_evidence(agent, fe.get("gap") or [], sp, fp)
                    for item in ev:
                        cc = item.get("code_chunk") or {}
                        for p in item["prepared"]:
                            c = p["c"]
                            sid = c.get("sqlite_id")
                            ch = str(c.get("channel") or "")
                            is_own = (sid == own) or (ci2p.get(sid) == own)
                            if not is_own:
                                continue
                            dec, s, v, a, cross, weak, fixed = gate_paper(agent, p)
                            if dec == "discarded":
                                continue
                            if cve not in hits or s > hits[cve]["s"]:
                                hits[cve] = {
                                    "cve": cve, "own": own, "sid": sid, "ch": ch,
                                    "mf": sorted(set(p["mf"])), "s": s, "v": v, "a": a,
                                    "dec": dec, "cross": cross, "weak": weak, "fixed": fixed,
                                    "structured_raw": float(c.get("structured_score") or 0.0),
                                    "anchor_raw": float(c.get("anchor_score") or 0.0),
                                    "file": fp, "cc": cc,
                                    "solution": c.get("solution") or "",
                                    "error_description": c.get("error_description") or "",
                                    "vector_layer": c.get("vector_layer"),
                                }

    for tag, cve in (("主案例（词法独立命中）", PRIMARY), ("次案例（语义独立准入）", SECONDARY)):
        h = hits.get(cve)
        if h is None:
            print(f"[{tag}] {cve}: 未找到命中候选")
            continue
        own = h["own"]
        t = ipinfo.get(own)
        mf = set(h["mf"])
        I_lex = int("error_code_clone" in mf)
        I_loc = int(bool(mf & PSF.LOCATE))
        I_desc = int(bool(mf & PSF.WEAK))
        lex_fields = sorted(mf & {"error_code_clone"})
        loc_fields = sorted(mf & PSF.LOCATE)
        desc_fields = sorted(mf & PSF.WEAK)
        s, v, a = h["s"], h["v"], h["a"]
        k = bool(v >= TAU and a >= THETA_A and s >= THETA_W)
        admit = bool(h["dec"] in ("formal", "explanatory"))
        F_pass = not (h["cross"] or h["weak"] or h["fixed"])

        frags = agent._extract_error_code_fragments(h["solution"])

        print("=" * 100)
        print(f"### {tag}：{cve}")
        print(f"1. 样本 CVE / 待审文件名：{cve} ；文件 {Path(h['file']).name}"
              f"（扁平路径 {Path(h['file']).name}）")
        print(f"2. 触发命中的错误代码（库内 solution 的 'Remove incorrect logic' 词元片段）：")
        if frags:
            for f in frags:
                print(f"     错误代码词元序列 = {f}")
        cc = h["cc"]
        text = cc.get("text") or ""
        start = int(cc.get("start_line") or 1)
        print(f"    待审代码分片 {Path(h['file']).name} L{cc.get('start_line')}-L{cc.get('end_line')} 中该错误代码的原样行：")
        shown = False
        for ln, raw in enumerate(text.splitlines(), start=start):
            raw_tok = set(agent._tokenize_code(raw))
            if any(set(f) <= raw_tok for f in frags):
                print(f"      L{ln}: {raw}")
                shown = True
        if not shown:
            print("      （未能逐行回显；下面是分片节选）")
            for raw in text.splitlines()[:14]:
                print(f"      {raw}")
        print(f"3. 命中字段逐项（matched_fields 原值 = {h['mf']}）：")
        print(f"     I_lex = {I_lex}   具体字段 = {lex_fields}（error_code_clone → 错误代码连续、原样、有序词元子串出现）")
        print(f"     I_loc = {I_loc}   具体字段 = {loc_fields}")
        print(f"     I_desc = {I_desc}   具体字段 = {desc_fields}")
        print(f"4. 门控分量：")
        print(f"     s(x) = 0.5·{I_lex} + 0.4·{I_loc} + 0.1·{I_desc} = {s}")
        print(f"     v(x) = {v:.4f}（候选 semantic_score；own 条目在原始 weaviate 检索中的最高相似度 = {own_weaviate_sim[cve]:.4f}）")
        print(f"     a(x) = {a:.2f}（anchor_score 锚点完整度）")
        print(f"     k(x) = [v≥0.65]∧[a≥0.35]∧[s≥0.2] = {bool(v>=TAU)}∧{bool(a>=THETA_A)}∧{bool(s>=THETA_W)} = {k}")
        print(f"     F(x)：跨文件拒绝={h['cross']}，弱结构无锚拒绝={h['weak']}，代码已修复拒绝={h['fixed']} → F(x)={'通过' if F_pass else '拒绝'}")
        print(f"     admit(x) = F(x) ∧ [s≥0.65 ∨ k(x)] = {F_pass} ∧ [{bool(s>=TS)} ∨ {k}] = {admit}  →  判定 = {h['dec']}")
        print(f"5. 命中知识库条目：")
        if h["ch"] == "curated_issue":
            crow = curated_by_id.get(h["sid"])
            if crow:
                cid, pid, cfp, csl, cel, csnip, csol = crow
                print(f"     channel = curated_issue ；evidence_id(curated_issues.id) = {cid} → pattern_id(sqlite_id) = {pid}")
                print(f"     curated 文件路径 = {cfp}，错误代码定位行 = {csl}-{cel}")
        else:
            print(f"     channel = {h['ch']} ；sqlite_id = {h['sid']} ；vector_layer = {h['vector_layer']}")
        print(f"     issue_patterns.id = {own} ；title = {t[1] if t else ''} ；error_type = {t[2] if t else ''} ；severity = {t[3] if t else ''}")
        print(f"     error_description（前 240 字）= {(t[4] or '')[:240]}")
        print(f"6. 库内修复方案（solution）与错误代码片段：")
        print(f"     solution = {h['solution']}")
        print(f"     file_pattern = {t[5] if t else ''} ；class_pattern = {t[6] if t else ''}")
        print()


asyncio.run(main())
