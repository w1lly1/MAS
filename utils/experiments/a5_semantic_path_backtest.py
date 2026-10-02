# -*- coding: utf-8 -*-
"""**A5 离线回测**：给"语义通路"换几种评分口径，量**收益与误报代价**（决定 B1）。

## 为什么要有这个回测
预检 2/A4 已经把机制查清：
* 现状语义判据 `v(x) ≥ τ=0.65` **不可达**（17,518 个候选最高 0.625；块级最高也只有 0.626）；
* 但块级的**排序信号是有的**（自己那条 z≥2 的有 23/30）；
* 于是问题变成：**换一种"用语义分"的方式，能不能把语义命中从 0 提上去，代价是多少误报？**

这件事必须**用数字回答**，否则动门控就是拿精度赌博（这正是当初关跨文件闸门的理由）。

## 四种评分口径（同一套候选、同一批样本，只换判据）
1. `abs_tau`：**现状**——语义分 ≥ 0.65 就放行（预期命中 0，用来复现基线）；
2. `zscore`：**相对分**——语义分按"该查询在全库 200 条上的分布"标准化，z ≥ Z 才放行
   （文献里的"按通道归一化/排名"那一类）；
3. `percentile`：**分位**——该候选落在该查询相似度分布的前 p%；
4. `lexical`：**词元通道**（针命中）——离线复算，作为"现有主力"的参照系；
5. `fused`：**融合**——词元命中 **或** 语义 z ≥ Z（最接近"语义词元互补"的形式化）；
6. `pair_llm`：**配对打分（LLM judge）**——把"当前代码块"与"库里那条的描述+修法"一起给 Qwen，
   要它判定 `same_defect / related / unrelated`。**这是唯一能表达"语义命中"语义的方式**，
   而不是拿两个向量的点积当判据。
   ⚠️ 真正的 **cross-encoder 重排**离线跑不了（本机没有重排模型，只有 distilbert/gpt2/codebert/Qwen），
   接口留在 `score_pair_cross_encoder()` 里并注明，待机器阶段补。

## 评价集（都在本地）
* **正样本 30 个**：kb-self 样本（自己那条知识**在**库里）→ 指标：**自己那条能不能被放行**；
* **负样本 19 个**：库外样本（库里没有它们的知识）→ 指标：**放行了多少条**（每一条都是误报面），
  并按"同文件撞车 / 跨文件"拆开（跨文件才是纯误报）。

用法:
    python -X utf8 utils/experiments/a5_semantic_path_backtest.py --stage vector
    python -X utf8 utils/experiments/a5_semantic_path_backtest.py --stage llm --llm-samples 10 --llm-topk 3
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_libs"))
sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()

from utils.kb_coverage import SOURCE_EXT  # noqa: E402

LAYERS = ("semantic", "code_pattern", "solution", "full")
TAU = 0.65
DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"


def cos(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return num / (na * nb) if na and nb else 0.0


def basename_of(p: str) -> str:
    return str(p or "").replace("\\", "/").split("/")[-1].replace("__", "/").split("/")[-1].lower()


class KB:
    def __init__(self, dump: Path, db: Path):
        self.vecs, self.rows = {}, {}
        for line in dump.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            L = str(r.get("vector_layer") or "")
            if L not in LAYERS:
                continue
            sid = int(r["sqlite_id"])
            self.vecs[(sid, L)] = [float(x) for x in (r.get("_vector") or [])]
            self.rows.setdefault(sid, {}).update({
                "file_pattern": r.get("file_pattern") or "",
                "solution": r.get("solution") or "",
                "error_description": r.get("error_description") or "",
                "problematic_pattern": r.get("problematic_pattern") or "",
                "llm_semantic": r.get("llm_semantic") or "",
            })
        self.sids = sorted(self.rows)
        con = sqlite3.connect("file:%s?mode=ro" % db.as_posix(), uri=True)
        self.title_by_sid = {int(i): (t or "").strip().upper() for i, t in
                             con.execute("select id, title from issue_patterns")}
        self.own_of_cve = {v: k for k, v in self.title_by_sid.items()}
        con.close()


def pick_file(cve: str, kb_base: str, part: str = "before"):
    d = DS / part / cve
    if not d.is_dir():
        return None
    files = [f for f in sorted(d.rglob("*")) if f.is_file() and f.suffix.lower() in SOURCE_EXT]
    if not files:
        return None
    if kb_base:
        for f in files:
            if f.name.lower() == kb_base.lower():
                return f
    return files[0]


def build_queries(agent, kb: KB, cves, kind: str):
    """返回 [{cve, file, chunks:[{text, start_line}], sims:{sid:maxcos}, lexical:{sid:bool}}]"""
    out = []
    for cve in cves:
        own = kb.own_of_cve.get(cve.upper())
        kb_base = basename_of((kb.rows.get(own) or {}).get("file_pattern", "")) if own else ""
        f = pick_file(cve, kb_base)
        if not f:
            continue
        chunks = [c for c in agent._split_file_into_context_chunks(str(f))
                  if len(c["text"].strip()) >= 40]
        if not chunks:
            continue
        # 每块嵌 4 层
        qv = [{"text": c["text"], "start_line": c["start_line"],
               "vec": {L: agent._query_embed(c["text"], L) for L in LAYERS}} for c in chunks]
        sims, argmax = {}, {}
        for sid in kb.sids:
            best, best_i = 0.0, 0
            for i, c in enumerate(qv):
                for L in LAYERS:
                    v = kb.vecs.get((sid, L))
                    if v:
                        s = cos(c["vec"][L], v)
                        if s > best:
                            best, best_i = s, i
            sims[sid] = best
            argmax[sid] = best_i
        # 词元通道（离线复算）：该条的针是否在任何一块里连续命中
        lexical = {}
        toks = [(c["text"], agent._tokenize_code(c["text"])) for c in qv]
        for sid in kb.sids:
            frags = agent._extract_error_code_fragments((kb.rows.get(sid) or {}).get("solution", ""))
            lexical[sid] = bool(frags) and any(
                agent._is_contiguous_subseq(fr, tk) for _, tk in toks for fr in frags)
        out.append({"cve": cve, "kind": kind, "file": f.name, "own": own,
                    "chunks": qv, "sims": sims, "argmax": argmax, "lexical": lexical})
        print("  [%s] %-16s 块数 %2d  自己的条目=%s" % (kind, cve, len(qv), own))
    return out


def stats_of(sims):
    vals = list(sims.values())
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
    return mu, sd


def decide(item, kb: KB, rule: str, param: float):
    """返回该样本被放行的 sid 列表。"""
    sims, own = item["sims"], item["own"]
    mu, sd = stats_of(sims)
    ranked = sorted(sims.items(), key=lambda kv: -kv[1])
    if rule == "abs_tau":
        return [s for s, v in sims.items() if v >= param]
    if rule == "zscore":
        return [s for s, v in sims.items() if sd > 1e-9 and (v - mu) / sd >= param]
    if rule == "percentile":
        k = max(1, int(len(sims) * param))
        return [s for s, _ in ranked[:k]]
    if rule == "lexical":
        return [s for s, hit in item["lexical"].items() if hit]
    if rule == "fused":
        return [s for s, v in sims.items()
                if item["lexical"].get(s) or (sd > 1e-9 and (v - mu) / sd >= param)]
    if rule == "pair_llm_strict":
        return item.get("llm_same") or []
    if rule == "pair_llm_loose":
        return item.get("llm_loose") or []
    if rule == "additive":
        # **加法融合**：语义不是"另一条独立通路"，而是**抬高同一个分数**（用户指出的正确形式）。
        #   s_lex(x) ∈ {0,1}   词元通道是否命中（针在被分析代码里连续出现）
        #   s_sem(x) = clamp(z/4, 0, 1)   语义分按"该查询在全库上的分布"标准化后映射到 [0,1]
        #   score(x) = s_lex + λ·s_sem,   admit ⇔ score ≥ θ        ← **只有一个阈值**
        # 与 `fused`（并集）的实质差别：并集是**两个阈值各自判定**，语义可以单飞；
        # 加法里"语义单独救回"要求 λ·s_sem ≥ θ —— 语义必须**自己扛够权重**才作数。
        lam, theta = param
        out = []
        for s, v in sims.items():
            s_lex = 1.0 if item["lexical"].get(s) else 0.0
            z = (v - mu) / sd if sd > 1e-9 else 0.0
            s_sem = max(0.0, min(1.0, z / 4.0))
            if s_lex + lam * s_sem >= theta:
                out.append(s)
        return out
    raise ValueError(rule)


def evaluate(items, kb: KB, rule: str, param: float):
    pos = [i for i in items if i["kind"] == "pos"]
    neg = [i for i in items if i["kind"] == "neg"]
    own_hit = 0
    for i in pos:
        adm = decide(i, kb, rule, param)
        if i["own"] in adm:
            own_hit += 1
    neg_total = neg_same = neg_cross = 0
    for i in neg:
        adm = decide(i, kb, rule, param)
        neg_total += len(adm)
        fbase = basename_of(i["file"])
        for s in adm:
            kbase = basename_of((kb.rows.get(s) or {}).get("file_pattern", ""))
            if kbase and fbase and kbase == fbase:
                neg_same += 1
            else:
                neg_cross += 1
    return {"own_hit": own_hit, "pos_n": len(pos),
            "neg_adm": neg_total, "neg_same": neg_same, "neg_cross": neg_cross, "neg_n": len(neg)}


def score_pair_cross_encoder(current_chunk: str, kb_row: dict) -> float:
    """【留接口】真正的 cross-encoder 配对打分。

    ⚠️ **离线不可跑**：本机模型缓存里只有 `distilbert-base-uncased` / `gpt2` /
    `microsoft/codebert-base` / `Qwen1.5-7B-Chat`，**没有重排（cross-encoder）模型**。
    联网阶段可以补 `cross-encoder/ms-marco-MiniLM-L6-v2` 一类，或自训一个小的配对打分器。
    当前实现直接抛错，避免"看起来跑了其实没跑"。
    """
    raise NotImplementedError("离线无 cross-encoder 模型；见本函数 docstring")


def _load_qwen(device: str = ""):
    """加载本地 Qwen1.5-7B-Chat。

    `device`：""＝自动（有 CUDA 就用 GPU，否则 CPU）；也可显式 "cpu"/"cuda"，
    或用环境变量 `MAS_JUDGE_DEVICE` 覆盖。GPU 上用 float16（7B 约 15GB 显存），
    本地 CPU 用 float32（此前 float32 在本机占 23.6GB 内存，是本机跑不动的原因）。
    调用方必须把输入搬到 `model.device` 上（见 `_judge_once`）。
    """
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    if not device:
        device = os.environ.get("MAS_JUDGE_DEVICE") or (
            "cuda" if torch.cuda.is_available() else "cpu")
    mp = ROOT / "model_cache/models--Qwen--Qwen1.5-7B-Chat"
    snaps = sorted(mp.glob("snapshots/*"))
    path = str(snaps[-1]) if snaps else "Qwen/Qwen1.5-7B-Chat"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, local_files_only=True,
        torch_dtype=(torch.float16 if device.startswith("cuda") else torch.float32),
        device_map=device, low_cpu_mem_usage=True)
    model.eval()
    return tok, model, torch


JUDGE_PROMPT = """你是代码安全审查员。下面给出【当前代码块】和【知识库里的一条历史漏洞记录】。
请判断：这条历史记录描述的缺陷，与当前代码块里体现的问题，是不是**同一个缺陷**。

只输出一行，格式严格如下（不要解释、不要多余文字）：
VERDICT: same_defect 或 related 或 unrelated

【当前代码块】
%s

【知识库记录】
错误类型: %s
问题描述: %s
修复前代码/修法: %s
语义描述: %s
"""


def run_llm_judge(agent, kb: KB, items, samples: int, topk: int, kind: str = "both"):
    """让本地 Qwen 对"当前代码块 vs 库里的记录"做配对判定。

    判定分两档记录（这是实测出来的必要区分）：
    * **严格档**：只认 `same_defect` —— 第一次小剂量试跑发现，"related" 会把紧邻的**无关**记录
      也放进来（实测：同一条 sample 的 top-2 里，自己那条 SAME、另一条无关的 REL），
      所以"REL 也算命中"会把精度打没；
    * **宽松档**：`same_defect` + `related` —— 留作对照，看它到底多引入多少误报。
    """
    tok, model, torch = _load_qwen()
    tested = {"pos": 0, "neg": 0}
    for item in items:
        if kind != "both" and item["kind"] != kind:
            continue
        if tested[item["kind"]] >= samples:
            if tested["pos"] >= samples and tested["neg"] >= samples:
                break
            continue
        ranked = sorted(item["sims"].items(), key=lambda kv: -kv[1])[:topk]
        # 保证"自己那条"在里面（否则测不到正样本）
        if item["own"] and all(s != item["own"] for s, _ in ranked):
            ranked[-1] = (item["own"], item["sims"][item["own"]])
        admitted, loose = [], []
        for sid, sim in ranked:
            # 用**该候选取到最大相似度的那一块**做判断 —— 与向量口径一致（同一对样本）
            idx = int((item.get("argmax") or {}).get(str(sid), 0) or 0)
            chunks = item.get("chunks") or []
            prompt_chunk = (chunks[idx]["text"] if idx < len(chunks) else chunks[0]["text"])[:1500]
            row = kb.rows.get(sid) or {}
            p = JUDGE_PROMPT % (prompt_chunk, row.get("error_type_str") or "",
                                (row.get("error_description") or "")[:600],
                                (row.get("solution") or "")[:600],
                                (row.get("llm_semantic") or "")[:600])
            msgs = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=16, do_sample=False)
            reply = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            verdict = "unrelated"
            for cand in ("same_defect", "related", "unrelated"):
                if cand in reply:
                    verdict = cand
                    break
            tag = "SAME" if verdict == "same_defect" else ("REL" if verdict == "related" else "unrel")
            print("     [%s] %-16s sid=%-4d sim=%.3f base=%-28s → %s"
                  % (item["kind"], item["cve"], sid, sim,
                     basename_of((kb.rows.get(sid) or {}).get("file_pattern", ""))[:28], tag))
            if verdict == "same_defect":
                admitted.append(sid)
            if verdict in ("same_defect", "related"):
                loose.append(sid)
        item["llm_same"] = admitted
        item["llm_loose"] = loose
        tested[item["kind"]] += 1
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path,
                    default=ROOT / "reports/server_final_20261002/weaviate_kb_dump_postswitch.jsonl")
    ap.add_argument("--db", type=Path, default=ROOT / "reports/mas_rebuild_candidate_v2.db")
    ap.add_argument("--pos-runs", type=Path, default=ROOT / "reports/arm1_runs.txt")
    ap.add_argument("--stage", choices=("vector", "llm"), default="vector")
    ap.add_argument("--llm-samples", type=int, default=10)
    ap.add_argument("--llm-topk", type=int, default=3)
    ap.add_argument("--llm-kind", choices=("pos", "neg", "both"), default="both",
                    help="只让 judge 测正样本还是负样本（分开跑，便于单独量误报侧）")
    ap.add_argument("--json-out", type=Path, default=ROOT / "reports/a5_backtest.json")
    args = ap.parse_args()

    from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None

    kb = KB(args.dump, args.db)
    print("KB 条目 %d，索引 (条目,层) %d；查询偏移修正=%s" % (len(kb.sids), len(kb.vecs),
                                                            agent.query_offset_correction))

    pos_cves = [l.strip().split("/")[0] for l in args.pos_runs.read_text(encoding="utf-8").splitlines()
                if l.strip()]
    # 负样本：三个库外批次里的 CVE（本地配置里有 local_target_dir / before 路径）
    neg_cves = []
    for cfg in ("held_fp4.json", "held_overlap15.json"):
        p = ROOT / "utils/experiments" / cfg
        if p.is_file():
            for it in json.loads(p.read_text(encoding="utf-8"))["items"]:
                neg_cves.append(it["cve"])
    neg_cves = sorted(set(neg_cves))
    print("正样本 %d 个（kb-self）、负样本 %d 个（库外）" % (len(pos_cves), len(neg_cves)))

    cache = ROOT / "reports/a5_items_cache.json"
    if cache.is_file():
        items = json.loads(cache.read_text(encoding="utf-8"))
        # ⚠️ JSON 的键**一定是字符串**，读回来必须把 sid 键转回 int：
        # 否则 `item["sims"][54]` 直接 KeyError，而更隐蔽的是
        # `own in admitted`（int vs str）会**静默算成 0**，让人以为"语义命中为 0"。
        for it in items:
            it["sims"] = {int(k): v for k, v in (it.get("sims") or {}).items()}
            it["lexical"] = {int(k): v for k, v in (it.get("lexical") or {}).items()}
            it["argmax"] = {int(k): v for k, v in (it.get("argmax") or {}).items()}
        print("已加载缓存 %s（已把 sid 键归一化为 int）" % cache.name)
    else:
        t0 = time.time()
        items = build_queries(agent, kb, pos_cves, "pos") + build_queries(agent, kb, neg_cves, "neg")
        print("候选构建用 %.1f 秒" % (time.time() - t0))
        # 缓存：保留块文本（LLM judge 档要用），**丢掉向量**（太大）
        slim = []
        for it in items:
            c = dict(it)
            c["chunks"] = [{"text": ch["text"], "start_line": ch["start_line"]}
                           for ch in it.get("chunks") or []]
            slim.append(c)
        cache.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")

    if args.stage == "llm":
        run_llm_judge(agent, kb, items, args.llm_samples, args.llm_topk, args.llm_kind)
        rules = [("pair_llm_strict", 0.0), ("pair_llm_loose", 0.0)]
    else:
        rules = [("abs_tau", TAU), ("zscore", 2.0), ("zscore", 3.0),
                 ("percentile", 0.01), ("percentile", 0.05), ("lexical", 0.0),
                 ("fused", 2.0), ("fused", 3.0),
                 # **加法融合的 λ–θ 扫描**（用户要的形式）：λ = 语义能替代多少词元证据，
                 # θ = 唯一的阈值。扫出来的是一条"命中 vs 跨文件误报"的权衡曲线。
                 ("additive", (0.5, 1.0)), ("additive", (1.0, 1.0)), ("additive", (1.5, 1.0)),
                 ("additive", (2.0, 1.0)), ("additive", (3.0, 1.0)),
                 ("additive", (1.5, 1.5)), ("additive", (2.0, 2.0))]

    print("\n" + "=" * 100)
    print("A5 回测结果（正样本=自己那条能否放行；负样本=放行了几条，跨文件才是纯误报）")
    print("=" * 100)
    print("  %-22s %14s %10s %12s %12s" %
          ("配置", "正样本命中", "负放行", "其中同文件", "其中跨文件"))
    results = []
    for rule, param in rules:
        r = evaluate(items, kb, rule, param)
        if isinstance(param, (tuple, list)):
            label = "%s(λ=%.1f, θ=%.2f)" % (rule, param[0], param[1])
        else:
            label = "%s%s" % (rule, ("(=%.3f)" % param) if param else "")
        print("  %-22s %10d/%-3d %10d %12d %12d"
              % (label, r["own_hit"], r["pos_n"], r["neg_adm"], r["neg_same"], r["neg_cross"]))
        results.append({"rule": label, **r})
    args.json_out.write_text(json.dumps({"tau": TAU, "results": results}, ensure_ascii=False, indent=1),
                             encoding="utf-8")
    print("\n  怎么读：**正样本命中越高越好、跨文件放行越低越好**。")
    print("  `abs_tau` 是现状（复现基线）；`lexical` 是现有主力通道；`fused` 才是「语义词元互补」的形式。")
    print("\n明细已写入 %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
