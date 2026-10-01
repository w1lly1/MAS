# -*- coding: utf-8 -*-
"""用大模型为知识库的每条知识生成 `llm_semantic`（大模型语义理解）。

## 为什么要这一步

离线实验已经证明：**用代码去查散文索引，正确条目只有 34% 能进 top-5（平均排到 47 名）；
换成同语言同语域的散文，100% 进 top-5、平均 1.5 名**（见《02》第十六节）。
而索引侧原本的散文（`error_description` / `problematic_pattern`）是**规则模板 + CVE 摘要**，
没有"从代码出发的语义"。本脚本就是把这块补上。

## 输出契约（这是"两侧能对上"的关键，不是随便写句话）

生成文本会被放进 `[llm_semantic] …` 小节，**只进 semantic / full 两层**。它必须满足：

1. **只写英文** —— 编码器 `distilbert-base-uncased` 是英文模型，中文基本进不了同一个空间；
2. **两句话、45–90 词** —— 第一句"这段代码在做什么"（点名函数与它操作的数据），
   第二句"风险模式"，语域对齐本库既有的 `problematic_pattern`（例如
   "Unchecked arithmetic or index usage may cause out-of-bounds access."）；
3. **不许出现 CVE 编号、版本号、文件路径** —— 那些是"抄答案的线索"：
   离线实验里的"改写版"正是靠删掉它们才成为一次**严格**检验（仍 78% 进 top-5）。
   真实分析时模型也拿不到这些，所以契约必须一致。

字段值**只放两句话本体**，不带 `[error_type]` 之类的标签 —— 标签由层构造器加，
并且 `error_type` 由条目自己的列决定（索引侧保证分类口径与库内完全一致）。

## 用法（GPU 服务器上跑）

    # 先小批验证格式（强烈建议）
    python utils/experiments/gen_llm_semantic.py --limit 15 --out reports/llm_semantic_probe15.json
    # 看输出格式没问题再全量
    python utils/experiments/gen_llm_semantic.py --out reports/llm_semantic.json
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.kb_coverage import SOURCE_EXT  # noqa: E402

DS = ROOT / "tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured"

SYSTEM = (
    "You write precise technical descriptions for a vulnerability knowledge base. "
    "You describe code excerpts factually. You never guess beyond what the excerpt shows."
)

# 7 类弱点家族**就是知识库的错误分类体系**（见 utils/bigvul_ingest/rules.py 的
# derive_error_type 与 derive_problematic_pattern）。把它作为**候选列表**交给模型选，
# 而不是让模型自由发明分类：两侧分类空间不一致的话，向量再像也对不上号（《02》第十六节）。
FAMILY_LIST = """input_validation      external input consumed without strict bounds or format validation
memory_overflow       unchecked arithmetic or index usage causing an out-of-bounds access
resource_exhaustion   allocation path lacking defensive limits or cleanup
race_condition        shared state updated without synchronization or ordering checks
authorization_bypass  security-critical capability checks incomplete or bypassable
dos                   error handling allowing repeated attacker-driven state transitions or loops
general               security-sensitive logic lacking explicit defensive checks"""

PROMPT = """You are given a code excerpt from a real project. Describe THIS code for a vulnerability knowledge base.

Step 1 - classify. Choose EXACTLY ONE weakness family from this fixed list:
{family_list}

Step 2 - describe. Write exactly two sentences:
- Sentence A: what this code DOES. Name the function and the buffer, array or structure it works on.
- Sentence B: what can GO WRONG here, in the style of this example:
    "Unchecked arithmetic or index usage may cause out-of-bounds access."
  It must contain a modality word such as may / can / could / allows / without / fails to.

Output format - EXACTLY two lines, nothing else, no markdown:
error_type: <one family name from the list, lowercase>
<the two sentences on one line>

Hard rules:
- ENGLISH only, plain ASCII. Never use Chinese or other non-ASCII characters.
- Do NOT write labels such as "Sentence A", "Sentence B", "Function:" or "Risk:" in the sentences.
- Never mention CVE identifiers, version numbers, or file paths.
- Describe only what the excerpt below actually shows.
{family_hint}{hints}
Language: {language}
Function: {function}

Code excerpt:
{code}

Two lines:"""

FAMILY_HINT = {
    "input_validation": "input validation - missing bounds or format checks",
    "memory_overflow": "memory safety - out-of-bounds access or integer overflow",
    "resource_exhaustion": "resource exhaustion - missing limits or missing cleanup",
    "race_condition": "race condition - unsynchronized shared state",
    "authorization_bypass": "authorization bypass - incomplete or bypassable check",
    "dos": "denial of service - unbounded loop or repeated attacker-driven state change",
    "general": "a defensive check missing on a security-sensitive path",
}


def render_hints(hints) -> str:
    """把首轮分析指出的可疑位置渲染进提示。

    **必须只喂"分析真的产出的"线索**：流水线里 `db_supplemented` 那类记录是
    **第二轮检索命中知识库之后**才生成的（内容就是 CVE 摘要原文），拿它当提示等于
    先把答案告诉模型再去检索 —— 循环论证，会把效果测虚高。调用方负责过滤。
    """
    if not hints:
        return ""
    lines = []
    for h in hints[:6]:
        loc = ("L%s" % h.get("line")) if h.get("line") else "(no line)"
        src = str(h.get("source") or "")[:22]
        lines.append("  %-8s %-22s %s" % (loc, src, str(h.get("text") or "")[:150]))
    return ("\nEarlier analysis flagged these spots in this file. They may be incomplete, "
            "unrelated, or wrong - use them only as a hint:\n" + "\n".join(lines) + "\n")

CJK = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
# 模型很喜欢把提示里的措辞抄成输出标签（实测 15 条里 15 条都写了 "Sentence 1:"），
# 所以在后处理里统一剥掉，而不是指望它听话。
LABEL_RE = re.compile(
    r"^\s*(?:sentence\s*[ab12]\s*[:.\-]\s*|function\s*[:.\-]\s*|risk\s*[:.\-]\s*"
    r"|description\s*[:.\-]\s*|\d+\s*[).]\s*|[-*\u2022]\s*)+", re.I)
# 模型不仅会在**开头**写 "Sentence 1:"，也会在**句中标** "Sentence B:"（实测：第一版只剥开头，
# 结果 14 条里有 4 条把 "Sentence B:" 留在了正文中间）。所以还要做一次全局清理。
# 注意别写成 `\b sentence`：`\b` 是零宽断言，放在空格**前面**要求"空格左侧是词字符"，
# 而句中标签恰恰出现在句号之后（`.` 与空格之间没有词边界）→ 永远匹配不上。
# 标签词本身就够独特，直接匹配即可，多余空格随后统一压缩。
INLINE_LABEL_RE = re.compile(r"(?:sentence\s*[ab12]\s*[:.\-]\s*"
                             r"|function\s*[:.\-]\s*|\brisk\s*[:.\-]\s*)", re.I)
CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.I)
VER_RE = re.compile(r"\b(?:before|after|through|prior to)\s+\d+\.\d+", re.I)
RISK_CUES = ("may ", "can ", "could ", "leads to", "allows ", "without ", "missing ",
             "unchecked", "not validated", "no bounds", "out-of-bounds", "overflow",
             "not protected", "unsynchronized", "bypass", "unbounded", "fails to",
             "does not ", "fails ", "insufficient", "lack")

FAMILIES = ("input_validation", "memory_overflow", "resource_exhaustion",
            "race_condition", "authorization_bypass", "dos", "general")
FAMILY_LINE_RE = re.compile(r"^\s*error_type\s*[:=]\s*([a-z_]+)\s*$", re.I | re.M)


def split_family(reply: str) -> tuple:
    """从回复里取出模型选定的弱点家族，并把它从正文里剥掉。

    返回 (family, 去掉家族行之后的正文)。家族没给或不在 7 类里时返回 ("", 原文) ——
    调用方决定回退策略。要求模型显式选一个家族，是为了让查询侧带上**同一套分类编号**，
    否则两侧的分类空间对不上（向量像也没用）。
    """
    text = reply or ""
    fam = ""
    m = FAMILY_LINE_RE.search(text)
    if m:
        cand = m.group(1).strip().lower()
        if cand in FAMILIES:
            fam = cand
        text = FAMILY_LINE_RE.sub(" ", text)
    return fam, text


def clean_reply(reply: str) -> str:
    """把模型回复整理成"功能句 + 风险句"。

    做法：剥标签 → 按句切 → **第一句当功能句**，再从后面挑**第一句带风险词**的当风险句。
    为什么不直接取"前两句"：实测模型会把前两句都写成"这段代码在做什么"，
    真正的风险句被推到第三句，直接截前两句会把风险信息整个丢掉（这正是第一版的问题）。
    """
    text = " ".join((reply or "").strip().split())
    text = LABEL_RE.sub("", text).strip()
    text = INLINE_LABEL_RE.sub(" ", text)
    text = " ".join(text.split())
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    if not parts:
        return ""
    first = parts[0]
    if not first.endswith((".", "!", "?")):
        first += "."
    second = ""
    for cand in parts[1:]:
        if any(c in cand.lower() for c in RISK_CUES):
            second = cand
            break
    if not second and len(parts) > 1:
        second = parts[1]
    return (first + " " + second).strip() if second else first


def vulnerable_window(code: str, solution: str, width: int = 2400) -> str:
    """取"包含漏洞代码的那一段"给模型看。

    为什么不是整个文件：大文件（如 amalgamation）塞不进上下文，而且离漏洞很远的
    代码会把描述带偏。定位方式与离线实验一致 —— 用库里 `Remove incorrect logic`
    的第一行去文件里找位置，再取它周围的窗口。
    """
    frag = re.search(r"Remove incorrect logic:\s*(.+?)(?:\.\s*Ensure corrected path:|$)",
                     solution or "", re.DOTALL)
    needle = frag.group(1).split(";")[0].strip()[:60] if frag else ""
    lines = code.splitlines()
    if not lines:
        return ""
    idx = next((i for i, ln in enumerate(lines) if needle and needle[:40] in ln),
               len(lines) // 2)
    half = max(4, width // 70)
    lo = max(0, idx - half // 2)
    return "\n".join(lines[lo:lo + half])[:width]


def code_of(cve: str, ds_root: Path, max_bytes: int = 2_000_000) -> str:
    d = ds_root / "before" / cve
    if not d.exists():
        return ""
    buf = []
    for f in sorted(d.rglob("*")):
        try:
            if f.is_file() and f.suffix.lower() in SOURCE_EXT and f.stat().st_size <= max_bytes:
                buf.append(f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(buf)


def compliance(text: str, file_pattern: str) -> dict:
    """格式合规检查 —— 不达标就不该入库（宁缺毋滥：坏文本会污染整层向量）。"""
    t = (text or "").strip()
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", t))
    sentences = len([s for s in re.split(r"[.!?]\s+", t) if s.strip()])
    return {
        "has_cjk": bool(CJK.search(t)),
        "has_cve": bool(CVE_RE.search(t)),
        "has_version": bool(VER_RE.search(t)),
        "mentions_path": bool(file_pattern and Path(file_pattern).name in t),
        "words": words,
        "sentences": sentences,
        "risk_cue": any(c in t.lower() for c in RISK_CUES),
        "empty": not t,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=ROOT / "infrastructure/database/mas.db")
    ap.add_argument("--dataset-root", type=Path, default=DS)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/llm_semantic.json")
    ap.add_argument("--model", default="Qwen/Qwen1.5-7B-Chat")
    ap.add_argument("--limit", type=int, default=0, help="只生成前 N 条（0=全部），用于小批验证")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--cves", default="", help="逗号分隔，只生成这些 CVE")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt，不加载模型")
    ap.add_argument("--code-json", type=Path, default=None,
                    help="**查询侧模式**：直接给 [{key, cve, code, language, function, hints}] 生成描述，"
                         "不查库、不从数据集读源码。用于模拟『分析时模型看到分片后写出的描述』。")
    ap.add_argument("--infer-family", action="store_true",
                    help="查询侧模式用：不给弱点家族的权威答案，让模型从 7 类里自己选"
                         "（分析时本来就不知道答案）")
    ap.add_argument("--families-out", type=Path, default=None,
                    help="把模型**选定的家族**另存一份，便于统计与知识库分类的一致性")
    args = ap.parse_args()

    # ---------------- 查询侧模式：输入是给定的代码片段 ----------------
    if args.code_json:
        items = json.loads(args.code_json.read_text(encoding="utf-8"))
        if isinstance(items, dict):
            items = [dict(v, cve=k) for k, v in items.items()]
        if args.limit:
            items = items[: args.limit]
        jobs = []
        for it in items:
            code = str(it.get("code") or "")
            # 查询侧：**不给**家族的权威答案，让模型自己从 7 类里选（真实情形就是不知道）
            fam = ("Choose it yourself from the list above."
                   if args.infer_family
                   else "The library classifies this entry as: %s. Use it unless the code clearly contradicts it."
                        % (str(it.get("family") or "unknown")))
            jobs.append({
                "id": None, "key": str(it.get("key") or it.get("cve") or "?"),
                "cve": str(it.get("cve") or it.get("id") or "?"),
                "file": str(it.get("file") or ""), "error_type": str(it.get("family") or ""),
                "prompt": PROMPT.format(family_list=FAMILY_LIST, family_hint=fam,
                                        hints=render_hints(it.get("hints")),
                                        language=str(it.get("language") or "C"),
                                        function=str(it.get("function") or "(unknown)"),
                                        code=code[:2400]),
                "chars": len(code[:2400]),
            })
        print("查询侧模式：待生成 %d 条（平均 %d 字符，带定位提示的 %d 条）"
              % (len(jobs), sum(j["chars"] for j in jobs) // max(1, len(jobs)),
                 sum(1 for it in items if it.get("hints"))))
    else:
        con = sqlite3.connect(str(args.db))
        rows = [(i, (t or "").strip().upper(), (fp or ""), (cp or ""), (et or ""),
                 (lang or ""), (sol or ""))
                for i, t, fp, cp, et, lang, sol in con.execute(
                    "select id, title, file_pattern, class_pattern, error_type, language, solution "
                    "from issue_patterns order by id")]
        con.close()
        if args.cves:
            want = {c.strip().upper() for c in args.cves.split(",") if c.strip()}
            rows = [r for r in rows if r[1] in want]
        rows = rows[args.start:]
        if args.limit:
            rows = rows[: args.limit]
        print("待生成 %d 条（模型 %s）" % (len(rows), args.model))

        # 组 prompt
        jobs = []
        for _id, cve, fp, cp, et, lang, sol in rows:
            code = code_of(cve, args.dataset_root)
            if not code:
                print("  [跳过] %-16s 读不到源码" % cve)
                continue
            window = vulnerable_window(code, sol)
            prompt = PROMPT.format(
                family_list=FAMILY_LIST,
                # 索引侧：告诉它库里的分类，保证 llm_semantic 的散文与层文本里的
                # `[error_type]` 那一行**不自相矛盾**；同时仍让它显式输出所选的家族，
                # 便于统计"模型判断"与"规则分类"的一致率。
                family_hint="The library classifies this entry as: %s. Use it unless the code "
                            "clearly contradicts it." % (et or "general"),
                hints="",
                language=lang or "C", function=cp or "(unknown)",
                code=window or code[:2000],
            )
            jobs.append({"id": _id, "cve": cve, "key": cve, "file": fp, "error_type": et,
                         "prompt": prompt, "chars": len(window or code[:2000])})
        print("  实际生成 %d 条（平均代码窗口 %d 字符）"
              % (len(jobs), sum(j["chars"] for j in jobs) // max(1, len(jobs))))

    if args.dry_run:
        print("\n=== 第一条 prompt（预览）===")
        if jobs:
            print(jobs[0]["prompt"][:1600])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    # 必须把模型名**解析成本地快照目录**再加载。直接传 "Qwen/Qwen1.5-7B-Chat" 时，
    # 新版 transformers 的 tokenizer 会在 from_pretrained 内部调一次
    # `_patch_mistral_regex → is_base_mistral → model_info()` **打网络**，
    # 而本机是 TRANSFORMERS_OFFLINE=1（流水线就这么配的），于是直接抛
    # OfflineModeIsEnabled —— 看起来像"模型没缓存"，其实缓存好好的。
    # 项目自己的 agent 也是先解析快照路径再加载（见 ai_driven_database_manage_agent）。
    model_path = args.model
    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(repo_id=args.model, local_files_only=True)
        print("使用本地快照: %s" % model_path)
    except Exception as e:  # noqa: BLE001
        print("⚠️ 解析本地快照失败，回退按模型名加载: %s" % e)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,
                                        local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True, local_files_only=True)
    model.eval()
    print("模型加载完成 %.1f s" % (time.time() - t0))

    out, bad, fams = {}, [], {}
    t0 = time.time()
    for i, j in enumerate(jobs, 1):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": j["prompt"]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=args.temperature > 0,
                                 temperature=max(args.temperature, 1e-5),
                                 top_p=0.9, pad_token_id=tok.eos_token_id)
        reply = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        family, body = split_family(reply)          # 先摘掉模型选定的家族行
        content = clean_reply(body)
        chk = compliance(content, j["file"])
        ok = not (chk["empty"] or chk["has_cjk"] or chk["has_cve"]
                  or chk["has_version"] or chk["mentions_path"]) and 20 <= chk["words"] <= 140
        if ok:
            key = j.get("key") or j["cve"]
            out[key] = content
            fams[key] = family or (j.get("error_type") or "")
        else:
            bad.append({"cve": j["cve"], "text": content, "raw": reply.strip()[:200], "why": chk})
        if i % 10 == 0 or i == len(jobs):
            print("  %3d/%d  用时 %.1f min  已通过 %d  不合格 %d"
                  % (i, len(jobs), (time.time() - t0) / 60, len(out), len(bad)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n写出 %d 条 -> %s" % (len(out), args.out))
    if args.families_out:
        args.families_out.parent.mkdir(parents=True, exist_ok=True)
        args.families_out.write_text(json.dumps(fams, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
        agree = sum(1 for j in jobs
                    if fams.get(j.get("key") or j["cve"]) and j.get("error_type")
                    and fams.get(j.get("key") or j["cve"]) == j["error_type"])
        with_fam = sum(1 for j in jobs if j.get("error_type"))
        print("家族选择写出 -> %s" % args.families_out)
        if with_fam:
            print("  与库里规则分类一致: %d / %d = %.1f%%"
                  % (agree, with_fam, 100 * agree / with_fam))

    print("\n=== 格式合规汇总 ===")
    print("  通过 %d / %d = %.1f%%" % (len(out), len(jobs), 100 * len(out) / max(1, len(jobs))))
    if bad:
        print("  不合格样例（前 5 条）:")
        for b in bad[:5]:
            print("    %-16s %s" % (b["cve"], str(b["why"])))
            print("        %s" % b["text"][:160])
    if out:
        print("\n=== 合格样例 ===")
        for cve in list(out)[:3]:
            print("  %-16s %s" % (cve, out[cve][:220]))


if __name__ == "__main__":
    main()
