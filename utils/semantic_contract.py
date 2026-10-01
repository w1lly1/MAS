# -*- coding: utf-8 -*-
"""**语义描述契约**：让"分析时写的描述"和"知识库里的描述"用同一套规则。

## 为什么要有这个模块（而不是各处再写一份）

离线实验已经证明：索引侧是英文散文，而查询侧以前送的是"原始代码 + 中文模板标签"，
两者不在同一个语义空间 —— 用代码查，正确条目只有 47% 进 top-5；
换成**同语言同语域**的描述后到 81%（见《02》第十六节）。

而这个契约需要在**三个地方**保持一致：
  ① 分析时（`ai_driven_security_agent` 产出描述）；
  ② 入库时（`utils/bigvul_ingest` 的 `llm_semantic` 旁挂）；
  ③ 离线实验脚本（`utils/experiments/gen_llm_semantic.py`）。
本项目已经吃过"同一逻辑两份实现、后来跑偏"的亏（层文本构造器就是），
所以**契约、解析、校验都只在这里实现一份**，别处 import 使用。

## 契约内容

模型必须输出**两行**：

    error_type: <七类之一，小写>
    <两句英文：合并成一行>

* 第一句 = 这段代码**做什么**（点到函数名与它操作的数据）；
* 第二句 = 这里**可能出什么问题**，句式为 `... may/can/could ...`；
* 只许英文（编码器 `distilbert-base-uncased` 是英文模型，中文基本进不了同一个空间）；
* 不许出现 CVE 编号、版本号、文件名（那些是"抄答案的线索"）。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# 知识库的错误分类体系（与 utils/bigvul_ingest/rules.py 的 VALID_ERROR_TYPES 同源）
FAMILIES = (
    "input_validation",
    "memory_overflow",
    "resource_exhaustion",
    "race_condition",
    "authorization_bypass",
    "dos",
    "general",
)

# 给模型看的家族清单（一行一个 + 一句解释），与 derive_problematic_pattern 的句式呼应
FAMILY_LIST = """input_validation      external input consumed without strict bounds or format validation
memory_overflow       unchecked arithmetic or index usage causing an out-of-bounds access
resource_exhaustion   allocation path lacking defensive limits or cleanup
race_condition        shared state updated without synchronization or ordering checks
authorization_bypass  security-critical capability checks incomplete or bypassable
dos                   error handling allowing repeated attacker-driven state transitions or loops
general               security-sensitive logic lacking explicit defensive checks"""

# 提示词模板（英文契约）。占位符：{family_list} {code_snippet} {language} {function} {hints}
SEMANTIC_PROMPT = """You are given a code excerpt from a real project. Describe THIS code for a vulnerability knowledge base.

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
{code_snippet}

Two lines:"""

FAMILY_LINE_RE = re.compile(r"^\s*error_type\s*[:=]\s*([a-z_]+)\s*$", re.I | re.M)
# 模型很喜欢把提示里的措辞抄成输出标签（实测 15/15 都写了 "Sentence 1:"），
# 所以统一剥掉，而不是指望它听话。
LEADING_LABEL_RE = re.compile(
    r"^\s*(?:sentence\s*[ab12]\s*[:.\-]\s*|function\s*[:.\-]\s*|risk\s*[:.\-]\s*"
    r"|description\s*[:.\-]\s*|\d+\s*[).]\s*|[-*\u2022]\s*)+", re.I)
# 句中也会冒出 "Sentence B:"（实测 14 条里 4 条）。注意别写成 `\b sentence`：
# `\b` 是零宽断言，放在空格前面要求"空格左侧是词字符"，而句中标签恰在句号之后 → 永不匹配。
INLINE_LABEL_RE = re.compile(r"(?:sentence\s*[ab12]\s*[:.\-]\s*"
                             r"|function\s*[:.\-]\s*|\brisk\s*[:.\-]\s*)", re.I)
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")
CVE_RE = re.compile(r"CVE-\d{4}-\d+", re.I)
VERSION_RE = re.compile(r"\b(?:before|after|through|prior to)\s+\d+\.\d+", re.I)
RISK_CUES = ("may ", "can ", "could ", "leads to", "allows ", "without ", "missing ",
             "unchecked", "not validated", "no bounds", "out-of-bounds", "overflow",
             "not protected", "unsynchronized", "bypass", "unbounded", "fails to",
             "does not ", "fails ", "insufficient", "lack")


def render_hints(hints: Optional[List[Dict]]) -> str:
    """把首轮已经指出的可疑位置渲染进提示。

    **只喂"分析真的产出的"线索**：流水线里 `db_supplemented` 那类记录是**检索命中知识库
    之后**才生成的（内容就是 CVE 摘要原文），拿它当提示等于先把答案告诉模型 ——
    循环论证。调用方负责过滤掉它们。
    """
    if not hints:
        return ""
    lines = []
    for h in hints[:6]:
        loc = ("L%s" % h.get("line")) if h.get("line") else "(no line)"
        lines.append("  %-8s %-22s %s" % (loc, str(h.get("source") or "")[:22],
                                          str(h.get("text") or "")[:150]))
    return ("\nEarlier analysis flagged these spots in this file. They may be incomplete, "
            "unrelated, or wrong - use them only as a hint:\n" + "\n".join(lines) + "\n")


def split_family(text: str) -> Tuple[str, str]:
    """取出模型选定的家族，并把那一行从正文里剥掉。返回 (family, 正文)。

    家族缺失或不在七类里时返回 `("", 原文)`，由调用方决定回退策略
    （分析侧回退到"不写家族"，知识库侧回退到规则分类）。
    """
    raw = text or ""
    fam = ""
    m = FAMILY_LINE_RE.search(raw)
    if m:
        cand = m.group(1).strip().lower()
        if cand in FAMILIES:
            fam = cand
        raw = FAMILY_LINE_RE.sub(" ", raw)
    return fam, raw


def clean_description(text: str) -> str:
    """整理成"功能句 + 风险句"。

    做法：剥标签 → 按句切 → **第一句当功能句**，再从后面挑**第一句带风险词**的当风险句。
    为什么不直接取前两句：实测模型会把前两句都写成"这段代码在做什么"，
    真正的风险句被推到第三句，直接截前两句会把风险信息整个丢掉。
    """
    t = " ".join(str(text or "").strip().split())
    t = LEADING_LABEL_RE.sub("", t).strip()
    t = " ".join(INLINE_LABEL_RE.sub(" ", t).split())
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", t) if p.strip()]
    if not parts:
        return ""
    first = parts[0] if parts[0].endswith((".", "!", "?")) else parts[0] + "."
    second = next((c for c in parts[1:] if any(k in c.lower() for k in RISK_CUES)), "")
    if not second and len(parts) > 1:
        second = parts[1]
    return (first + " " + second).strip() if second else first


def parse_contract(text: str) -> Tuple[str, str]:
    """把模型回复解析成 `(family, 两句描述)`。这是**唯一**的契约解析入口。"""
    fam, body = split_family(text)
    return fam, clean_description(body)


def compliance(text: str, file_pattern: str = "") -> Dict[str, object]:
    """格式合规检查：不达标就不该写进库（坏文本会污染整层向量）。"""
    t = str(text or "").strip()
    words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", t))
    return {
        "empty": not t,
        "has_cjk": bool(CJK_RE.search(t)),
        "has_cve": bool(CVE_RE.search(t)),
        "has_version": bool(VERSION_RE.search(t)),
        "mentions_path": bool(file_pattern and file_pattern.split("/")[-1] in t),
        "words": words,
        "sentences": len([s for s in re.split(r"[.!?]\s+", t) if s.strip()]),
        "risk_cue": any(c in t.lower() for c in RISK_CUES),
    }


def is_acceptable(text: str, file_pattern: str = "", min_words: int = 20,
                  max_words: int = 140) -> bool:
    """是否达到入库标准（英文、无抄答案线索、长度合理）。"""
    c = compliance(text, file_pattern)
    return not (c["empty"] or c["has_cjk"] or c["has_cve"] or c["has_version"]
                or c["mentions_path"]) and min_words <= c["words"] <= max_words
