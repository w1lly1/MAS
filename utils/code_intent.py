# -*- coding: utf-8 -*-
"""确定性「代码功能意图」文本构造器（C3 修复用）

问题：gap 通道占检索量约 80%，其查询向量是【原始 C 源码片段】，而索引侧
`code_pattern`/`full` 层文本是【散文/元数据】（error_description、problematic_pattern、
file_pattern、class_pattern、language、solution）。查询是代码、索引是散文 —— 跨模态失配。
且固定窗口切片会把大量 license/copyright 样板文本带进查询，进一步稀释语义。

修法（确定性、无 LLM，符合论文"二次阶段禁止自由生成式缺口发现"的红线）：
把代码片段压成"功能意图骨架"：
  [code_intent] file:.. | lang:..
  [funcs] 函数名/签名
  [api]   被调用的高风险 C/系统 API + 主要被调用名
  [control] 控制流关键字
  [types] struct/enum/typedef 名 + _t 类型 + 大写宏
  [comments] 去掉 license/版权样板后的注释文本

用法：
    from utils.code_intent import build_code_intent
    text = build_code_intent(code_chunk, file_path="/path/to/foo.c")
"""
from __future__ import annotations

import os
import re
from typing import List

MAX_COMMENT_CHARS = 420
MAX_API_NAMES = 24
MAX_CALL_NAMES = 18
MAX_FUNCS = 10
MAX_TYPES = 14
MAX_TOTAL_CHARS = 1000
MAX_FALLBACK_CHARS = 420

# 许可证/版权样板特征词（命中即丢弃该注释块）
_BOILERPLATE = re.compile(
    r"copyright|licen[cs]e|all rights reserved|redistribut|warrant|"
    r"merchantab|free software foundation|gnu |bsd |apache|mit licen|"
    r"permission is hereby granted|disclaimer|see the .*licen|"
    r"__start_of|__end_of|this file is part of|modification|either version",
    re.IGNORECASE,
)

# 高风险 / 语义强 C 与系统 API（漏洞模式高度相关）
_RISKY_APIS = {
    "memcpy", "memmove", "memset", "bcopy", "strcpy", "strncpy", "strcat", "strncat",
    "sprintf", "vsprintf", "snprintf", "vsnprintf", "scanf", "sscanf", "fscanf",
    "gets", "fgets", "getc", "fgetc", "read", "pread", "write", "pwrite",
    "fread", "fwrite", "fopen", "open", "close", "lseek", "mmap", "munmap",
    "malloc", "calloc", "realloc", "free", "alloca", "valloc", "memalign",
    "strlen", "strnlen", "strcmp", "strncmp", "strstr", "strchr", "strtol", "atoi", "atol",
    "index", "rindex", "bsearch", "qsort", "memcmp",
    "system", "popen", "exec", "execl", "execv", "fork", "setuid", "setgid",
    "getenv", "putenv", "chroot", "chdir", "unlink", "rename", "stat", "fstat",
    "socket", "accept", "connect", "recv", "recvfrom", "send", "sendto", "bind", "listen",
    "printf", "fprintf", "vfprintf", "syslog", "perror", "assert", "abort", "exit",
    "pthread_create", "pthread_mutex_lock", "sem_wait", "signal", "sigaction",
    "kfree", "kmalloc", "vmalloc", "copy_from_user", "copy_to_user", "get_user", "put_user",
    "spin_lock", "spin_unlock", "mutex_lock", "mutex_unlock", "rcu_read_lock",
    "free_skb", "skb_copy", "printk", "pr_devel", "pr_err", "pr_info", "WARN_ON", "BUG_ON",
}

_CTRL = ["if", "else", "for", "while", "do", "switch", "case", "goto", "return", "break", "continue"]

_COMMENT_BLOCK = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_STRING_LIT = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_FUNC_DEF = re.compile(
    r"^[ \t]*(?:[A-Za-z_][\w \t\*]*?[\s\*]+)([A-Za-z_]\w*)\s*\([^;{)]*\)[^;{]*\{",
    re.MULTILINE,
)
_STRUCT = re.compile(r"\b(?:struct|enum|union)\s+([A-Za-z_]\w*)")
_TYPEDEF = re.compile(r"\btypedef\b[^;]*?\b([A-Za-z_]\w*)\s*;", re.DOTALL)
_TYPE_T = re.compile(r"\b([A-Za-z_]\w*_t)\b")
_MACRO = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")
_WS = re.compile(r"[ \t]+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PUNCT_ONLY = re.compile(r"^[\W_]+$")

# 这些词紧跟 "(" 时不是函数调用，应排除
_NOT_CALLS = set(_CTRL) | {
    "if", "for", "while", "switch", "sizeof", "return", "defined", "do", "else",
    "int", "char", "void", "long", "short", "unsigned", "signed", "float", "double",
    "struct", "union", "enum", "const", "static", "inline", "extern", "register",
    "volatile", "class", "public", "private", "protected", "template", "typename",
    "new", "delete", "catch", "throw", "try", "namespace", "using", "operator",
    "and", "or", "not", "true", "false", "nullptr", "NULL",
}


def _strip_noise(code: str) -> str:
    """去掉字符串字面量、【块注释】与行注释，避免把注释/消息文本里的词当标识符。"""
    s = code or ""
    s = _STRING_LIT.sub('""', s)
    s = _COMMENT_BLOCK.sub(" ", s)
    s = _LINE_COMMENT.sub("", s)
    return s


def extract_comments(code: str) -> str:
    """抽取非样板注释文本；license/版权块整块丢弃，分隔线与 URL 一并清理。"""
    kept: List[str] = []

    def _clean(body: str) -> str:
        body = _WS.sub(" ", body.replace("\n", " ")).strip()
        body = re.sub(r"^\*+", "", body).strip()
        return body

    def _usable(body: str) -> bool:
        if len(body) < 12 or _BOILERPLATE.search(body) or _PUNCT_ONLY.match(body):
            return False
        # 去掉纯分隔线/框线片段后仍需有内容
        core = _URL.sub("", body)
        core = re.sub(r"[-=+*#/|_~]{2,}", " ", core)
        return len(core.strip()) >= 10

    for m in _COMMENT_BLOCK.finditer(code or ""):
        body = _clean(m.group(0)[2:-2])
        if _usable(body):
            kept.append(_URL.sub("", body).strip())
    for m in _LINE_COMMENT.finditer(code or ""):
        body = _clean(m.group(0)[2:])
        if _usable(body):
            kept.append(_URL.sub("", body).strip())
    joined = " | ".join(kept)
    return joined[:MAX_COMMENT_CHARS]


def _ordered_unique(items) -> List[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_code_intent(code: str, file_path: str = "", lang: str = "") -> str:
    """把代码片段压成确定性「功能意图骨架」文本。"""
    code = code or ""
    stripped = _strip_noise(code)

    calls = _ordered_unique([c for c in _CALL.findall(stripped) if c not in _NOT_CALLS])
    risky = [c for c in calls if c in _RISKY_APIS]
    other = [c for c in calls if c not in _RISKY_APIS]
    funcs = _ordered_unique([f for f in _FUNC_DEF.findall(stripped) if f not in _NOT_CALLS])
    ctrl = [k for k in _CTRL if re.search(r"\b%s\b" % re.escape(k), stripped)]
    types = _ordered_unique(
        _STRUCT.findall(stripped) + _TYPEDEF.findall(stripped)
        + _TYPE_T.findall(stripped) + _MACRO.findall(stripped)
    )
    comments = extract_comments(code)

    if not lang:
        ext = os.path.splitext(str(file_path or ""))[1].lower().lstrip(".")
        lang = {"c": "C", "h": "C", "cpp": "C++", "cc": "C++", "cxx": "C++",
                "py": "Python", "java": "Java"}.get(ext, ext or "unknown")

    parts = ["[code_intent] file: %s | lang: %s" % (os.path.basename(str(file_path or "")) or "-", lang)]
    if funcs:
        parts.append("[funcs] " + ", ".join(funcs[:MAX_FUNCS]))
    if risky:
        parts.append("[api_risky] " + ", ".join(risky[:MAX_API_NAMES]))
    if other:
        parts.append("[api] " + ", ".join(other[:MAX_CALL_NAMES]))
    if ctrl:
        parts.append("[control] " + ", ".join(ctrl))
    if types:
        parts.append("[types] " + ", ".join(types[:MAX_TYPES]))
    if comments:
        parts.append("[comments] " + comments)

    # 兜底：整段是许可证/版权样板（无任何 funcs/api/control/types/comments 信号）时，
    # 骨架会退化成只有标题行 → 查询近乎空内容，比原始代码更差。
    # 此时补一段「去注释后的精简代码」作为下限，保证查询始终携带可检索内容。
    if len(parts) == 1:
        body = _WS.sub(" ", _strip_noise(code).replace("\n", " ")).strip()
        if body:
            parts.append("[raw] " + body[:MAX_FALLBACK_CHARS])
    return "\n".join(parts)[:MAX_TOTAL_CHARS]


def build_code_intent_compact(code: str, file_path: str = "", max_chars: int = 260) -> str:
    """紧凑版「功能意图」：**不带任何标签、不带文件/语言头**，只有词元本身。

    用途（C3 v2 的「增强」模式）：把这段文字**追加**到原始代码查询后面，
    而不是替换它。关键约束是**尽量少引入所有查询共有的恒定字符**——
    因为恒定字符会把查询重新推向雷同（详见《03_踩过的坑.md》坑 9），
    而"查询间相似度"正是本项目的早期预警指标。

    与 build_code_intent 的区别：那个是"替换型"骨架（带 [funcs]/[api_risky] 等标签，
    实测会让查询间相似度从 ~0 升到 0.3+）；这个是"追加型"词元串。
    """
    code = code or ""
    stripped = _strip_noise(code)
    calls = _ordered_unique([c for c in _CALL.findall(stripped) if c not in _NOT_CALLS])
    risky = [c for c in calls if c in _RISKY_APIS]
    other = [c for c in calls if c not in _RISKY_APIS]
    funcs = _ordered_unique([f for f in _FUNC_DEF.findall(stripped) if f not in _NOT_CALLS])
    types = _ordered_unique(_STRUCT.findall(stripped) + _TYPE_T.findall(stripped))
    toks: List[str] = []
    toks += funcs[:6]
    toks += risky[:10]
    toks += other[:8]
    toks += types[:4]
    comments = extract_comments(code)
    if comments:
        toks.append(_WS.sub(" ", comments)[:100])
    out = " ".join(t for t in toks if t)
    return out[:max_chars]
