# -*- coding: utf-8 -*-
"""诊断消融瓶颈：分别计时 embed / Weaviate 查询 / 结构化匹配。"""
import asyncio
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent
from infrastructure.embeddings.codebert_embedder import embed_text

# 128 核无卡机器上，PyTorch 多线程会让 distilbert 前向传播急剧变慢（线程越多越慢），
# 实测 threads=1 最快（~159ms/次 vs 8线程~1.1s/次、16线程~18s/次）。显式压到 1 线程。
import os  # noqa: E402

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch  # noqa: F401

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


async def main():
    print("[step0] 构造 agent ...", flush=True)
    agent = AIDrivenSecondPassAnalysisAgent()
    print("[step1] initialize() 中（连 Weaviate，不加载 LLM）...", flush=True)
    await agent.initialize()
    print("[step2] initialize() 完成", flush=True)

    txt = "The snd_ctl_elem_add function in sound/core/control.c suffers from a buffer overflow" * 3
    snippet = "int snd_ctl_elem_add(struct snd_card *card) { if (!card) return -EINVAL; }"

    # 1) embed 计时（预热后）
    print("[step3] 预热 embed_text(distilbert) ...", flush=True)
    embed_text(txt, "semantic")
    print("[step4] 预热完成，开始计时", flush=True)
    t = time.time()
    for _ in range(10):
        embed_text(txt, "semantic")
    print(f"embed_text(distilbert) x10: {time.time()-t:.3f}s  ({(time.time()-t)/10*1000:.1f}ms/次)", flush=True)

    # 2) Weaviate 查询计时
    print("[step5] Weaviate 查询计时 ...", flush=True)
    vec = agent.vector_service
    qv = embed_text(txt, "semantic")
    t = time.time()
    for _ in range(10):
        vec.search_knowledge_items(query_vector=qv, limit=5, layer="semantic")
    print(f"Weaviate semantic 查询 x10: {time.time()-t:.3f}s  ({(time.time()-t)/10*1000:.1f}ms/次)", flush=True)

    # 3) 结构化匹配计时（SQLite issue_patterns + curated_issues 两条路径分别测）
    print("[step6] 拉取 patterns/curated ...", flush=True)
    sqlite_patterns = await agent.db_service.get_issue_patterns(status="active")
    curated = await agent.db_service.get_curated_issues()
    issue = {"description": txt, "file": "/tmp/x.c", "source": "security_ai", "line": None}
    t = time.time()
    for p in (sqlite_patterns or [])[:200]:
        agent._evaluate_pattern_match(p, txt, "security_ai", "/tmp/x.c", issue)
    print(f"SQLite 结构化匹配 200 pattern x1: {time.time()-t:.3f}s", flush=True)

    t = time.time()
    for c in (curated or [])[:327]:
        agent._match_curated_issue(c, issue, "/tmp/x.c")
    print(f"curated 结构化匹配 327 x1: {time.time()-t:.3f}s", flush=True)
    print("[done]", flush=True)

    await agent.stop()


asyncio.run(main())
