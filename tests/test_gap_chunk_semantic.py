# -*- coding: utf-8 -*-
"""补漏（gap）通道接上首轮语义：**在 80% 的查询上**把"原始代码查询"换成"语义查询"。

背景（《01》问题 0 之二）：补漏通道约占线上查询量 80%，但它一直是拿分片的**原始代码**去查，
而索引侧是英文散文 —— 两边模态不对齐。首轮其实已经写出了正确语域的语义描述，
只是从来没传到这条通道上。

改动只有一件事：给分片拼 issue 字典时，**按行号区间**把首轮的 `llm_semantic` 接过来。
`_build_query_text` 本来就优先用这个字段，所以不需要改查询构造。

这批测试盯住四件事：
1. 行号落在分片区间内 → 语义被接上，且查询文本走语义分支；
2. **不许张冠李戴**：文件不同、或行号不在区间内 → 一律不接；
3. 一个区间里有多个候选 → 取离分片起点最近的那个（确定性）；
4. **fail-open**：没有语义时行为与改动前一模一样（查询文本仍是原始代码形态）。
"""
import unittest

from core.agents.ai_driven_second_pass_analysis_agent import AIDrivenSecondPassAnalysisAgent

SEM = "The function parses the packet header without validating the declared length, risking out-of-bounds read."


def _agent():
    agent = AIDrivenSecondPassAnalysisAgent()
    agent._debug_log = lambda *a, **k: None
    return agent


class TestGapChunkSemanticAttachment(unittest.TestCase):
    def test_attaches_semantic_when_line_inside_chunk_range(self):
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 100, "end_line": 160, "text": "int parse(void) { ... }"}
        issues = [{"file": "/src/net/packet.c", "line": 120, "llm_semantic": SEM, "llm_family": "input_validation"}]
        lookup = agent._semantic_lookup_from_issues(issues)
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        self.assertEqual(out["llm_semantic"], SEM)
        self.assertEqual(out["llm_family"], "input_validation")
        self.assertEqual(out["semantic_from"], "first_pass_line_overlap")

    def test_query_text_switches_to_semantic_branch(self):
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 100, "end_line": 160, "text": "RAW_CODE_SHOULD_NOT_APPEAR"}
        lookup = agent._semantic_lookup_from_issues(
            [{"file": "/src/net/packet.c", "line": 130, "llm_semantic": SEM, "llm_family": "input_validation"}])
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        text = agent._build_query_text(out, "/src/net/packet.c")
        self.assertIn("out-of-bounds", text)            # 语义描述进来了
        self.assertNotIn("RAW_CODE_SHOULD_NOT_APPEAR", text)   # 原始代码预览不再进查询
        self.assertIn("error_type: input_validation", text)

    def test_does_not_attach_semantic_from_another_file(self):
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 100, "end_line": 160, "text": "x"}
        lookup = agent._semantic_lookup_from_issues(
            [{"file": "/src/other/thing.c", "line": 120, "llm_semantic": SEM}])
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        self.assertNotIn("llm_semantic", out)

    def test_does_not_attach_semantic_when_line_outside_range(self):
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 100, "end_line": 160, "text": "x"}
        lookup = agent._semantic_lookup_from_issues(
            [{"file": "/src/net/packet.c", "line": 400, "llm_semantic": SEM}])
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        self.assertNotIn("llm_semantic", out)

    def test_picks_candidate_closest_to_chunk_start(self):
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 100, "end_line": 200, "text": "x"}
        lookup = agent._semantic_lookup_from_issues([
            {"file": "/src/net/packet.c", "line": 190, "llm_semantic": "FAR"},
            {"file": "/src/net/packet.c", "line": 104, "llm_semantic": "NEAR"},
        ])
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        self.assertEqual(out["llm_semantic"], "NEAR")

    def test_issues_without_semantic_are_ignored(self):
        agent = _agent()
        lookup = agent._semantic_lookup_from_issues([
            {"file": "/src/net/packet.c", "line": 120, "description": "no semantic here"},
            {"file": "/src/net/packet.c", "line": 130, "llm_semantic": "   "},
        ])
        self.assertEqual(lookup, [])

    def test_line_taken_from_line_number_alias(self):
        """首轮 issue 的行号字段名不止一个，别名也要认。"""
        agent = _agent()
        lookup = agent._semantic_lookup_from_issues(
            [{"file": "/a/b.c", "line_number": "150", "llm_semantic": SEM}])
        self.assertEqual(lookup[0]["line"], 150)

    def test_fail_open_without_any_semantic(self):
        """没有语义时，查询文本必须**与改动前一致**（原始代码形态）。"""
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 10, "end_line": 20, "text": "int raw(void) {}"}
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=[])
        self.assertNotIn("llm_semantic", out)
        text = agent._build_query_text(out, "/src/net/packet.c")
        self.assertIn("source_code_chunk L10-20", text)
        self.assertIn("int raw", text)

    def test_switch_off_restores_history(self):
        """配置开关关掉后，即使传了 issues 也不接语义（供 A/B 用）。"""
        agent = _agent()
        agent.gap_chunk_semantic_lookup = False
        chunk = {"file": "/src/net/packet.c", "start_line": 100, "end_line": 160, "text": "x"}
        issues = [{"file": "/src/net/packet.c", "line": 120, "llm_semantic": SEM}]
        # 开关是在 _collect_gap_evidence_from_code_chunks 里生效的，这里直接验证取值语义
        lookup = (agent._semantic_lookup_from_issues(issues)
                  if agent.gap_chunk_semantic_lookup else [])
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        self.assertNotIn("llm_semantic", out)

    # ---- 首轮"块行区间"口径：这是实测必须走的那条路（首轮 line_number 一直是 None） ----

    def test_uses_chunk_line_span_when_no_line_number(self):
        """首轮只给了块的行区间、没有 line_number 时也必须能接上（真实产出的样子）。"""
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 40, "end_line": 90, "text": "x"}
        lookup = agent._semantic_lookup_from_issues([{
            "file": "/src/net/packet.c", "line": None, "line_number": None,
            "chunk_start_line": 1, "chunk_end_line": 45,   # 与 40-90 重叠
            "llm_semantic": SEM, "llm_family": "input_validation",
        }])
        self.assertEqual(lookup[0]["span_start"], 1)
        self.assertEqual(lookup[0]["span_end"], 45)
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        self.assertEqual(out["llm_semantic"], SEM)

    def test_non_overlapping_span_is_not_attached(self):
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 200, "end_line": 260, "text": "x"}
        lookup = agent._semantic_lookup_from_issues([{
            "file": "/src/net/packet.c", "chunk_start_line": 1, "chunk_end_line": 45,
            "llm_semantic": SEM,
        }])
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        self.assertNotIn("llm_semantic", out)

    def test_overlap_picks_closest_span(self):
        agent = _agent()
        chunk = {"file": "/src/net/packet.c", "start_line": 100, "end_line": 140, "text": "x"}
        lookup = agent._semantic_lookup_from_issues([
            {"file": "/src/net/packet.c", "chunk_start_line": 1, "chunk_end_line": 105, "llm_semantic": "FAR"},
            {"file": "/src/net/packet.c", "chunk_start_line": 99, "chunk_end_line": 150, "llm_semantic": "NEAR"},
        ])
        out = agent._code_chunk_as_issue(chunk, semantic_lookup=lookup)
        self.assertEqual(out["llm_semantic"], "NEAR")


if __name__ == "__main__":
    unittest.main()
