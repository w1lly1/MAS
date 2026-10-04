# -*- coding: utf-8 -*-
"""不许静默降级：向量编码器拿不到真实向量时必须**响亮地失败**（《03_踩过的坑》坑 44）。

背景（真实事故）：服务器批量跑批时 `HF_HOME` 没进非交互式 shell，transformers 找不到
本地模型 ⇒ `_ensure_distilbert` 失败 ⇒ 每次 `embed()` 都退化成 `_fallback_embed()` 的
768 维平凡向量。批次照常跑完、报告照常生成，但向量通道整体失效（135 个查询只命中
1 个不同结果集），由此报废了一整天的向量侧结论。

本测试锁死三条：
1. 默认（未显式放行）模型不可用 / 白化失败 ⇒ 抛 `EmbedderUnavailable`，**不返回平凡向量**；
2. 只有显式 `MAS_ALLOW_EMBED_FALLBACK=1` 才回到"平凡向量 + 响亮告警"的旧行为；
3. `assert_embedder_healthy()` 在模型不可用、向量全零、近乎常量时都拒绝放行。
"""
from __future__ import annotations

import pytest

from infrastructure.embeddings import codebert_embedder as ce


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """每个用例都从"干净编码器 + 未放行降级"出发，且不污染全局单例。

    必须清 lru_cache：`embed_text`/`_forward_raw` 都按文本缓存，
    否则上一个用例缓存下来的真实向量会让下一个用例"假通过"。
    """
    monkeypatch.delenv(ce.ALLOW_FALLBACK_ENV, raising=False)
    monkeypatch.setattr(ce, "_fallback_calls", 0)
    monkeypatch.setattr(ce, "_embedder", None, raising=False)
    ce.embed_text.cache_clear()
    ce._forward_raw.cache_clear()
    yield
    ce.embed_text.cache_clear()
    ce._forward_raw.cache_clear()


class _BrokenForward:
    """把前向打成"模型没加载"（返回 None）。"""

    def __init__(self, monkeypatch):
        monkeypatch.setattr(ce.TextEmbedder, "_forward_raw", lambda self, text: None)


class TestStrictByDefault:
    def test_model_unavailable_raises_instead_of_fallback(self, monkeypatch):
        _BrokenForward(monkeypatch)
        with pytest.raises(ce.EmbedderUnavailable) as ei:
            ce.embed_text("some issue text", "semantic")
        # 报错必须可操作：说清原因 + 说清怎么显式放行
        msg = str(ei.value)
        assert "distilbert" in msg.lower()
        assert ce.ALLOW_FALLBACK_ENV in msg

    def test_whitening_failure_also_raises(self, monkeypatch):
        monkeypatch.setattr(ce.TextEmbedder, "_forward_raw",
                            lambda self, text: [0.1] * ce.EMBED_DIM)
        monkeypatch.setattr(ce, "_apply_whitening",
                            lambda vec, layer: (_ for _ in ()).throw(ValueError("boom")))
        with pytest.raises(ce.EmbedderUnavailable) as ei:
            ce.embed_text("x", "semantic")
        assert "白化" in str(ei.value)

    def test_exception_is_a_runtime_error_so_nobody_swallows_it_silently(self):
        # 继承 RuntimeError：既能被通用兜底捕获（不至于让进程崩得莫名其妙），
        # 又不会被 `except (ValueError, KeyError)` 这类"预期错误"顺手吞掉。
        assert issubclass(ce.EmbedderUnavailable, RuntimeError)


class TestExplicitOptIn:
    def test_env_opt_in_returns_fallback_vector_and_warns(self, monkeypatch, capsys):
        _BrokenForward(monkeypatch)
        monkeypatch.setenv(ce.ALLOW_FALLBACK_ENV, "1")
        vec = ce.embed_text("fallback me", "semantic")
        assert len(vec) == ce.EMBED_DIM
        assert vec == ce._fallback_embed("fallback me")
        out = capsys.readouterr().out
        assert "降级" in out and ce.ALLOW_FALLBACK_ENV in out

    def test_warning_is_printed_once_not_per_vector(self, monkeypatch, capsys):
        _BrokenForward(monkeypatch)
        monkeypatch.setenv(ce.ALLOW_FALLBACK_ENV, "true")
        for i in range(5):
            ce.embed_text(f"text-{i}", "semantic")
        assert capsys.readouterr().out.count("[embedder][WARN]") == 1
        assert ce.embedder_status()["fallback_calls"] == 5

    def test_status_reports_degradation(self, monkeypatch):
        _BrokenForward(monkeypatch)
        assert ce.embedder_status()["fallback_allowed"] is False
        monkeypatch.setenv(ce.ALLOW_FALLBACK_ENV, "yes")
        ce.embed_text("t", "semantic")
        st = ce.embedder_status()
        assert st["fallback_allowed"] is True and st["fallback_calls"] == 1


class TestAssertHealthy:
    def test_refuses_when_model_unavailable(self, monkeypatch):
        _BrokenForward(monkeypatch)
        with pytest.raises(ce.EmbedderUnavailable):
            ce.assert_embedder_healthy()

    def test_refuses_probe_that_is_all_zero(self, monkeypatch):
        monkeypatch.setattr(ce.TextEmbedder, "_forward_raw",
                            lambda self, text: [0.0] * ce.EMBED_DIM)
        with pytest.raises(ce.EmbedderUnavailable) as ei:
            ce.assert_embedder_healthy()
        assert "全零" in str(ei.value)

    def test_refuses_probe_that_is_nearly_constant(self, monkeypatch):
        monkeypatch.setattr(ce.TextEmbedder, "_forward_raw",
                            lambda self, text: [0.05] * ce.EMBED_DIM)
        with pytest.raises(ce.EmbedderUnavailable) as ei:
            ce.assert_embedder_healthy()
        assert "常量" in str(ei.value)

    def test_refuses_wrong_dimension(self, monkeypatch):
        monkeypatch.setattr(ce.TextEmbedder, "_forward_raw", lambda self, text: [0.1, 0.2])
        with pytest.raises(ce.EmbedderUnavailable) as ei:
            ce.assert_embedder_healthy()
        assert "维度" in str(ei.value)

    def test_passes_with_real_looking_vector(self, monkeypatch):
        vec = [((i * 37) % 101) / 101.0 - 0.5 for i in range(ce.EMBED_DIM)]
        monkeypatch.setattr(ce.TextEmbedder, "_forward_raw", lambda self, text: vec)
        # 白化也打成恒等（本用例只验自检流程，不验白化数学）
        monkeypatch.setattr(ce, "_apply_whitening", lambda v, layer: list(v))
        st = ce.assert_embedder_healthy()
        assert st["model"] == ce.DISTILBERT
        assert st["probe_norm_ok"] is True
