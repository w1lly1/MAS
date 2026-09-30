# -*- coding: utf-8 -*-
"""离线跑门控逻辑时，给缺失的第三方包装占位实现。

## 为什么需要它

门控判定（`_gate_candidate` / `_unified_structured_score` / `_candidate_code_fixed` /
`_error_code_clone_matched` / `_match_curated_issue`）全是纯逻辑：输入候选字典，输出判定，
**不碰向量库、不碰网络、不碰 GPU**。

但导入 `core.agents.ai_driven_second_pass_analysis_agent` 的链条上挂着
`infrastructure.database.weaviate.service`，它在**模块顶层**就 `import weaviate`。
于是只要本机没装 `weaviate` 包（做门控改动的开发机、轻量 CI 容器），
**连导入都过不去** —— 门控测试与离线预筛都跑不起来，只能上 GPU 服务器，
而这正是过去"改了判据却没人验证"的根源之一。

## 行为约定

* 真包存在 → 直接返回，**不做任何事**（GPU 服务器上的行为与之前完全一致）。
* 真包缺失 → 装占位模块。门控逻辑用到的类/函数**不会**被调用；
  一旦有代码真去连向量库（`connect_to_local` 等），会立刻抛错，
  而不是静默走假数据——避免"看起来跑通了其实是空壳"。
"""
from __future__ import annotations

import sys
import types


def install_weaviate_stub() -> bool:
    """装了占位实现返回 True；真包已在（未改动）返回 False。"""
    try:
        import weaviate  # noqa: F401
        return False
    except Exception:
        pass

    class _Missing(Exception):
        pass

    def _boom(*args, **kwargs):
        raise RuntimeError(
            "weaviate 是占位实现（本机未安装真包）。"
            "门控逻辑不应触发向量库调用；若触发了，说明该路径确实依赖外部服务。"
        )

    def _mkmod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        # 必须设 __file__/__path__：否则 inspect 拿不到模块文件，测试收集阶段就会崩
        m.__file__ = "<weaviate-stub:%s>" % name
        m.__path__ = []  # type: ignore[attr-defined]

        def _ga(attr):
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            return _boom

        m.__getattr__ = _ga  # type: ignore[attr-defined]
        return m

    root = _mkmod("weaviate")
    root.WeaviateClient = type("WeaviateClient", (), {})
    root.Client = type("Client", (), {})
    root.connect_to_local = _boom
    root.connect_to_weaviate_cloud = _boom
    root.connect_to_custom = _boom
    root.auth = _mkmod("weaviate.auth")
    root.auth.AuthApiKey = type("AuthApiKey", (), {})
    root.exceptions = _mkmod("weaviate.exceptions")
    root.exceptions.WeaviateConnectionError = _Missing
    root.classes = _mkmod("weaviate.classes")
    root.classes.config = _mkmod("weaviate.classes.config")
    root.classes.config.Property = type("Property", (), {"__init__": lambda self, **kw: None})
    root.classes.config.DataType = type("DataType", (), {})
    root.classes.query = _mkmod("weaviate.classes.query")
    root.classes.query.Filter = type("Filter", (), {})
    for name, mod in (
        ("weaviate", root),
        ("weaviate.auth", root.auth),
        ("weaviate.exceptions", root.exceptions),
        ("weaviate.classes", root.classes),
        ("weaviate.classes.config", root.classes.config),
        ("weaviate.classes.query", root.classes.query),
    ):
        sys.modules.setdefault(name, mod)
    return True
