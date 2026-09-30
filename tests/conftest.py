# -*- coding: utf-8 -*-
"""pytest 公共夹具。

门控判定是纯逻辑，不需要向量库；但导入链上挂着 `import weaviate`。
本机没装真包时，整个门控测试套件会**连收集都过不去**（历史上测试网就这样长期失效）。
这里只在真包缺失时装一个占位实现，实现在 `utils/offline_imports.py`（离线脚本共用同一份）。
装了真包的机器（GPU 服务器）上本文件完全不生效。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.offline_imports import install_weaviate_stub  # noqa: E402

install_weaviate_stub()
