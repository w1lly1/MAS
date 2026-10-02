#!/usr/bin/env bash
# GPU 实例开机后的状态侦察：确认机器、显存、磁盘、代码版本、关键文件是否都在。
# 只读脚本，不改任何东西。用法：bash /root/autodl-tmp/srv_recon_gpu.sh
set -uo pipefail
cd /root/autodl-tmp/MAS || exit 1

echo "=== 机器 ==="
hostname
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo "=== 磁盘 ==="
df -h /root/autodl-tmp | tail -1

echo "=== 代码版本 ==="
git log --oneline -1
git status --porcelain | head -5

echo "=== Python 环境 ==="
venv/bin/python -V
venv/bin/python - <<'PY'
import importlib
for name in ("torch", "transformers", "weaviate", "sentence_transformers"):
    try:
        m = importlib.import_module(name)
        print("  %-22s %s" % (name, getattr(m, "__version__", "?")))
    except Exception as e:
        print("  %-22s 缺失：%s" % (name, type(e).__name__))
try:
    import torch
    print("  cuda_available         %s" % torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  cuda_device            %s" % torch.cuda.get_device_name(0))
except Exception as e:
    print("  cuda 探测失败：%s" % e)
PY

echo "=== 模型缓存 ==="
ls -d model_cache/models--Qwen--Qwen1.5-7B-Chat 2>/dev/null || echo "  缺 Qwen1.5-7B-Chat"
ls -d model_cache/models--distilbert-base-uncased 2>/dev/null || echo "  缺 distilbert"

echo "=== A5/A5b 中间产物 ==="
for f in reports/a5_items_cache.json reports/a5b_llm_verdicts.json reports/a5_backtest.json; do
  if [ -f "$f" ]; then
    echo "  有 $f ($(stat -c %s "$f") 字节)"
  else
    echo "  无 $f"
  fi
done

echo "=== 知识库/服务 ==="
ls -d weaviate-data 2>/dev/null && echo "  weaviate-data 存在"
pgrep -af weaviate | head -2 || echo "  Weaviate 未运行"
ls -la data/*.db 2>/dev/null | head -5
