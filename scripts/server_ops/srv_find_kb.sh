#!/usr/bin/env bash
# 找知识库落盘位置：Weaviate 数据目录 + SQLite 知识库。只读。
set -uo pipefail
echo "=== /root/autodl-tmp 一级目录 ==="
ls -la /root/autodl-tmp | head -20
echo
echo "=== 全盘找 weaviate-data 目录 ==="
find /root/autodl-tmp -maxdepth 4 -name "weaviate-data" -not -path "*/venv/*" 2>/dev/null
echo
echo "=== 找 .db 文件（排除 venv/site-packages） ==="
find /root/autodl-tmp -maxdepth 4 -name "*.db" -not -path "*/venv/*" -not -path "*/site-packages/*" -printf "%10s  %p\n" 2>/dev/null | head -20
echo
echo "=== 仓库里的知识库相关目录 ==="
ls -d /root/autodl-tmp/MAS/*/ 2>/dev/null | head -25
