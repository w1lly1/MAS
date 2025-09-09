#!/bin/bash

# ChatGLM2-6B 测试运行脚本
# 确保在正确的虚拟环境中运行

set -e  # 遇到错误时停止

echo "🔧 ChatGLM2-6B 兼容性测试脚本"
echo "=================================="

# 检查当前目录
EXPECTED_DIR="/var/fpwork/tiyi/project/MAS/MAS"
CURRENT_DIR=$(pwd)

if [ "$CURRENT_DIR" != "$EXPECTED_DIR" ]; then
    echo "❌ 错误：必须在 $EXPECTED_DIR 目录下运行"
    echo "当前目录：$CURRENT_DIR"
    echo "🔄 自动切换到正确目录..."
    cd "$EXPECTED_DIR"
fi

echo "✅ 当前目录：$(pwd)"

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo "🔧 激活虚拟环境..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ 虚拟环境已激活：$VIRTUAL_ENV"
    else
        echo "❌ 找不到虚拟环境：venv/bin/activate"
        exit 1
    fi
else
    echo "✅ 虚拟环境已激活：$VIRTUAL_ENV"
fi

# 检查必要依赖
echo "🔍 检查必要依赖..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "❌ PyTorch未安装"
    exit 1
}

python -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
    echo "❌ Transformers未安装"
    exit 1
}

python -c "import sentencepiece; print(f'SentencePiece: {sentencepiece.__version__}')" || {
    echo "❌ SentencePiece未安装"
    exit 1
}

echo "✅ 所有依赖检查通过"

# 运行兼容性测试
echo ""
echo "🚀 开始ChatGLM兼容性测试..."
echo "=================================="

python tests/chatglm_tests/chatglm_compatibility_test.py

echo ""
echo "🏁 测试完成"
