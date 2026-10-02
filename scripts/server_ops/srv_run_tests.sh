#!/bin/bash
# 环境一致性核对：在服务器上跑完整测试套件，把失败集合落盘，
# 以便和本地基线（此前为 45 failed / 8 errors）逐条比对。
# 注意：服务器上有些工具二进制（pylint/bandit 等）与本地不同，失败集合**本来就可能有差异**，
#       差异要人工归类，不能直接当成"代码不一致"。
set -u
cd /root/autodl-tmp/MAS || exit 1
mkdir -p reports
echo "核心数: $(nproc)"
nohup venv/bin/python -X utf8 -u -m pytest tests -q --tb=no -p no:cacheprovider \
  > reports/server_pytest.log 2>&1 &
echo "pid=$!"
