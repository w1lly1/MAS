#!/bin/bash
# 看服务器上那次全量测试的结果（用于和本地基线 45 failed / 8 errors 比对）。
set -u
cd /root/autodl-tmp/MAS || exit 1
echo "pytest 进程数: $(pgrep -c pytest)"
echo "--- 汇总行 ---"
grep -E '[0-9]+ (passed|failed)' reports/server_pytest.log | tail -2
echo "--- 失败用例清单（最多 60 行）---"
grep -E '^(FAILED|ERROR)' reports/server_pytest.log | head -60
echo "失败行数: $(grep -cE '^(FAILED|ERROR)' reports/server_pytest.log)"
