#!/bin/bash
# 服务器拉取（带重试 + HTTP/1.1）：实测 https 拉取偶发
# "RPC failed; curl 16 Error in the HTTP2 framing layer"，直接重试往往就好。
set -u
cd /root/autodl-tmp/MAS || exit 1
export GIT_HTTP_VERSION=HTTP/1.1

for i in 1 2 3 4 5; do
  echo "--- 第 $i 次 fetch ---"
  if git fetch origin 2>&1 | tail -2; then
    if git merge --ff-only origin/main 2>&1 | tail -2; then
      break
    fi
  fi
  sleep 5
done

echo
echo "HEAD: $(git log --oneline -1)"
echo "dirty: $(git status --short | wc -l)"
echo "白名单修复是否到位（应为 2）：$(grep -c 'chunk_start_line' core/agents/analysis_result_summary_agent.py)"
