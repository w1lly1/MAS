#!/bin/bash
# 把服务器仓库同步到**本地 git 的最新提交**（服务器连不上 GitHub，所以用 git bundle）。
#
# 为什么用 bundle 而不是逐个 scp 文件：bundle 自带提交历史，
# 服务器同步后 `git log` 是真的、`git status` 是干净的，
# 以后再排查"服务器跑的到底是哪版代码"就有据可查（逐个覆盖文件会留下一堆"已修改"的假象）。
#
# 同步前后都做检查：备份配置、打印两边版本、同步后验证 HEAD 与期望一致。
#
# 用法: bash /root/autodl-tmp/srv_sync_code.sh /root/autodl-tmp/mas_repo_20261002.bundle d82d6f2
set -u
BUNDLE="${1:?用法: srv_sync_code.sh <bundle 路径> <期望的短 hash>}"
EXPECT="${2:?需要期望的短 hash（用于同步后核对）}"
cd /root/autodl-tmp/MAS || exit 1

echo "===== 1) 同步前 ====="
echo "  当前 HEAD: $(git rev-parse --short HEAD)"
echo "  工作区改动:"; git status --porcelain | head -8

echo
echo "===== 2) 备份服务器上的配置与改动 ====="
STAMP=$(date +%Y%m%d_%H%M%S)
cp -f infrastructure/config/ai_agent_config.json "/root/autodl-tmp/ai_agent_config.server_$STAMP.json"
git stash push -u -m "server-before-sync-$STAMP" \
  -- core/agents/analysis_result_summary_agent.py infrastructure/config/ai_agent_config.json 2>&1 | tail -2
echo "  配置备份: /root/autodl-tmp/ai_agent_config.server_$STAMP.json"

echo
echo "===== 3) 从 bundle 拉取并快进 ====="
git fetch "$BUNDLE" main 2>&1 | tail -3
git merge --ff-only FETCH_HEAD 2>&1 | tail -3

echo
echo "===== 4) 同步后核对 ====="
echo "  新 HEAD: $(git rev-parse --short HEAD)"
if [ "$(git rev-parse --short HEAD)" = "$EXPECT" ]; then
  echo "  ✅ HEAD 与期望一致（$EXPECT）"
else
  echo "  ❌ HEAD 与期望不一致（期望 $EXPECT）—— 先别继续，人工看一下"
fi
echo "  工作区改动:"; git status --porcelain | head -8
echo "  配置文件 sha256: $(sha256sum infrastructure/config/ai_agent_config.json | cut -c1-16)"
echo "  新配置里是否还有被删的死键（应为 0）: $(grep -c 'enable_readability_enhancement_ai' infrastructure/config/ai_agent_config.json)"
echo "  新配置里是否有 T6 新增的 search path 键（应 >=1）: $(grep -c 'tool_search_paths' infrastructure/config/ai_agent_config.json)"
echo "SYNC_OK"
