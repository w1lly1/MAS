#!/bin/bash
# 同步服务器仓库到本地 git 最新提交（v2：处理"未跟踪文件挡路"）。
#
# 第一次失败的原因：快进会覆盖服务器上**未跟踪**的文件（那晚生成的 retry 配置、我 scp 上去的 scripts/ 等），
# git 拒绝执行，于是服务器停在旧提交上。做法：把这些挡路文件**移到备份目录**（不删），再快进。
#
# 这些文件既然"会被 incoming 覆盖"，说明它们**在仓库里也是被跟踪的** —— 所以仓库版本就是权威版本，
# 移动即可（备份目录保留原样，以便核对是否有人手工改过）。
set -u
BUNDLE="${1:?用法: srv_sync_code.sh <bundle 路径> <期望短 hash>}"
EXPECT="${2:?需要期望的短 hash}"
cd /root/autodl-tmp/MAS || exit 1
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="/root/autodl-tmp/_untracked_backup_$STAMP"

echo "===== 1) 同步前 ====="
echo "  HEAD: $(git rev-parse --short HEAD)"

echo
echo "===== 2) 找出挡路的未跟踪文件并移走 ====="
git fetch "$BUNDLE" main 2>&1 | tail -2
CONFLICT=$(git merge --ff-only FETCH_HEAD 2>&1 | sed -n '/untracked working tree files would be overwritten/,/^Please move/p' \
  | grep -E '^\s+\S' | sed 's/^\s*//' | grep -v '^Please')
if [ -n "$CONFLICT" ]; then
  mkdir -p "$BACKUP"
  echo "  挡路文件 $(echo "$CONFLICT" | wc -l) 个，移到 $BACKUP"
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    mkdir -p "$BACKUP/$(dirname "$f")"
    mv -f "$f" "$BACKUP/$f" 2>/dev/null && echo "    moved $f"
  done <<< "$CONFLICT"
  # 空目录也要清掉，否则 git 仍可能报冲突
  find . -type d -empty -not -path './.git/*' -delete 2>/dev/null
else
  echo "  没有未跟踪文件冲突"
fi

echo
echo "===== 3) 快进 ====="
git merge --ff-only FETCH_HEAD 2>&1 | tail -4

echo
echo "===== 4) 核对 ====="
NEW=$(git rev-parse --short HEAD)
echo "  HEAD: $NEW（期望 $EXPECT）"
[ "$NEW" = "$EXPECT" ] && echo "  ✅ 一致" || echo "  ❌ 不一致 —— 停下来人工看"
echo "  残留改动: $(git status --porcelain | wc -l) 项"
git status --porcelain | head -6
echo "  配置 sha256: $(sha256sum infrastructure/config/ai_agent_config.json | cut -c1-16)"
echo "  死键残留（应为 0）: $(grep -c 'enable_readability_enhancement_ai' infrastructure/config/ai_agent_config.json)"
echo "  T6 新增键（应 >=1）: $(grep -c 'tool_search_paths' infrastructure/config/ai_agent_config.json)"
echo "  备份目录: $BACKUP"
echo "  （stash 里的旧改动已被仓库版本取代，可丢弃：git stash drop）"
echo "SYNC2_DONE"
