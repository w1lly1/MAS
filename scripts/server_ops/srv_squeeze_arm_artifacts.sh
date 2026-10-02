#!/bin/bash
# 腾空间（**保留证据**）：把三臂的完整产物目录压成一个 tgz，校验通过后再删原目录。
#
# 为什么要这么做：held 批次正在跑，磁盘从 7.3G 往下掉（每样本约 0.2G），
# 而 Weaviate 在 90% 会把自己切成只读 —— 中途爆盘比"事后压缩"严重得多。
#
# 两个安全措施：
#   1) `nice -n 19 ionice -c3`：把压缩的优先级压到最低，避免和正在跑的批处理抢 CPU/IO
#      （吃过亏：一边跑批一边跑吃 CPU 的诊断脚本，批处理静默少做了一大段，见《03》坑 14）；
#   2) **先压缩、先校验、再删除**：校验"条目数 + 抽样文件 sha256 逐字节一致"，
#      任何一步不对就停下不动原目录。
#
# 用法: nohup bash /root/autodl-tmp/srv_squeeze_arm_artifacts.sh > /root/autodl-tmp/squeeze.log 2>&1 &
set -u

ROOT=/root/autodl-tmp/MAS/reports/arm_artifacts
OUT=/root/autodl-tmp/arm_artifacts_full_20261002.tgz
DIRS="arm1_newkb_newcode arm2_baseline_oldkb arm3_ablate_oldkb"

cd "$ROOT" || exit 1
echo "[$(date +%H:%M:%S)] 压缩前：$(du -sh "$ROOT" | cut -f1)，磁盘可用 $(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')MB"

echo "[$(date +%H:%M:%S)] 开始压缩（低优先级）"
nice -n 19 ionice -c3 tar czf "$OUT" $DIRS
echo "[$(date +%H:%M:%S)] 压缩完成：$(du -sh "$OUT" | cut -f1)"

echo "[$(date +%H:%M:%S)] 校验"
N=$(tar tzf "$OUT" | wc -l)
echo "  归档条目数: $N"
SAMPLE="arm1_newkb_newcode/batch_summary.csv"
BEFORE=$(sha256sum "$SAMPLE" | awk '{print $1}')
AFTER=$(tar xzf "$OUT" -O "$SAMPLE" | sha256sum | awk '{print $1}')
echo "  抽样文件 $SAMPLE"
echo "    原文件 sha256: $BEFORE"
echo "    归档内 sha256: $AFTER"

if [ "$BEFORE" = "$AFTER" ] && [ "$N" -gt 100 ]; then
  echo "[$(date +%H:%M:%S)] 校验通过 → 删除原目录（证据已在归档里）"
  rm -rf $DIRS
  echo "  删除后：$(du -sh "$ROOT" | cut -f1)，磁盘可用 $(df -m /root/autodl-tmp | tail -1 | awk '{print $4}')MB"
  echo "SQUEEZE_OK"
else
  echo "*** 校验不通过（abort）：**不删**原目录，请人工看"
  echo "SQUEEZE_FAILED"
fi
