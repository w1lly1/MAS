#!/bin/bash
# 修完"整层余弦表"之后的**库内单样本冒烟**：确认融合真的拿到了语义分。
#
# 判据（这次是"必须能看见语义原料"）：
#   ① 出现 fusion_score；
#   ② 出现**非零的** fusion_semantic_term（说明表里有相似度、没被钳成 0）；
#   ③ 若出现 gate_branch=fused_semantic，说明融合**真的多放行了**（这才是有收益）；
#      若没有出现，要看 fusion_score 与 θ 的差距有多大（是被否决挡住、还是分不够）。
set -u
cd /root/autodl-tmp/MAS || exit 1
LOG=/root/autodl-tmp/smoke_kbself1.log

echo "=== 1) 确认开关状态 ==="
bash /root/autodl-tmp/srv_set_gate_fusion.sh show 2>&1 | grep -E 'gate_fusion|weaviate_top_k|配置 sha'

echo
echo "=== 2) 跑 1 个库内样本（smoke_kb1.json = CVE-2018-8788）==="
date +%H:%M:%S
./venv/bin/python mas.py batch -c utils/experiments/smoke_kb1.json > "$LOG" 2>&1
echo "退出码=$?"

echo
echo "=== 3) 定位本次 run ==="
RUN=$(ls -dt reports/analysis/CVE-2018-8788/*/ | head -1)
echo "  $RUN"

echo
echo "=== 4) 融合是否拿到语义原料（按通道诊断）==="
ONERUN=$(echo "${RUN#reports/analysis/}" | sed 's:/$::')
echo "$ONERUN" > /tmp/one_run.txt
bash /root/autodl-tmp/srv_diag_fusion_terms.sh /tmp/one_run.txt || true
echo
bash /root/autodl-tmp/srv_diag_fusion_by_channel.sh /tmp/one_run.txt || true

echo
echo "=== 5) 该 run 的放行（new_findings）与分支 ==="
grep -rho '"gate_branch": "[a-z_]*"' "$RUN" 2>/dev/null | sort | uniq -c
grep -rho 'fused_semantic' "$RUN" 2>/dev/null | wc -l
echo "SMOKE_KBSELF_DONE"
