#!/bin/bash
# 语法检查 + 自匹配隐患扫描（一次性，2026-10-04 坑 49 之后）
set -u
cd /root/autodl-tmp || exit 1
fail=0
for f in srv_wait_batch.sh srv_chain_arm.sh srv_chain_rest.sh srv_start_batch.sh \
         srv_chain_held_live.sh srv_chain_lambda_live.sh srv_wait_then_eval_arm.sh \
         srv_wait_then_compare_arm.sh srv_chain_lam3.sh srv_tonight.sh \
         srv_arm_done.sh srv_arm_progress.sh srv_arm_samples_state.sh srv_arm_status.sh \
         srv_health.sh srv_state_now.sh srv_shutdown_prep.sh srv_clean30_status.sh ; do
  if [ -f "$f" ]; then
    if bash -n "$f" 2>/tmp/_syn.txt; then echo "  OK   $f"; else echo "  FAIL $f : $(head -2 /tmp/_syn.txt)"; fail=1; fi
  else
    echo "  缺失 $f"
  fi
done
echo
echo "--- 仍含自匹配模式（pgrep -f 'mas.py batch'）的非注释行 ---"
grep -n "pgrep -f 'mas.py batch'" *.sh 2>/dev/null | grep -v '^[^:]*:[0-9]*: *#' | grep -v srv_wait_batch || echo "  无（已清干净）"
echo
echo "SYNTAX_CHECK_DONE fail=$fail"
