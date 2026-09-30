# BigVul / CVE-2006-5331 验收说明
#
# 代码侧已完成：
# 1) ingest 填充 file_pattern / class_pattern / 漏洞窗口 snippet / diff solution / curated status=resolved
# 2) second-pass 查询 curated open+resolved；文件/函数锚定加分；gap 行号与去重
# 3) 报告展示知识摘要与锚定函数
#
# 要使 run 对 traps.c 真正生效，必须重灌知识库后再分析：
#
#   # 1. 重新生成 structured ingest（已写入 utils/bigvul_ingest/output/structured_ingest_top20_sample.json）
#   python -m utils.bigvul_ingest.build_structured_ingest --output-name structured_ingest_top20_sample.json --start 0 --count 20
#
#   # 2. 按项目现有流程将该 JSON 灌入 SQLite，并同步 Weaviate（DatabaseIngest / db manage agent）
#   #    注意：旧库中 CVE-2006-5331 仍是空 file_pattern + 通用 solution，不重灌则二轮仍难锚定。
#
#   # 3. 重跑分析目标：
#   #    tests/BigVul/.../before/CVE-2006-5331/6c4841c2/arch__powerpc__kernel__traps.c
#
# 期望 r2 报告：
# - 描述含 altivec_unavailable_exception 或 traps.c CVE 摘要（不只是 "dos"）
# - 行号落在约 901–914
# - 建议动作含 CONFIG_ALTIVEC / SIGILL 相关修复语义
# - 同文件无关 memory_overflow 不应压过本 CVE（structured 锚定更高）
