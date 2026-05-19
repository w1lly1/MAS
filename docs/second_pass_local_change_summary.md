Summary of planned local changes for second-pass analysis agent

This commit adds a short summary describing the intended local code modifications before implementing them.

Planned modifications (concise):

- `analysis_result_summary_agent.py` (_forward_to_readability_enhancement): include `original_analysis` (a trimmed map of the raw analysis outputs collected by SummaryAgent) in the message content when forwarding consolidated reports to the second-pass agent.

- `ai_driven_second_pass_analysis_agent.py`:
  - `handle_message`: read `original_analysis` from incoming message.content and pass it into the second-pass pipeline.
  - `_run_second_pass`: accept an additional `original_analysis` parameter and forward it to gap-discovery and evidence collection as needed.
  - `_llm_gap_discovery`: accept `original_analysis` parameter and include a truncated/summary form in LLM prompt context to enable gap discovery using raw analysis outputs (e.g., static/security findings, AI analysis recommendations).
  - `_collect_evidence` / `_derive_new_findings`: optionally consult `original_analysis` snippets (summarized) for additional evidence, with strict truncation to control token usage.

- `infrastructure/config/prompts.py` (or prompt templates): update the `second_pass_gap_discovery` prompt to accept a placeholder for `original_analysis` JSON/summarized text and ensure the agent truncates or summarizes it before insertion.

- Tests and validation:
  - Add unit test(s) to ensure `SummaryAgent` forwards `original_analysis` and `AIDrivenSecondPassAnalysisAgent` receives it and includes it in gap-discovery prompts (use a stubbed text generator to inspect prompts).
  - Add a small integration test comparing gap-discovery outputs with and without the forwarded `original_analysis` to validate behavior and token usage.

Design notes / constraints:
- `original_analysis` should be a configurable subset (e.g., `static_analysis`, `security_analysis`, `ai_analysis`) to avoid excessive payload size.
- All forwarded raw analysis content must be truncated or summarized before being sent to the LLM to avoid token-overflow and noisy input.
- Default behavior remains backward-compatible: if `original_analysis` is not provided, second-pass falls back to existing evidence-only approach.

Files touched (planned):
- core/agents/analysis_result_summary_agent.py
- core/agents/ai_driven_second_pass_analysis_agent.py
- infrastructure/config/prompts.py (or prompt templates)
- tests/ (new unit/integration tests)

Commit intent: record the plan before making code changes, so the change rationale and intended implementation are preserved in version control.
