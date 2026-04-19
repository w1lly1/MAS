# BigVul -> MAS DB Tasks

This folder contains a converter that turns BigVul metadata into MAS database task inputs (`target/action/data`) that can be consumed by `db_manage_agent`.

## Generated Files

- `output/bigvul_top20_db_tasks.json`
  - Full task payload (session + issue_pattern + curated_issue tasks)
- `output/bigvul_top20_summary.json`
  - Aggregate stats for generated tasks
- `output/bigvul_top20_validation.json`
  - Schema validation results

## Run

```powershell
.\venv\Scripts\Activate.ps1
python utils/bigvul_ingest/build_db_tasks_from_metadata.py --start 0 --count 20
```

Optional parameters:

```powershell
python utils/bigvul_ingest/build_db_tasks_from_metadata.py \
  --metadata-root tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/metadata \
  --before-root tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/before \
  --after-root tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/after \
  --output-dir utils/bigvul_ingest/output \
  --start 0 \
  --count 20 \
  --max-snippet-chars 2000 \
  --session-id bigvul-top20-batch-001
```

## Deployment Plan

1. Build task JSON from metadata.
2. Inspect `bigvul_top20_validation.json`; proceed only if `valid=true`.
3. Feed `tasks` array to MAS `db_manage_agent` (dry-run first).
4. Execute write mode import to SQLite.
5. Trigger `sync` for `issue_pattern` to Weaviate layered vectors.
6. Verify counts against `bigvul_top20_summary.json`.
7. Rollback by `session_id` if needed.

## Rollback Strategy

- Use generated `session_id` as batch key.
- Delete this batch's `curated_issue` records first.
- Delete related `issue_pattern` records and sync delete to Weaviate.
- Delete the `review_session` record last.
