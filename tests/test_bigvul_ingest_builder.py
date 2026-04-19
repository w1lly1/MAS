import json
from pathlib import Path

from utils.bigvul_ingest.build_db_tasks_from_metadata import BuildConfig, build_payload, _validate_tasks


def test_bigvul_builder_top2_contract():
    cfg = BuildConfig(
        metadata_root=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/metadata"),
        before_root=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/before"),
        after_root=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/after"),
        output_dir=Path("utils/bigvul_ingest/output"),
        start=0,
        count=2,
        max_snippet_chars=500,
        session_id="bigvul-test-session",
    )

    payload = build_payload(cfg)
    assert "meta" in payload
    assert "tasks" in payload
    assert isinstance(payload["tasks"], list)
    assert len(payload["tasks"]) >= 1

    targets = [t.get("target") for t in payload["tasks"]]
    assert "review_session" in targets
    assert "issue_pattern" in targets
    assert "curated_issue" in targets

    validation = _validate_tasks(payload["tasks"])
    assert validation["valid"] is True
    assert validation["errors"] == []
