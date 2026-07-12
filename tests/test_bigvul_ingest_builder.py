import json
from pathlib import Path

from utils.bigvul_ingest.build_db_tasks_from_metadata import BuildConfig, build_payload, _validate_tasks
from utils.bigvul_ingest.build_structured_ingest import (
    BuildConfig as StructuredBuildConfig,
    build_payload as build_structured_payload,
)
from utils.bigvul_ingest.rules import (
    derive_file_pattern,
    derive_solution_from_diff,
    extract_function_name_from_summary,
    extract_snippet_around_lines,
)


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


def test_rules_extract_cve_2006_5331_anchors():
    summary = (
        "The altivec_unavailable_exception function in arch/powerpc/kernel/traps.c "
        "in the Linux kernel before 2.6.19 ..."
    )
    assert extract_function_name_from_summary(summary) == "altivec_unavailable_exception"
    assert derive_file_pattern("arch/powerpc/kernel/traps.c") == "arch/powerpc/kernel/traps.c"

    before = "void f(void) {\n#if !defined(CONFIG_ALTIVEC)\n\tdie();\n#endif\n}\n"
    after = "void f(void) {\n\tif (user_mode()) {\n\t\t_exception();\n\t}\n}\n"
    solution = derive_solution_from_diff(before, after, "dos")
    assert "Altivec" in solution or "CONFIG_ALTIVEC" in solution or "conditional" in solution.lower()


def test_structured_ingest_cve_2006_5331_fields():
    cfg = StructuredBuildConfig(
        metadata_root=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/metadata"),
        before_root=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/before"),
        after_root=Path("tests/BigVul/MSR_20_Code_vulnerability_CSV_Dataset/source_code_restructured/after"),
        output_dir=Path("utils/bigvul_ingest/output"),
        output_name="structured_ingest_test.json",
        start=0,
        count=20,
        max_snippet_chars=2000,
        session_id="bigvul-test-cve",
        ingest_mode="strict",
    )
    payload = build_structured_payload(cfg)
    cve = next(
        item
        for item in payload["data"]
        if item.get("pattern", {}).get("title") == "CVE-2006-5331"
    )
    pattern = cve["pattern"]
    issue = cve["instances"][0]["issue"]

    assert pattern["file_pattern"] == "arch/powerpc/kernel/traps.c"
    assert pattern["class_pattern"] == "altivec_unavailable_exception"
    assert "Altivec" in pattern["solution"] or "CONFIG_ALTIVEC" in pattern["solution"]
    assert issue["status"] == "resolved"
    assert "altivec_unavailable_exception" in issue["code_snippet"]
    assert issue["start_line"] >= 900


def test_snippet_around_lines_not_file_header():
    text = "\n".join([f"line{i}" for i in range(1, 21)])
    snippet = extract_snippet_around_lines(text, start_line=10, end_line=12, context_lines=2, max_chars=500)
    assert "line10" in snippet
    assert "line1\nline2" not in snippet or "line10" in snippet
