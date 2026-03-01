import json
from core.utils.pretty_db import tabulate_grouped_items, export_grouped_items_json, export_grouped_items_csv


def sample_grouped():
    return {
        "curated_issue": [
            {"id": 1, "file_path": "src/a.py", "problem": "npe"},
            {"id": 2, "file_path": "src/b.py", "problem": "leak"},
        ],
        "issue_pattern": [
            {"pattern_id": 10, "title": "NullPointer"},
        ],
    }


def test_tabulate_grouped_items_returns_tables():
    grouped = sample_grouped()
    tables = tabulate_grouped_items(grouped, max_col_width=40)
    assert isinstance(tables, dict)
    assert "curated_issue" in tables
    assert "issue_pattern" in tables
    # basic content check
    assert "file_path" in tables["curated_issue"]


def test_export_grouped_items_create_files(tmp_path):
    grouped = sample_grouped()
    json_path = tmp_path / "out.json"
    written = export_grouped_items_json(grouped, str(json_path))
    assert written == str(json_path)
    with open(written, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "curated_issue" in data

    csv_dir = tmp_path / "csvs"
    paths = export_grouped_items_csv(grouped, str(csv_dir))
    assert any(p.endswith("curated_issue.csv") for p in paths)
    assert any(p.endswith("issue_pattern.csv") for p in paths)