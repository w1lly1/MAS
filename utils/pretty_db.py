from typing import Any, Dict, List
from textwrap import shorten
import json
import csv
from pathlib import Path

from .pretty_table import format_table

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover - fallback when tabulate not installed
    tabulate = None


def _to_dict(item: Any) -> Dict[str, Any]:
    if item is None:
        return {}
    if isinstance(item, dict):
        return item
    try:
        return dict(item)
    except Exception:
        pass
    try:
        return {k: getattr(item, k) for k in getattr(item, "keys", lambda: [])()}
    except Exception:
        pass
    # Last resort: try __dict__ or string
    try:
        return getattr(item, "__dict__", {"value": str(item)})
    except Exception:
        return {"value": str(item)}


def _format_value(v: Any, max_col_width: int = 80) -> str:
    if v is None:
        return ""
    s = str(v)
    s = s.replace("\n", " ")
    if len(s) > max_col_width:
        return shorten(s, width=max_col_width, placeholder="…")
    return s


def tabulate_grouped_items(grouped_items: Dict[str, List[Any]], max_col_width: int = 120, tablefmt: str = "github") -> Dict[str, str]:
    """
    Convert grouped items (mapping target -> list-of-records) into pretty tables using tabulate.

    Returns a dict target->table_text. If `tabulate` is not installed, falls back to a simple CSV-like dump.
    """
    result: Dict[str, str] = {}
    for target, items in grouped_items.items():
        if not items:
            continue
        rows: List[Dict[str, Any]] = [_to_dict(i) for i in items]
        headers = sorted({k for r in rows for k in r.keys()})
        # try to reorder headers according to DB model (primary key, foreign keys, then others)
        try:
            headers = _preferred_column_order(target, headers)
        except Exception:
            pass
        table_rows = [[_format_value(r.get(h, ""), max_col_width) for h in headers] for r in rows]
        try:
            # use shared format_table (rich/tabulate fallback) for consistent rendering
            table_text = format_table(table_rows, headers=headers, max_col_width=max_col_width, tablefmt=tablefmt)
        except Exception:
            # fallback to tabulate or simple CSV-like
            if tabulate:
                try:
                    table_text = tabulate(table_rows, headers=headers, tablefmt=tablefmt, showindex=False)
                except Exception:
                    table_lines = [", ".join(headers)]
                    table_lines += [", ".join(row) for row in table_rows]
                    table_text = "\n".join(table_lines)
            else:
                table_lines = [", ".join(headers)]
                table_lines += [", ".join(row) for row in table_rows]
                table_text = "\n".join(table_lines)
        result[target] = table_text
    return result


def _preferred_column_order(target: str, headers: List[str]) -> List[str]:
    """Return headers reordered following the SQLAlchemy model definition.

    Priority: primary key columns (in model order) -> foreign key columns -> other columns (model order) -> any remaining headers.
    If models cannot be imported or mapping not found, returns the original headers.
    """
    # Normalize target to singular form used in our mapping
    tgt = (target or "").lower()
    mapping = {
        "issue_pattern": "IssuePattern",
        "issue_patterns": "IssuePattern",
        "issuepattern": "IssuePattern",
        "pattern": "IssuePattern",
        "curated_issue": "CuratedIssue",
        "curated_issues": "CuratedIssue",
        "curated": "CuratedIssue",
        "review_session": "ReviewSession",
        "review_sessions": "ReviewSession",
        "session": "ReviewSession",
    }
    class_name = mapping.get(tgt)
    if not class_name:
        return headers

    try:
        from infrastructure.database.sqlite import models as models_mod

        cls = getattr(models_mod, class_name, None)
        if cls is None:
            return headers

        table = getattr(cls, "__table__", None)
        if table is None:
            return headers

        # model column order
        model_cols = [c.name for c in table.columns]
        # primary key columns
        pk_cols = [c.name for c in table.primary_key.columns]
        # foreign key columns
        fk_cols = [c.name for c in table.columns if len(c.foreign_keys) > 0]

        ordered = []
        # add primary keys in model order
        for c in model_cols:
            if c in pk_cols and c in headers and c not in ordered:
                ordered.append(c)
        # add foreign keys
        for c in model_cols:
            if c in fk_cols and c in headers and c not in ordered:
                ordered.append(c)
        # add remaining model-defined columns
        for c in model_cols:
            if c in headers and c not in ordered:
                ordered.append(c)
        # finally append any headers unknown to model
        for h in headers:
            if h not in ordered:
                ordered.append(h)
        return ordered
    except Exception:
        return headers


def export_grouped_items_json(grouped_items: Dict[str, List[Any]], dest: str) -> str:
    """Export grouped items to a JSON file. Returns the written file path."""
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    serializable = {k: [_to_dict(i) for i in v] for k, v in grouped_items.items()}
    with p.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    return str(p)


def export_grouped_items_csv(grouped_items: Dict[str, List[Any]], dest_dir: str) -> List[str]:
    """Export each group's items to a separate CSV file under dest_dir. Returns list of file paths."""
    out_paths: List[str] = []
    d = Path(dest_dir)
    d.mkdir(parents=True, exist_ok=True)
    for target, items in grouped_items.items():
        rows = [_to_dict(i) for i in items]
        if not rows:
            continue
        headers = sorted({k for r in rows for k in r.keys()})
        filename = d / f"{target}.csv"
        with filename.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for r in rows:
                writer.writerow([r.get(h, "") for h in headers])
        out_paths.append(str(filename))
    return out_paths
