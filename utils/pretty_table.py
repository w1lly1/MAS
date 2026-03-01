"""
Small table formatting helper wrapping `tabulate` with safe fallbacks and cell wrapping.

Provides `format_table(rows, headers=..., max_col_width=..., tablefmt=...)` which
returns a single string (possibly multi-line) suitable for logging or printing.
"""
from typing import List, Dict, Any, Optional
import shutil
import textwrap

try:
    from rich.table import Table as RichTable
    from rich.console import Console as RichConsole
    _rich_available = True
except Exception:
    RichTable = None
    RichConsole = None
    _rich_available = False

try:
    from tabulate import tabulate
except Exception:
    tabulate = None


def _wrap_cell(s: Any, width: int) -> str:
    if s is None:
        return ""
    s = str(s)
    # Replace newlines with spaces to keep cell height predictable, but allow wrapping
    s = s.replace("\n", " ")
    if len(s) <= width:
        return s
    # wrap preserving words
    return "\n".join(textwrap.wrap(s, width=width))


def format_table(rows: List[Dict[str, Any]] | List[List[Any]], headers: Optional[List[str]] = None,
                 max_col_width: Optional[int] = None, tablefmt: str = "github", showindex: bool = False) -> str:
    """
    Render rows into a table string.

    - rows: a list of dicts (preferred) or list-of-lists. If dicts are provided and `headers` omitted,
      headers are inferred from the first row's keys (sorted for stability).
    - max_col_width: if None, compute from terminal width, otherwise wrap cells to this width.
    - tablefmt: passed to `tabulate` when available.
    """
    data = []
    if not rows:
        return ""

    # normalize dict rows to list of lists
    if isinstance(rows[0], dict):
        if headers is None:
            headers = sorted({k for r in rows for k in r.keys()})
        data = [[r.get(h, "") for h in headers] for r in rows]
    else:
        data = [list(r) for r in rows]
        if headers is None:
            headers = []

    # compute default max_col_width from terminal size if not provided
    if max_col_width is None:
        try:
            term_w = shutil.get_terminal_size((120, 25)).columns
            num_cols = max(1, len(headers) or max((len(r) for r in data), default=1))
            # leave some margin for table chars
            max_col_width = max(20, (term_w - (num_cols * 3)) // num_cols)
        except Exception:
            max_col_width = 40

    wrapped_rows: List[List[str]] = []
    for row in data:
        wrapped_row = [_wrap_cell(cell, max_col_width) for cell in row]
        wrapped_rows.append(wrapped_row)

    # Prefer rich rendering when available (better handling of wide/unicode/boxes)
    if _rich_available:
        try:
            # compute console width
            try:
                term_w = shutil.get_terminal_size((120, 25)).columns
            except Exception:
                term_w = 120
            console = RichConsole(record=True, width=term_w)
            table = RichTable(box=None)
            # add columns
            for h in (headers or []):
                table.add_column(str(h), overflow="fold", no_wrap=False)
            for row in wrapped_rows:
                table.add_row(*[str(c) for c in row])
            # capture output instead of printing directly to avoid side-effects
            with console.capture() as capture:
                console.print(table)
            text = capture.get()
            return text
        except Exception:
            # fall through to other renderers
            pass

    # If tabulate available, use it; otherwise build a simple grid
    if tabulate:
        try:
            return tabulate(wrapped_rows, headers=headers or (), tablefmt=tablefmt, showindex=showindex, stralign="left")
        except Exception:
            pass

    # fallback simple renderer (CSV-like)
    lines = []
    if headers:
        lines.append(" | ".join(headers))
        lines.append("-" * max(20, len(lines[0])))
    for row in wrapped_rows:
        lines.append(" | ".join(row))
    return "\n".join(lines)
