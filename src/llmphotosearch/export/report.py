"""Markdown and JSON result reports."""

import json
from collections.abc import Sequence
from datetime import datetime

from llmphotosearch.ranking.search import Match


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def to_markdown(description: str, matches: Sequence[Match]) -> str:
    lines = [
        "# Photo search results",
        "",
        f"- **Description:** {description}",
        f"- **Date:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Matches:** {len(matches)}",
        "",
        "| # | Score | Path |",
        "|---:|---:|---|",
    ]
    for rank, m in enumerate(matches, 1):
        lines.append(f"| {rank} | {m.score * 100:.1f} % | `{_escape_cell(str(m.path))}` |")
    return "\n".join(lines) + "\n"


def to_json(description: str, matches: Sequence[Match]) -> str:
    data = {
        "description": description,
        "date": datetime.now().isoformat(timespec="seconds"),
        "matches": [
            {"rank": rank, "score": round(m.score, 4), "path": str(m.path)}
            for rank, m in enumerate(matches, 1)
        ],
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
