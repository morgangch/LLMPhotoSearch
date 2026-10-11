"""Markdown and JSON result reports."""

import json
from collections.abc import Sequence
from datetime import datetime

from llmphotosearch.ranking.search import Match


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|")


def to_markdown(description: str, matches: Sequence[Match], reranked: bool = True) -> str:
    """`reranked` False means scores are uncalibrated embedding similarities."""
    lines = [
        "# Photo search results",
        "",
        f"- **Description:** {description}",
        f"- **Date:** {datetime.now().isoformat(timespec='seconds')}",
        f"- **Images:** {len(matches)}",
        "",
        f"| # | {'Match' if reranked else 'Similarity'} | Path |",
        "|---:|---:|---|",
    ]
    for rank, m in enumerate(matches, 1):
        lines.append(f"| {rank} | {m.score * 100:.1f} % | `{_escape_cell(str(m.path))}` |")
    return "\n".join(lines) + "\n"


def to_json(description: str, matches: Sequence[Match], reranked: bool = True) -> str:
    def entry(rank: int, m: Match) -> dict:
        e = {"rank": rank, "score": round(m.score, 4), "path": str(m.path)}
        if m.prefilter_score is not None:
            e["prefilter_score"] = round(m.prefilter_score, 4)
        return e

    data = {
        "description": description,
        "date": datetime.now().isoformat(timespec="seconds"),
        "score": "match_probability" if reranked else "embedding_similarity",
        "matches": [entry(rank, m) for rank, m in enumerate(matches, 1)],
    }
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"
