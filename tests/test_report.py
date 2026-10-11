import json
from pathlib import Path

from llmphotosearch.export.report import to_json, to_markdown
from llmphotosearch.ranking.search import Match

MATCHES = [Match(Path("/photos/chat|roux.jpg"), 0.877), Match(Path("/photos/b.jpg"), 0.5)]


def test_markdown_lists_matches_in_order():
    md = to_markdown("un chat roux", MATCHES)
    assert "un chat roux" in md
    rows = [line for line in md.splitlines() if line.startswith("| 1") or line.startswith("| 2")]
    assert rows[0].startswith("| 1 | 87.7 % |")
    assert "chat\\|roux.jpg" in rows[0]


def test_json_is_valid_and_ranked():
    data = json.loads(to_json("un chat roux", MATCHES))
    assert [m["rank"] for m in data["matches"]] == [1, 2]
    assert data["matches"][0]["score"] == 0.877
