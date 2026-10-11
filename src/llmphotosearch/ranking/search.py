"""Ranking of indexed images against a description."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from llmphotosearch.index.indexer import IndexResult


class TextEmbedder(Protocol):
    def embed_text(self, text: str) -> np.ndarray: ...
    def probability(self, similarity: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Match:
    path: Path
    score: float
    """Match probability in [0, 1]."""


def prefilter(description: str, index: IndexResult, embedder: TextEmbedder) -> list[Match]:
    """All indexed images, best first, scored by embedding similarity."""
    if not index.paths:
        return []
    similarity = index.vectors @ embedder.embed_text(description)
    probs = embedder.probability(similarity)
    order = np.argsort(-similarity)
    return [Match(index.paths[i], float(probs[i])) for i in order]
