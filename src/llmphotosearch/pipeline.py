"""End-to-end search, shared by the CLI and the UI."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from huggingface_hub.utils import disable_progress_bars, logging
from platformdirs import user_cache_path

from llmphotosearch.backends import hardware, vlm
from llmphotosearch.backends.llama_server import (
    Backend,
    LlamaServer,
    default_backend,
    ensure_server_binary,
)
from llmphotosearch.backends.siglip import SigLIP
from llmphotosearch.core.images import iter_images
from llmphotosearch.index.indexer import build_index
from llmphotosearch.index.store import EmbeddingStore
from llmphotosearch.ranking.rerank import match_probability, to_english
from llmphotosearch.ranking.search import Match, prefilter

# Progress is reported through `Reporter`; keep the HF Hub quiet.
disable_progress_bars()
logging.set_verbosity_error()


@dataclass
class SearchOptions:
    top_k: int = 30
    """Candidates kept by the pre-filter and re-ranked by the VLM."""
    threshold: float = 0.5
    """Minimum match probability for a re-ranked image to count as a match."""
    model: str | None = None
    """Key of `vlm.MODELS`; chosen from the hardware when None."""
    backend: Backend | None = None
    """llama.cpp build; chosen from the hardware when None."""
    fast: bool = False
    """Skip VLM re-ranking: rank by embedding similarity only (no calibrated score)."""


@dataclass
class SearchResult:
    candidates: list[Match]
    """Best first. Re-ranked by the VLM unless `reranked` is False."""
    reranked: bool
    threshold: float
    indexed: int
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def matches(self) -> list[Match]:
        """Candidates at or above the threshold; empty when not re-ranked."""
        if not self.reranked:
            return []
        return [m for m in self.candidates if m.score >= self.threshold]


class Reporter(Protocol):
    def stage(self, name: str, total: int | None = None) -> None:
        """A new step starts; `total` units of work, or unknown."""

    def advance(self, n: int = 1) -> None: ...


def run_search(
    description: str,
    folders: Sequence[Path],
    options: SearchOptions,
    reporter: Reporter,
) -> SearchResult:
    images = list(iter_images(folders))
    embedder = SigLIP()

    server = None
    if not options.fast:
        hw = hardware.detect()
        backend = options.backend or default_backend(hw)
        spec = vlm.MODELS[options.model or vlm.default_model(hw)]
        binary = ensure_server_binary(hw, backend, _DownloadProgress(reporter))
        reporter.stage("Downloading model (first run only)")
        model_path, mmproj_path = spec.download()
        # Starts loading in the background while images are indexed.
        server = LlamaServer(
            binary,
            model_path,
            mmproj_path,
            gpu=backend != "cpu",
            log_file=user_cache_path("llmphotosearch", appauthor=False) / "llama-server.log",
        )

    try:
        reporter.stage("Downloading image encoder (first run only)")
        embedder.download()
        reporter.stage("Indexing images", total=len(images))
        with EmbeddingStore() as store:
            index = build_index(images, embedder, store, on_progress=reporter.advance)

        if server is None:
            candidates = prefilter(description, index, embedder)[: options.top_k]
            return SearchResult(
                candidates, False, options.threshold, len(index.paths), index.failed
            )

        reporter.stage("Loading model")
        server.wait_ready()
        # The VLM reads any language, but the embedding pre-filter is best in English.
        query = to_english(server, description)
        candidates = prefilter(query, index, embedder)[: options.top_k]

        reporter.stage("Analyzing candidates", total=len(candidates))
        max_side = 768 if backend in ("cuda", "metal") else 512
        reranked = []
        for c in candidates:
            score = match_probability(server, c.path, description, max_side)
            reranked.append(Match(c.path, score, prefilter_score=c.score))
            reporter.advance()
        reranked.sort(key=lambda m: -m.score)
        return SearchResult(reranked, True, options.threshold, len(index.paths), index.failed)
    finally:
        if server is not None:
            server.stop()


class _DownloadProgress:
    """Adapts byte-level download callbacks to a Reporter, one stage per file."""

    def __init__(self, reporter: Reporter):
        self._reporter = reporter
        self._total = None
        self._done = 0

    def __call__(self, done: int, total: int) -> None:
        if total != self._total or done < self._done:
            self._reporter.stage("Downloading inference engine (first run only)", total or None)
            self._total, self._done = total, 0
        self._reporter.advance(done - self._done)
        self._done = done
