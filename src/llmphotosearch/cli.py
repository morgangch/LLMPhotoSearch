"""Command-line interface."""

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TimeRemainingColumn,
)

from llmphotosearch import __version__
from llmphotosearch.backends.vlm import MODELS
from llmphotosearch.export.report import to_json, to_markdown
from llmphotosearch.ranking.search import Match

app = typer.Typer(
    name="photosearch",
    help="Search photo folders with a natural-language description, using a local VLM.",
    no_args_is_help=True,
)
console = Console(stderr=True)

MAX_PRINTED_MATCHES = 3
DEFAULT_REPORT = Path("photosearch-results.md")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"photosearch {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    pass


class _ProgressReporter:
    def __init__(self, progress: Progress):
        self._progress = progress
        self._task: TaskID | None = None

    def stage(self, name: str, total: int | None = None) -> None:
        if self._task is not None:
            self._progress.remove_task(self._task)
        self._task = self._progress.add_task(name, total=total)

    def advance(self, n: int = 1) -> None:
        if self._task is not None:
            self._progress.advance(self._task, n)


@app.command()
def search(
    description: Annotated[str, typer.Argument(help="What you are looking for (any language).")],
    folders: Annotated[
        list[Path],
        typer.Argument(
            exists=True, file_okay=False, help="One or more folders, scanned recursively."
        ),
    ],
    threshold: Annotated[
        float, typer.Option(min=0, max=100, help="Minimum match percentage.")
    ] = 50.0,
    top_k: Annotated[
        int, typer.Option(min=1, help="Number of pre-selected candidates analyzed by the VLM.")
    ] = 30,
    model: Annotated[
        str | None,
        typer.Option(
            help=f"VLM to use: {', '.join(MODELS)} [default: chosen from hardware].",
            show_default=False,
        ),
    ] = None,
    backend: Annotated[
        Literal["cuda", "vulkan", "metal", "cpu"] | None,
        typer.Option(help="Inference backend [default: chosen from hardware].", show_default=False),
    ] = None,
    fast: Annotated[
        bool,
        typer.Option(
            "--fast", help="Skip the VLM: rank by embedding similarity only, without match %."
        ),
    ] = False,
    out: Annotated[
        Path | None,
        typer.Option(
            help=f"Report path (.md or .json), written when more than {MAX_PRINTED_MATCHES} "
            f"images match [default: {DEFAULT_REPORT}].",
            show_default=False,
        ),
    ] = None,
) -> None:
    """Find the images in FOLDERS that best match DESCRIPTION."""
    # Imported here so that `--help` stays fast (ONNX Runtime is slow to load).
    from llmphotosearch.backends.llama_server import EngineError
    from llmphotosearch.pipeline import SearchOptions, run_search

    if model is not None and model not in MODELS:
        raise typer.BadParameter(f"choose from {', '.join(MODELS)}", param_hint="--model")

    options = SearchOptions(top_k, threshold / 100, model, backend, fast)
    try:
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            result = run_search(description, folders, options, _ProgressReporter(progress))
    except EngineError as e:
        console.print(f"[red]Inference engine error:[/] {e}")
        raise typer.Exit(code=2) from e

    for path, error in result.failed:
        console.print(f"[yellow]Skipped unreadable image[/] {path} ({error})")
    if not result.indexed:
        console.print("No images found.")
        raise typer.Exit(code=1)

    if not result.reranked:
        for m in result.candidates[:MAX_PRINTED_MATCHES]:
            typer.echo(m.path)
        if out:
            _write_report(out, description, result.candidates, reranked=False)
        return

    matches = result.matches
    if not matches:
        console.print(f"No image matches at {threshold:g} % or more among {result.indexed}.")
        raise typer.Exit(code=1)
    if len(matches) <= MAX_PRINTED_MATCHES and out is None:
        for m in matches:
            typer.echo(f"{m.score * 100:5.1f} %  {m.path}")
        return
    console.print(f"{len(matches)} matching images among {result.indexed}.")
    _write_report(out or DEFAULT_REPORT, description, matches, reranked=True)


def _write_report(out: Path, description: str, matches: list[Match], reranked: bool) -> None:
    render = to_json if out.suffix.lower() == ".json" else to_markdown
    out.write_text(render(description, matches, reranked), encoding="utf-8")
    console.print("Report written to:")
    typer.echo(out.resolve())


@app.command()
def ui() -> None:
    """Open the local web UI in the browser."""
    typer.echo("The UI is not implemented yet.", err=True)
    raise typer.Exit(code=1)
