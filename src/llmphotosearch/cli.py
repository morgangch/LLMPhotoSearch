"""Command-line interface."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn

from llmphotosearch import __version__

app = typer.Typer(
    name="photosearch",
    help="Search photo folders with a natural-language description, using a local VLM.",
    no_args_is_help=True,
)
console = Console(stderr=True)

MAX_PRINTED_MATCHES = 3


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


@app.command()
def search(
    description: Annotated[str, typer.Argument(help="What you are looking for (any language).")],
    folders: Annotated[
        list[Path],
        typer.Argument(
            exists=True, file_okay=False, help="One or more folders, scanned recursively."
        ),
    ],
    top_k: Annotated[
        int, typer.Option(min=1, help="Number of best candidates kept by the pre-filter.")
    ] = 30,
    out: Annotated[
        Path | None,
        typer.Option(help="Also write the candidates to this report (.md or .json)."),
    ] = None,
) -> None:
    """Find the images in FOLDERS that best match DESCRIPTION."""
    from llmphotosearch.backends.siglip import SigLIP
    from llmphotosearch.core.images import iter_images
    from llmphotosearch.export.report import to_json, to_markdown
    from llmphotosearch.index.indexer import build_index
    from llmphotosearch.index.store import EmbeddingStore
    from llmphotosearch.ranking.search import prefilter

    images = list(iter_images(folders))
    if not images:
        console.print("No images found.")
        raise typer.Exit(code=1)

    embedder = SigLIP()
    with (
        EmbeddingStore() as store,
        Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress,
    ):
        task = progress.add_task("Indexing", total=len(images))
        index = build_index(
            images, embedder, store, on_progress=lambda n: progress.advance(task, n)
        )

    for path, error in index.failed:
        console.print(f"[yellow]Skipped unreadable image[/] {path} ({error})")
    if not index.paths:
        raise typer.Exit(code=1)

    # Embedding scores rank well but are not calibrated percentages; the
    # thresholded match percentage comes with VLM re-ranking.
    candidates = prefilter(description, index, embedder)[:top_k]
    for m in candidates[:MAX_PRINTED_MATCHES]:
        typer.echo(m.path)

    if out:
        render = to_json if out.suffix.lower() == ".json" else to_markdown
        out.write_text(render(description, candidates), encoding="utf-8")
        console.print(f"{len(candidates)} candidates written to {out.resolve()}")


@app.command()
def ui() -> None:
    """Open the local web UI in the browser."""
    typer.echo("The UI is not implemented yet.", err=True)
    raise typer.Exit(code=1)
