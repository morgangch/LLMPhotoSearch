"""Command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from llmphotosearch import __version__

app = typer.Typer(
    name="photosearch",
    help="Search photo folders with a natural-language description, using a local VLM.",
    no_args_is_help=True,
)


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
    threshold: Annotated[
        float, typer.Option(min=0, max=100, help="Minimum match percentage to keep an image.")
    ] = 50.0,
    top_k: Annotated[
        int, typer.Option(min=1, help="Number of pre-filtered candidates re-ranked by the VLM.")
    ] = 30,
    model: Annotated[
        str | None, typer.Option(help="VLM to use (default: chosen from detected hardware).")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Report path when more than 3 images match.")
    ] = None,
) -> None:
    """Find the images in FOLDERS that best match DESCRIPTION."""
    typer.echo("Search is not implemented yet.", err=True)
    raise typer.Exit(code=1)


@app.command()
def ui() -> None:
    """Open the local web UI in the browser."""
    typer.echo("The UI is not implemented yet.", err=True)
    raise typer.Exit(code=1)
