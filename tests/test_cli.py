from typer.testing import CliRunner

from llmphotosearch import __version__
from llmphotosearch.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_search_requires_existing_folder(tmp_path):
    result = runner.invoke(app, ["search", "a cat", str(tmp_path / "missing")])
    assert result.exit_code != 0
