"""Tests for the CLI, using a local repository (no network)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from py_extractor import cli


@pytest.fixture
def local_repo(tmp_path: Path) -> Path:
  """A tiny on-disk package to extract from."""
  repo = tmp_path / "repo"
  (repo / "mypkg").mkdir(parents=True)
  (repo / "mypkg" / "__init__.py").write_text("from .core import Model\n", encoding="utf-8")
  (repo / "mypkg" / "core.py").write_text(
      "import numpy as np\n"
      "from torch import nn\n"
      "\n"
      "class Model(nn.Module):\n"
      "    def forward(self, x):\n"
      "        return np.zeros_like(x)\n",
      encoding="utf-8",
  )
  return repo


# The CLI tests must never hit the network. `--no-third-party` disables the
# PyPI->GitHub resolver, and `--cache-dir` keeps any incidental cache local.
def _args(repo: Path, symbol: str, *extra: str) -> list[str]:
  return [str(repo), symbol, "--no-third-party", *extra]


def test_cli_writes_file(local_repo: Path, tmp_path: Path) -> None:
  out = tmp_path / "out.py"
  code = cli.main(
      _args(local_repo, "Model", "--out", str(out), "--report", "none", "--cache-dir",
            str(tmp_path / "c")))
  assert code == 0
  assert out.exists()
  ast.parse(out.read_text())


def test_cli_stdout(capsys, local_repo: Path, tmp_path: Path) -> None:
  code = cli.main(_args(local_repo, "Model", "--report", "none", "--cache-dir",
                        str(tmp_path / "c")))
  assert code == 0
  captured = capsys.readouterr()
  assert "class Model" in captured.out


def test_cli_report_text(capsys, local_repo: Path, tmp_path: Path) -> None:
  cli.main(
      _args(local_repo, "Model", "--report", "text", "--out", "/dev/null", "--cache-dir",
            str(tmp_path / "c")))
  captured = capsys.readouterr()
  assert "Target: mypkg.core:Model" in captured.err


def test_cli_report_json(capsys, local_repo: Path, tmp_path: Path) -> None:
  cli.main(
      _args(local_repo, "Model", "--report", "json", "--out", "/dev/null", "--cache-dir",
            str(tmp_path / "c")))
  captured = capsys.readouterr()
  payload = json.loads(captured.err)
  assert payload["target"]["symbol"] == "Model"


def test_cli_missing_symbol_returns_error(local_repo: Path, tmp_path: Path) -> None:
  code = cli.main(
      _args(local_repo, "DoesNotExist", "--report", "none", "--cache-dir", str(tmp_path / "c")))
  assert code == 1


def test_cli_bad_repo_returns_error() -> None:
  code = cli.main(["!!!not-a-repo!!!", "Thing", "--report", "none"])
  assert code == 1


def test_cli_file_hint(local_repo: Path, tmp_path: Path) -> None:
  out = tmp_path / "out.py"
  code = cli.main(
      _args(local_repo, "Model", "--file", "mypkg/core.py", "--out", str(out), "--report", "none",
            "--cache-dir", str(tmp_path / "c")))
  assert code == 0


def test_cli_strict_trust_still_runs(local_repo: Path, tmp_path: Path) -> None:
  out = tmp_path / "out.py"
  code = cli.main(
      _args(local_repo, "Model", "--strict-trust", "--out", str(out), "--report", "none",
            "--cache-dir", str(tmp_path / "c")))
  assert code == 0
  # numpy is untrusted under strict policy -> must appear as a kept import anyway.
  text = out.read_text()
  assert "import numpy" in text


def test_cli_trust_extra(local_repo: Path, tmp_path: Path) -> None:
  out = tmp_path / "out.py"
  code = cli.main(
      _args(local_repo, "Model", "--trust", "some_pkg", "--out", str(out), "--report", "none",
            "--cache-dir", str(tmp_path / "c")))
  assert code == 0


def test_cli_version(capsys) -> None:
  with pytest.raises(SystemExit) as exc:
    cli.main(["--version"])
  assert exc.value.code == 0
  assert "py_extractor" in capsys.readouterr().out
