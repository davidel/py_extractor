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


@pytest.fixture
def ambiguous_repo(tmp_path: Path) -> Path:
  """A repo where the same symbol is defined in two files."""
  repo = tmp_path / "ambig"
  (repo / "pkg").mkdir(parents=True)
  (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
  (repo / "pkg" / "one.py").write_text("class Thing:\n    tag = 'one'\n", encoding="utf-8")
  (repo / "pkg" / "two.py").write_text("class Thing:\n    tag = 'two'\n", encoding="utf-8")
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


# --- symbol disambiguation -------------------------------------------------


def test_cli_non_interactive_ambiguous_fails(ambiguous_repo: Path, tmp_path: Path) -> None:
  """Without a TTY the CLI must not hang; it reports candidates and exits 1."""
  code = cli.main(
      _args(ambiguous_repo, "Thing", "--report", "none", "--cache-dir", str(tmp_path / "c")))
  assert code == 1


def test_cli_yes_flag_ambiguous_fails(ambiguous_repo: Path, tmp_path: Path, capsys) -> None:
  code = cli.main(
      _args(ambiguous_repo, "Thing", "-y", "--report", "none", "--cache-dir", str(tmp_path / "c")))
  assert code == 1
  err = capsys.readouterr().err
  assert "multiple files" in err
  assert "--file" in err


def test_cli_file_resolves_ambiguity(ambiguous_repo: Path, tmp_path: Path) -> None:
  out = tmp_path / "out.py"
  code = cli.main(
      _args(ambiguous_repo, "Thing", "--file", "pkg/two.py", "--out", str(out), "--report", "none",
            "--cache-dir", str(tmp_path / "c")))
  assert code == 0
  assert "tag = 'two'" in out.read_text()


def test_cli_prompt_selects_second(ambiguous_repo, tmp_path, monkeypatch, capsys) -> None:
  """Interactive prompt: feeding '2' picks the second candidate."""
  import sys

  monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
  monkeypatch.setattr("builtins.input", lambda prompt="": "2")
  out = tmp_path / "out.py"
  code = cli.main(
      _args(ambiguous_repo, "Thing", "--out", str(out), "--report", "none", "--cache-dir",
            str(tmp_path / "c")))
  assert code == 0
  text = out.read_text()
  # Candidates are ranked deterministically: pkg/one.py (1) then pkg/two.py (2).
  assert "tag = 'two'" in text


def test_cli_prompt_blank_uses_default(ambiguous_repo, tmp_path, monkeypatch) -> None:
  import sys

  monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
  monkeypatch.setattr("builtins.input", lambda prompt="": "")
  out = tmp_path / "out.py"
  code = cli.main(
      _args(ambiguous_repo, "Thing", "--out", str(out), "--report", "none", "--cache-dir",
            str(tmp_path / "c")))
  assert code == 0


# --- verification flags ----------------------------------------------------


@pytest.fixture
def plain_repo(tmp_path: Path) -> Path:
  """A repo with a dependency-free symbol (importable without third parties)."""
  repo = tmp_path / "plain"
  (repo / "pkg").mkdir(parents=True)
  (repo / "pkg" / "__init__.py").write_text("", encoding="utf-8")
  (repo / "pkg" / "mod.py").write_text("class Simple:\n    def value(self):\n        return 42\n",
                                       encoding="utf-8")
  return repo


def test_cli_verify_compile_ok(local_repo: Path, tmp_path: Path, capsys) -> None:
  out = tmp_path / "out.py"
  code = cli.main(
      _args(local_repo, "Model", "--out", str(out), "--report", "none", "--cache-dir",
            str(tmp_path / "c"), "--verify-compile"))
  assert code == 0
  assert "verify-compile: OK" in capsys.readouterr().err


def test_cli_verify_import_ok(plain_repo: Path, tmp_path: Path, capsys) -> None:
  out = tmp_path / "out.py"
  code = cli.main(
      _args(plain_repo, "Simple", "--out", str(out), "--report", "none", "--cache-dir",
            str(tmp_path / "c"), "--verify-import"))
  assert code == 0
  assert "verify-import: OK" in capsys.readouterr().err


def test_cli_verify_import_reports_missing_dependency(plain_repo: Path, tmp_path: Path,
                                                      capsys) -> None:
  """Importing a symbol whose (untrusted) dep is absent must fail loudly."""
  (plain_repo / "pkg" / "needs_dep.py").write_text(
      "import definitely_not_installed_pkg_xyz\n\nclass NeedsDep:\n"
      "    x = definitely_not_installed_pkg_xyz.VALUE\n",
      encoding="utf-8")
  out = tmp_path / "out.py"
  code = cli.main(
      _args(plain_repo, "NeedsDep", "--out", str(out), "--report", "none", "--cache-dir",
            str(tmp_path / "c"), "--verify-import"))
  assert code == 1
  assert "verify-import: FAILED" in capsys.readouterr().err


def test_cli_verify_import_failure(plain_repo: Path, tmp_path: Path, capsys) -> None:
  """A missing base class raises NameError at import and must be caught."""
  (plain_repo / "pkg" / "broken.py").write_text("class Broken(nonexistent_base):\n    pass\n",
                                                encoding="utf-8")
  out = tmp_path / "out.py"
  code = cli.main(
      _args(plain_repo, "Broken", "--out", str(out), "--report", "none", "--cache-dir",
            str(tmp_path / "c"), "--verify-import"))
  assert code == 1
  assert "verify-import: FAILED" in capsys.readouterr().err


def test_cli_no_verify_by_default(local_repo: Path, tmp_path: Path, capsys) -> None:
  out = tmp_path / "out.py"
  code = cli.main(
      _args(local_repo, "Model", "--out", str(out), "--report", "none", "--cache-dir",
            str(tmp_path / "c")))
  assert code == 0
  assert "verify-" not in capsys.readouterr().err
