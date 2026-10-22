"""End-to-end tests for the public extract_symbol() API."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from py_extractor import extract_symbol
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.errors import SymbolNotFoundError


@pytest.fixture
def repo(tmp_path: Path) -> Path:
  base = tmp_path / "repo"
  (base / "app").mkdir(parents=True)
  (base / "app" / "__init__.py").write_text("", encoding="utf-8")
  (base / "app" / "main.py").write_text(
      "import os\n"
      "import numpy as np\n"
      "from torch import nn\n"
      "from .helpers import SCALE\n"
      "\n"
      "class App(nn.Module):\n"
      "    def run(self):\n"
      "        return np.zeros(3) * SCALE + os.getpid()\n",
      encoding="utf-8",
  )
  (base / "app" / "helpers.py").write_text("SCALE = 2\n", encoding="utf-8")
  return base


def test_extract_symbol_local(repo: Path, tmp_path: Path) -> None:
  result = extract_symbol(
      str(repo),
      "App",
      options=ExtractOptions(cache_dir=tmp_path / "c", trust=TrustPolicy()),
      fetch_third_party=False,
  )
  ast.parse(result.source)
  assert "class App" in result.source
  assert "SCALE = 2" in result.source
  assert "import numpy as np" in result.source
  assert "from torch import nn" in result.source


def test_extract_symbol_kwargs_build_policy(repo: Path, tmp_path: Path) -> None:
  result = extract_symbol(
      str(repo),
      "App",
      trusted={"torch"},
      cache_dir=tmp_path / "c",
      fetch_third_party=False,
  )
  assert "from torch import nn" in result.source


def test_extract_symbol_strict_trust(repo: Path, tmp_path: Path) -> None:
  result = extract_symbol(
      str(repo),
      "App",
      strict_trust=True,
      cache_dir=tmp_path / "c",
      fetch_third_party=False,
  )
  # numpy/torch no longer trusted -> kept as imports anyway, but flagged.
  ast.parse(result.source)
  assert "import numpy" in result.source


def test_extract_symbol_missing_symbol(repo: Path, tmp_path: Path) -> None:
  with pytest.raises(SymbolNotFoundError) as exc:
    extract_symbol(
        str(repo),
        "Nope",
        options=ExtractOptions(cache_dir=tmp_path / "c"),
        fetch_third_party=False,
    )
  # The repo-wide message points at the nested-member form.
  assert "not found anywhere in repository" in str(exc.value)
  assert "'Class.member'" in str(exc.value)


def test_extract_symbol_file_hint(repo: Path, tmp_path: Path) -> None:
  result = extract_symbol(
      str(repo),
      "App",
      file_hint="app/main.py",
      options=ExtractOptions(cache_dir=tmp_path / "c"),
      fetch_third_party=False,
  )
  assert result.target.path == "app/main.py"


def test_offline_option_refuses_network(tmp_path: Path) -> None:
  from py_extractor.errors import SourceError

  with pytest.raises(SourceError):
    extract_symbol(
        "github:psf/requests",
        "Request",
        options=ExtractOptions(cache_dir=tmp_path / "c", offline=True),
        fetch_third_party=False,
    )


def test_extract_with_third_party_resolver_inlines_foreign(repo: Path, tmp_path: Path) -> None:
  """A full run where a third-party dep is resolved from another repo."""
  from py_extractor.source.tree import SourceTree

  (repo / "app" / "uses.py").write_text(
      "import helperlib\n\nclass Uses:\n    value = helperlib.thing\n", encoding="utf-8")

  foreign = SourceTree.from_mapping({"helperlib/__init__.py": "def thing():\n    return 1\n"})

  captured = {}

  def resolver_factory(fetcher):

    def resolver(root, options):
      if root == "helperlib":
        captured["called"] = True
        return "acme/helperlib", foreign
      return None

    return resolver

  import py_extractor as pkg

  original = pkg.make_third_party_resolver
  try:
    pkg.make_third_party_resolver = resolver_factory
    result = extract_symbol(
        str(repo),
        "Uses",
        options=ExtractOptions(cache_dir=tmp_path / "c", trust=TrustPolicy()),
    )
  finally:
    pkg.make_third_party_resolver = original

  assert captured.get("called")
  assert "acme/helperlib" in result.source or "helperlib" in result.source
  ast.parse(result.source)
