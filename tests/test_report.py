"""Tests for the text/JSON reports and the third-party resolver factory."""

from __future__ import annotations

import json
from pathlib import Path

from py_extractor import make_third_party_resolver
from py_extractor.analysis.resolver import ResolvedSymbol
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.errors import SourceError
from py_extractor.extract.closure import compute_closure
from py_extractor.pypi import PYPI_JSON_URL
from py_extractor.report import render_json, render_text
from py_extractor.source.tree import SourceTree


def _closure() -> object:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py":
          "import numpy as np\nimport obscure\n\nclass C:\n    x = np.zeros(1) + obscure.y\n",
  })
  options = ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy())
  target = ResolvedSymbol("pkg/mod.py", "pkg.mod", "C", False)
  return compute_closure(tree, target, options, repo_id="demo/repo")


def test_render_text_contains_sections() -> None:
  text = render_text(_closure())
  assert "Target: pkg.mod:C" in text
  assert "Inlined definitions" in text
  assert "Imports kept as-is" in text
  assert "Warnings" in text


def test_render_text_no_warnings_message() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "import numpy as np\n\nclass C:\n    x = np.zeros(1)\n",
  })
  options = ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy())
  target = ResolvedSymbol("pkg/mod.py", "pkg.mod", "C", False)
  text = render_text(compute_closure(tree, target, options, repo_id="demo/repo"))
  assert "Warnings: none" in text


def test_render_json_is_valid() -> None:
  payload = json.loads(render_json(_closure()))
  assert payload["target"]["symbol"] == "C"
  assert payload["units"]
  assert payload["kept_imports"]


def _licensed_closure() -> object:
  tree = SourceTree.from_mapping({
      "LICENSE": "MIT License\n\nPermission is hereby granted, free of charge\n"
                 'The software is provided "as is", without warranty of any kind.\n',
      "pkg/__init__.py": "",
      "pkg/mod.py": "class C:\n  pass\n",
  })
  options = ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy())
  target = ResolvedSymbol("pkg/mod.py", "pkg.mod", "C", False)
  return compute_closure(tree, target, options, repo_id="demo/repo")


def test_render_text_contains_licenses_section() -> None:
  text = render_text(_licensed_closure())
  assert "Licenses (1):" in text
  assert "[L1] MIT License (SPDX: MIT)" in text
  assert "demo/repo" in text


def test_render_json_contains_licenses() -> None:
  payload = json.loads(render_json(_licensed_closure()))
  assert payload["licenses"] == [{
      "id": "L1",
      "repo": "demo/repo",
      "name": "MIT License",
      "spdx": "MIT",
      "origin": "file-text",
      "path": "LICENSE",
      "copyright": None,
  }]


class _FakeFetcher:

  def __init__(self, responses):
    self.responses = responses

  def fetch_json(self, url, **kwargs):
    if url in self.responses:
      return self.responses[url]
    raise SourceError(f"404 {url}")


def test_resolver_finds_repo_and_returns_tree(monkeypatch) -> None:
  payload = {"info": {"project_urls": {"Source": "https://github.com/acme/obscure"}}}
  fetcher = _FakeFetcher({PYPI_JSON_URL.format(dist="obscure"): payload})

  foreign_tree = SourceTree.from_mapping({"obscure/__init__.py": "y = 1\n"})

  import py_extractor as pkg

  def fake_load(repo_ref, options):
    assert repo_ref.slug == "acme/obscure"
    return foreign_tree

  monkeypatch.setattr(pkg, "load_tree", fake_load)
  resolver = make_third_party_resolver(fetcher)
  result = resolver("obscure", ExtractOptions(cache_dir=Path("/tmp/x")))
  assert result is not None
  repo_id, tree = result
  assert repo_id == "acme/obscure"
  assert tree.has_module("obscure")


def test_resolver_returns_none_when_no_pypi_match() -> None:
  fetcher = _FakeFetcher({})
  resolver = make_third_party_resolver(fetcher)
  assert resolver("nope", ExtractOptions(cache_dir=Path("/tmp/x"))) is None


def test_resolver_returns_none_when_repo_missing_module(monkeypatch) -> None:
  payload = {"info": {"project_urls": {"Source": "https://github.com/acme/thing"}}}
  fetcher = _FakeFetcher({PYPI_JSON_URL.format(dist="thing"): payload})
  foreign_tree = SourceTree.from_mapping({"other/__init__.py": ""})

  import py_extractor as pkg

  monkeypatch.setattr(pkg, "load_tree", lambda repo_ref, options: foreign_tree)
  resolver = make_third_party_resolver(fetcher)
  assert resolver("thing", ExtractOptions(cache_dir=Path("/tmp/x"))) is None
