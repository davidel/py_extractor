"""Offline end-to-end extraction from a (faked) Hugging Face repository.

These tests exercise the full HF path -- ``parse_repo`` -> ``load_tree`` ->
``load_hf_tree`` -> ``extract_symbol`` -- without touching the network by
swapping in a fake fetcher that serves the tree API JSON and per-file text.
"""

from __future__ import annotations

import ast

import py_extractor
from py_extractor import source as source_mod
from py_extractor.config import ExtractOptions
from py_extractor.errors import SourceError
from py_extractor.source.repo import RepoRef

_LAST: dict[str, list[str]] = {"json": [], "text": []}


class _FakeFetcher:
  """A stand-in for :class:`Fetcher` serving a canned HF repo."""

  def __init__(self, cache_dir=None, offline=False, **_):
    self.cache_dir = cache_dir
    self.offline = offline

  def fetch_json(self, url, **kwargs):
    _LAST["json"].append(url)
    return [
        {
            "path": "pkg/__init__.py",
            "type": "file"
        },
        {
            "path": "pkg/core.py",
            "type": "file"
        },
        {
            "path": "README.md",
            "type": "file"
        },
    ]

  def fetch_text(self, url, **kwargs):
    _LAST["text"].append(url)
    if url.endswith("pkg/__init__.py"):
      return "from .core import Widget\n"
    if url.endswith("pkg/core.py"):
      return ("import math\n\n\nclass Widget:\n  def area(self, r):\n"
              "    return math.pi * r * r\n")
    raise SourceError(f"unexpected URL {url}")


def _install_fake_fetcher(monkeypatch) -> None:
  _LAST["json"].clear()
  _LAST["text"].clear()
  monkeypatch.setattr(py_extractor, "Fetcher", _FakeFetcher)
  monkeypatch.setattr(source_mod, "Fetcher", _FakeFetcher)


def test_extract_from_hf_repo_offline(monkeypatch, tmp_path) -> None:
  _install_fake_fetcher(monkeypatch)
  result = py_extractor.extract_symbol(
      "hf:owner/name",
      "Widget",
      options=ExtractOptions(cache_dir=tmp_path / "c"),
      fetch_third_party=False,
  )
  ast.parse(result.source)  # valid Python
  assert "class Widget" in result.source
  assert "import math" in result.source  # stdlib kept as an import
  assert "def area" in result.source
  assert _LAST["json"] == ["https://huggingface.co/api/models/owner/name/tree/HEAD?recursive=true"]


def test_extract_from_hf_dataset_repo_offline(monkeypatch, tmp_path) -> None:
  _install_fake_fetcher(monkeypatch)
  result = py_extractor.extract_symbol(
      "hf:datasets/owner/name",
      "Widget",
      options=ExtractOptions(cache_dir=tmp_path / "c"),
      fetch_third_party=False,
  )
  ast.parse(result.source)
  assert _LAST["json"] == [
      "https://huggingface.co/api/datasets/owner/name/tree/HEAD?recursive=true"
  ]
  assert all("/datasets/owner/name/resolve/HEAD/" in u for u in _LAST["text"])


def test_hf_repo_type_is_parsed_for_url_building(monkeypatch, tmp_path) -> None:
  _install_fake_fetcher(monkeypatch)
  result = py_extractor.extract_symbol(
      "hf:spaces/owner/name",
      "Widget",
      options=ExtractOptions(cache_dir=tmp_path / "c"),
      fetch_third_party=False,
  )
  assert "class Widget" in result.source
  assert _LAST["json"] == ["https://huggingface.co/api/spaces/owner/name/tree/HEAD?recursive=true"]
  assert all("/spaces/owner/name/resolve/HEAD/" in u for u in _LAST["text"])


def test_load_tree_hf_uses_repo_type(monkeypatch) -> None:
  _install_fake_fetcher(monkeypatch)
  tree = source_mod.load_tree(RepoRef("hf", "o", "n", "main", repo_type="dataset"))
  assert tree.exists("pkg/core.py")
