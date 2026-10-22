"""Tests for source acquisition (local, GitHub tarball, Hugging Face)."""

from __future__ import annotations

import io
import tarfile

import pytest

from py_extractor.config import ExtractOptions
from py_extractor.errors import RepoNotFoundError, SourceError
from py_extractor.source import fetcher as fetcher_mod
from py_extractor.source import hf as hf_mod
from py_extractor.source import load_tree
from py_extractor.source.repo import RepoRef


def _tarball(files: dict[str, str], top: str = "repo-main") -> bytes:
  buffer = io.BytesIO()
  with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
    for path, text in files.items():
      data = text.encode()
      info = tarfile.TarInfo(name=f"{top}/{path}")
      info.size = len(data)
      archive.addfile(info, io.BytesIO(data))
  return buffer.getvalue()


def test_load_tree_local(tmp_path) -> None:
  (tmp_path / "pkg").mkdir()
  (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
  (tmp_path / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
  tree = load_tree(RepoRef("local", "", "repo", path=tmp_path), ExtractOptions())
  assert tree.get("pkg/mod.py") == "x = 1\n"


def test_load_tree_missing_local_raises(tmp_path) -> None:
  ref = RepoRef("local", "", "repo", path=tmp_path / "nope")
  with pytest.raises(RepoNotFoundError):
    load_tree(ref, ExtractOptions())


def test_load_tree_github_uses_tarball(monkeypatch) -> None:
  body = _tarball({"pkg/mod.py": "y = 2\n"})
  monkeypatch.setattr(fetcher_mod, "fetch_bytes", lambda *a, **k: body)
  # Fetcher.fetch_tarball calls the module-level fetch_bytes.
  tree = load_tree(RepoRef("github", "o", "n", "main"), ExtractOptions(cache_dir="/tmp/x"))
  assert "pkg/mod.py" in tree.files


def test_load_tree_github_404_raises(monkeypatch) -> None:

  def boom(*a, **k):
    raise RepoNotFoundError("404")

  monkeypatch.setattr(fetcher_mod, "fetch_bytes", boom)
  with pytest.raises(RepoNotFoundError):
    load_tree(RepoRef("github", "o", "n", "main"), ExtractOptions(cache_dir="/tmp/x2"))


def test_load_hf_tree(monkeypatch) -> None:
  entries = [
      {
          "path": "pkg/mod.py"
      },
      {
          "path": "README.md"
      },
  ]

  class _FakeFetcher:

    def fetch_json(self, url, **kwargs):
      return entries

    def fetch_text(self, url, **kwargs):
      return "z = 3\n"

  tree = hf_mod.load_hf_tree(RepoRef("hf", "owner", "name", "main"), _FakeFetcher())
  assert tree.get("pkg/mod.py") == "z = 3\n"
  assert not tree.exists("README.md")


def test_load_hf_tree_bad_response(monkeypatch) -> None:

  class _FakeFetcher:

    def fetch_json(self, url, **kwargs):
      return {"not": "a list"}

  with pytest.raises(SourceError):
    hf_mod.load_hf_tree(RepoRef("hf", "o", "n", "main"), _FakeFetcher())


def test_load_tree_unsupported_kind() -> None:
  with pytest.raises(RepoNotFoundError):
    load_tree(RepoRef("weird", "o", "n"), ExtractOptions())
