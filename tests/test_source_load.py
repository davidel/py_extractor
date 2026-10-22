"""Tests for source acquisition (local, GitHub tarball, Hugging Face)."""

from __future__ import annotations

import io
import tarfile

import pytest

from py_extractor import source as source_mod
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


class _FakeHF:
  """A fake fetcher serving a canned HF tree and per-file contents."""

  def __init__(self, entries, files=None, raise_json=None):
    self.entries = entries
    self.files = files or {}
    self.raise_json = raise_json
    self.json_urls: list[str] = []
    self.text_urls: list[str] = []

  def fetch_json(self, url, **kwargs):
    self.json_urls.append(url)
    if self.raise_json is not None:
      raise self.raise_json
    return self.entries

  def fetch_text(self, url, **kwargs):
    self.text_urls.append(url)
    return self.files.get(url, "")


def test_load_hf_tree() -> None:
  entries = [
      {
          "path": "pkg/mod.py"
      },
      {
          "path": "README.md"
      },
  ]
  fake = _FakeHF(entries, {"https://huggingface.co/owner/name/resolve/main/pkg/mod.py": "z = 3\n"})
  tree = hf_mod.load_hf_tree(RepoRef("hf", "owner", "name", "main"), fake)
  assert tree.get("pkg/mod.py") == "z = 3\n"
  assert not tree.exists("README.md")


def test_load_hf_tree_requests_correct_urls() -> None:
  entries = [{"path": "pkg/mod.py", "type": "file"}]
  fake = _FakeHF(entries)
  repo = RepoRef("hf", "owner", "name", "dev")
  hf_mod.load_hf_tree(repo, fake)
  assert fake.json_urls == ["https://huggingface.co/api/models/owner/name/tree/dev?recursive=true"]
  assert fake.text_urls == ["https://huggingface.co/owner/name/resolve/dev/pkg/mod.py"]


def test_load_hf_tree_dataset_prefix() -> None:
  entries = [{"path": "pkg/mod.py", "type": "file"}]
  fake = _FakeHF(entries)
  repo = RepoRef("hf", "owner", "name", "main", repo_type="dataset")
  hf_mod.load_hf_tree(repo, fake)
  assert fake.json_urls == [
      "https://huggingface.co/api/datasets/owner/name/tree/main?recursive=true"
  ]
  assert fake.text_urls == ["https://huggingface.co/datasets/owner/name/resolve/main/pkg/mod.py"]


def test_load_hf_tree_keeps_pyi_files() -> None:
  entries = [
      {
          "path": "pkg/mod.py",
          "type": "file"
      },
      {
          "path": "pkg/mod.pyi",
          "type": "file"
      },
  ]
  fake = _FakeHF(entries)
  tree = hf_mod.load_hf_tree(RepoRef("hf", "o", "n", "main"), fake)
  assert tree.exists("pkg/mod.py")
  assert tree.exists("pkg/mod.pyi")


def test_load_hf_tree_skips_directories_and_non_dicts() -> None:
  entries = [
      {
          "path": "pkg",
          "type": "directory"
      },
      {
          "path": "vendor",
          "type": "directory"
      },
      "not-a-dict",
      None,
      {
          "path": "pkg/mod.py",
          "type": "file"
      },
      {
          "path": "notes.txt",
          "type": "file"
      },
      {
          "type": "file"
      },  # no path
  ]
  fake = _FakeHF(entries)
  tree = hf_mod.load_hf_tree(RepoRef("hf", "o", "n", "main"), fake)
  assert tree.paths() == ["pkg/mod.py"]
  assert fake.text_urls == ["https://huggingface.co/o/n/resolve/main/pkg/mod.py"]


def test_load_hf_tree_handles_packages_and_nested_dirs() -> None:
  entries = [
      {
          "path": "pkg/__init__.py",
          "type": "file"
      },
      {
          "path": "pkg/sub/__init__.py",
          "type": "file"
      },
      {
          "path": "pkg/sub/mod.py",
          "type": "file"
      },
  ]
  files = {
      "https://huggingface.co/o/n/resolve/main/pkg/__init__.py": "",
      "https://huggingface.co/o/n/resolve/main/pkg/sub/__init__.py": "",
      "https://huggingface.co/o/n/resolve/main/pkg/sub/mod.py": "v = 1\n",
  }
  fake = _FakeHF(entries, files)
  tree = hf_mod.load_hf_tree(RepoRef("hf", "o", "n", "main"), fake)
  assert tree.is_package_dir("pkg")
  assert tree.get("pkg/sub/mod.py") == "v = 1\n"


def test_load_hf_tree_empty() -> None:
  fake = _FakeHF([])
  tree = hf_mod.load_hf_tree(RepoRef("hf", "o", "n", "main"), fake)
  assert tree.paths() == []


def test_load_hf_tree_bad_response() -> None:
  fake = _FakeHF({"not": "a list"})
  with pytest.raises(SourceError):
    hf_mod.load_hf_tree(RepoRef("hf", "o", "n", "main"), fake)


def test_load_hf_tree_missing_ref_hint() -> None:
  fake = _FakeHF([], raise_json=RepoNotFoundError("404"))
  with pytest.raises(RepoNotFoundError, match="pass --ref"):
    hf_mod.load_hf_tree(RepoRef("hf", "o", "n", "main"), fake)


def test_load_tree_dispatches_to_hf(monkeypatch) -> None:
  entries = [{"path": "pkg/mod.py", "type": "file"}]
  fake = _FakeHF(entries, {"https://huggingface.co/o/n/resolve/main/pkg/mod.py": "k = 9\n"})
  monkeypatch.setattr(source_mod, "Fetcher", lambda *a, **k: fake)
  tree = load_tree(RepoRef("hf", "o", "n", "main"), ExtractOptions(cache_dir="/tmp/hf"))
  assert tree.get("pkg/mod.py") == "k = 9\n"
  assert fake.json_urls == ["https://huggingface.co/api/models/o/n/tree/main?recursive=true"]


def test_load_tree_unsupported_kind() -> None:
  with pytest.raises(RepoNotFoundError):
    load_tree(RepoRef("weird", "o", "n"), ExtractOptions())
