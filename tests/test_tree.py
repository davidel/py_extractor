"""Tests for the in-memory source tree and its module index."""

from __future__ import annotations

import io
import tarfile

from py_extractor.source.tree import SourceTree


def _make_tarball(files: dict[str, str], top: str = "repo-abc123") -> bytes:
  buffer = io.BytesIO()
  with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
    for path, text in files.items():
      data = text.encode()
      info = tarfile.TarInfo(name=f"{top}/{path}")
      info.size = len(data)
      archive.addfile(info, io.BytesIO(data))
  return buffer.getvalue()


def test_add_and_get() -> None:
  tree = SourceTree()
  tree.add("pkg/mod.py", "x = 1\n")
  assert tree.exists("pkg/mod.py")
  assert tree.get("pkg/mod.py") == "x = 1\n"
  assert tree.get("missing.py") is None


def test_package_dirs_detected() -> None:
  tree = SourceTree.from_mapping({"pkg/__init__.py": "", "pkg/mod.py": ""})
  assert tree.is_package_dir("pkg")
  assert not tree.is_package_dir("pkg/mod.py")


def test_resolve_simple_module() -> None:
  tree = SourceTree.from_mapping({"pkg/__init__.py": "", "pkg/mod.py": ""})
  assert tree.resolve_module("pkg.mod") == "pkg/mod.py"
  assert tree.resolve_module("pkg") == "pkg/__init__.py"


def test_nested_package_preferred() -> None:
  tree = SourceTree.from_mapping({
      "a/__init__.py": "",
      "a/b/__init__.py": "",
      "a/b/c.py": "",
  })
  assert tree.resolve_module("a.b.c") == "a/b/c.py"


def test_src_layout_resolution() -> None:
  tree = SourceTree.from_mapping({"src/mypkg/__init__.py": "", "src/mypkg/mod.py": ""})
  assert tree.resolve_module("mypkg.mod") == "src/mypkg/mod.py"


def test_namespace_package_without_init() -> None:
  tree = SourceTree.from_mapping({"ns/sub/mod.py": "VALUE = 1\n"})
  assert tree.has_module("ns.sub.mod")
  assert tree.resolve_module("ns.sub.mod") == "ns/sub/mod.py"


def test_has_module_and_submodules() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/a.py": "",
      "pkg/b.py": "",
      "pkg/sub/__init__.py": "",
  })
  assert tree.has_module("pkg")
  assert "a" in tree.submodules_of("pkg")
  assert "sub" in tree.submodules_of("pkg")


def test_missing_module_returns_none() -> None:
  tree = SourceTree.from_mapping({"pkg/__init__.py": ""})
  assert tree.resolve_module("pkg.does_not_exist") is None
  assert not tree.has_module("pkg.does_not_exist")


def test_from_tarball_strips_top_level() -> None:
  body = _make_tarball({"pkg/__init__.py": "", "pkg/mod.py": "x = 1\n"})
  tree = SourceTree.from_tarball(body)
  assert tree.exists("pkg/mod.py")
  assert tree.get("pkg/mod.py") == "x = 1\n"


def test_from_tarball_only_python_files() -> None:
  body = _make_tarball({"pkg/mod.py": "", "README.md": "# hi", "setup.cfg": "x"})
  tree = SourceTree.from_tarball(body)
  assert tree.exists("pkg/mod.py")
  assert not tree.exists("README.md")
  assert not tree.exists("setup.cfg")


def test_paths_sorted() -> None:
  tree = SourceTree.from_mapping({"b.py": "", "a.py": "", "c/d.py": ""})
  assert tree.paths() == ["a.py", "b.py", "c/d.py"]
