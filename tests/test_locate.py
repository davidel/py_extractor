"""Tests for high-level symbol location."""

from __future__ import annotations

import pytest

from py_extractor import locate_symbol
from py_extractor.errors import SymbolNotFoundError
from py_extractor.source.tree import SourceTree


@pytest.fixture
def tree() -> SourceTree:
  return SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "class Model:\n    pass\n\ndef helper():\n    pass\n",
      "pkg/other.py": "class Other:\n    pass\n",
  })


def test_locate_by_name(tree: SourceTree) -> None:
  resolved = locate_symbol(tree, "Model")
  assert resolved.path == "pkg/mod.py"
  assert resolved.symbol == "Model"


def test_locate_with_file_hint(tree: SourceTree) -> None:
  resolved = locate_symbol(tree, "Other", file_hint="pkg/other.py")
  assert resolved.path == "pkg/other.py"


def test_locate_with_module_hint(tree: SourceTree) -> None:
  resolved = locate_symbol(tree, "Other", file_hint="pkg.other")
  assert resolved.path == "pkg/other.py"


def test_locate_method(tree: SourceTree) -> None:
  resolved = locate_symbol(tree, "Model", file_hint="pkg/mod.py")
  assert resolved.symbol == "Model"


def test_locate_missing_raises(tree: SourceTree) -> None:
  with pytest.raises(SymbolNotFoundError):
    locate_symbol(tree, "DoesNotExist")


def test_locate_bad_hint_raises(tree: SourceTree) -> None:
  with pytest.raises(SymbolNotFoundError):
    locate_symbol(tree, "Model", file_hint="pkg/nope.py")


def test_locate_prefers_module_named_after_symbol() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/model.py": "class Model:\n    pass\n",
      "pkg/other.py": "class Model:\n    pass\n",
  })
  resolved = locate_symbol(tree, "Model")
  assert resolved.path == "pkg/model.py"
