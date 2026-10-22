"""Tests for high-level symbol location."""

from __future__ import annotations

import pytest

from py_extractor import locate_symbol
from py_extractor.errors import AmbiguousSymbolError, SymbolNotFoundError
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
  """A symbol defined in two files raises; the name-matching file is recommended."""
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/model.py": "class Model:\n    pass\n",
      "pkg/other.py": "class Model:\n    pass\n",
  })
  with pytest.raises(AmbiguousSymbolError) as exc:
    locate_symbol(tree, "Model")
  assert exc.value.candidates[0] == "pkg/model.py"


def test_unique_symbol_still_auto_resolves() -> None:
  """A single match needs no prompt/choose callback and never raises."""
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/model.py": "class Model:\n    pass\n",
  })
  assert locate_symbol(tree, "Model").path == "pkg/model.py"


def test_ambiguous_symbol_raises_with_candidates() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/aaa.py": "class Dup:\n    pass\n",
      "pkg/bbb.py": "class Dup:\n    pass\n",
  })
  with pytest.raises(AmbiguousSymbolError) as exc:
    locate_symbol(tree, "Dup")
  assert set(exc.value.candidates) == {"pkg/aaa.py", "pkg/bbb.py"}
  assert exc.value.symbol == "Dup"


def test_choose_callback_resolves_ambiguity() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/aaa.py": "class Dup:\n    pass\n",
      "pkg/bbb.py": "class Dup:\n    pass\n",
  })
  seen = {}

  def choose(symbol: str, candidates: list[str]) -> str:
    seen["symbol"] = symbol
    seen["candidates"] = candidates
    return "pkg/bbb.py"

  resolved = locate_symbol(tree, "Dup", choose=choose)
  assert resolved.path == "pkg/bbb.py"
  assert seen["symbol"] == "Dup"
  assert set(seen["candidates"]) == {"pkg/aaa.py", "pkg/bbb.py"}


def test_choose_declining_raises() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/aaa.py": "class Dup:\n    pass\n",
      "pkg/bbb.py": "class Dup:\n    pass\n",
  })
  with pytest.raises(AmbiguousSymbolError):
    locate_symbol(tree, "Dup", choose=lambda symbol, candidates: None)


def test_choose_invalid_path_raises() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/aaa.py": "class Dup:\n    pass\n",
      "pkg/bbb.py": "class Dup:\n    pass\n",
  })
  with pytest.raises(SymbolNotFoundError):
    locate_symbol(tree, "Dup", choose=lambda symbol, candidates: "pkg/nope.py")


def test_file_hint_overrides_ambiguity() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/aaa.py": "class Dup:\n    pass\n",
      "pkg/bbb.py": "class Dup:\n    pass\n",
  })
  resolved = locate_symbol(tree, "Dup", file_hint="pkg/aaa.py")
  assert resolved.path == "pkg/aaa.py"


def test_ranked_candidates_put_module_named_file_first() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/zmodel.py": "class Model:\n    pass\n",
      "pkg/model.py": "class Model:\n    pass\n",
  })
  with pytest.raises(AmbiguousSymbolError) as exc:
    locate_symbol(tree, "Model")
  # 'pkg/model.py' matches the symbol name and must be the recommended candidate.
  assert exc.value.candidates[0] == "pkg/model.py"
