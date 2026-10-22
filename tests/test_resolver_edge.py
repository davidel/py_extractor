"""Edge-case tests for import resolution, re-exports and symbol location."""

from __future__ import annotations

import ast

import pytest

from py_extractor import locate_symbol
from py_extractor.analysis.imports import collect_imports
from py_extractor.analysis.resolver import resolve_import_name
from py_extractor.errors import SymbolNotFoundError
from py_extractor.source.tree import SourceTree


def _imp(source: str, module: str = "pkg.mod", is_package: bool = False):
  return collect_imports(ast.parse(source), module, is_package)


def test_reexport_chain_two_hops() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "from .api import Public\n",
      "pkg/api.py": "from .impl import Public\n",
      "pkg/impl.py": "class Public:\n    pass\n",
  })
  (imp,) = _imp("from pkg import Public\n", "consumer")
  resolved = resolve_import_name(tree, imp, "Public", current_module="consumer")
  assert resolved is not None
  assert resolved.path == "pkg/impl.py"
  assert resolved.symbol == "Public"


def test_reexport_cycle_terminates() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "from .a import Thing\n",
      "pkg/a.py": "from .b import Thing\n",
      "pkg/b.py": "from .a import Thing\n",
  })
  (imp,) = _imp("from pkg import Thing\n", "consumer")
  # Should not infinitely recurse; returns None because Thing is never defined.
  assert resolve_import_name(tree, imp, "Thing", current_module="consumer") is None


def test_resolve_symbol_defined_in_init() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "class Top:\n    pass\n",
  })
  (imp,) = _imp("from pkg import Top\n", "consumer")
  resolved = resolve_import_name(tree, imp, "Top", current_module="consumer")
  assert resolved is not None
  assert resolved.path == "pkg/__init__.py"
  assert resolved.symbol == "Top"


def test_resolve_whole_submodule_import() -> None:
  tree = SourceTree.from_mapping({"a/__init__.py": "", "a/b/__init__.py": ""})
  (imp,) = _imp("import a.b\n", "consumer")
  resolved = resolve_import_name(tree, imp, "a", current_module="consumer")
  assert resolved is not None
  assert resolved.path == "a/b/__init__.py"


def test_syntax_error_file_is_skipped_in_locate() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/broken.py": "def (:\n",
      "pkg/good.py": "class Target:\n    pass\n",
  })
  resolved = locate_symbol(tree, "Target")
  assert resolved.path == "pkg/good.py"


def test_locate_ambiguous_returns_first_deterministically() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/aaa.py": "class Dup:\n    pass\n",
      "pkg/bbb.py": "class Dup:\n    pass\n",
  })
  first = locate_symbol(tree, "Dup")
  second = locate_symbol(tree, "Dup")
  assert first == second
  assert first.path in {"pkg/aaa.py", "pkg/bbb.py"}


def test_locate_method_with_hint() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "class C:\n    def m(self):\n        pass\n",
  })
  resolved = locate_symbol(tree, "C.m", file_hint="pkg/mod.py")
  assert resolved.symbol == "C.m"


def test_locate_unknown_method_raises() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "class C:\n    pass\n",
  })
  with pytest.raises(SymbolNotFoundError):
    locate_symbol(tree, "C.missing", file_hint="pkg/mod.py")
