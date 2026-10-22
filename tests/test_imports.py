"""Tests for import parsing and relative-import resolution."""

from __future__ import annotations

import ast

import pytest

from py_extractor.analysis.imports import collect_imports, resolve_relative
from py_extractor.errors import ResolutionError


def _imports(source: str, module: str = "pkg.mod", is_package: bool = False):
  return collect_imports(ast.parse(source), module, is_package)


def test_plain_import() -> None:
  (imp,) = _imports("import os\n")
  assert imp.kind == "import"
  assert imp.target == "os"
  assert imp.root == "os"
  assert imp.names[0].asname is None


def test_import_alias_preserved() -> None:
  (imp,) = _imports("import numpy as np\n")
  assert imp.names[0].name == "numpy"
  assert imp.names[0].asname == "np"
  assert imp.names[0].bound_name == "np"


def test_dotted_import_alias() -> None:
  (imp,) = _imports("import a.b.c as abc\n")
  assert imp.target == "a.b.c"
  assert imp.names[0].bound_name == "abc"
  assert imp.root == "a"


def test_from_import_names() -> None:
  (imp,) = _imports("from torch import nn, optim\n")
  assert imp.kind == "from"
  assert imp.module == "torch"
  assert [n.name for n in imp.names] == ["nn", "optim"]
  assert not imp.is_star


def test_from_import_alias() -> None:
  (imp,) = _imports("from torch import nn as NN\n")
  assert imp.names[0].name == "nn"
  assert imp.names[0].asname == "NN"
  assert imp.names[0].bound_name == "NN"


def test_star_import_detected() -> None:
  (imp,) = _imports("from os import *\n")
  assert imp.is_star
  assert imp.explicit_names == []


def test_relative_single_dot_package() -> None:
  (imp,) = _imports("from .utils import x\n", module="pkg.mod")
  assert imp.module == "pkg.utils"
  assert imp.level == 1


def test_relative_double_dot() -> None:
  (imp,) = _imports("from ..helpers import y\n", module="pkg.a.b.mod")
  assert imp.module == "pkg.a.helpers"
  assert imp.level == 2


def test_relative_bare_dot_import() -> None:
  (imp,) = _imports("from . import sibling\n", module="pkg.a.b.mod")
  assert imp.module == "pkg.a.b"


def test_relative_from_package_init() -> None:
  (imp,) = _imports("from . import core\n", module="pkg", is_package=True)
  assert imp.module == "pkg"


def test_nested_imports_collected() -> None:
  source = "def f():\n    import os\n    if True:\n        from x import y\n"
  imports = _imports(source)
  assert {imp.root for imp in imports} == {"os", "x"}


def test_escaping_relative_import_is_skipped() -> None:
  # A relative import that rises above the top-level package is invalid and is
  # skipped rather than aborting analysis of the rest of the file.
  imports = _imports("from ...deep import x\nimport os\n", module="pkg.mod")
  assert {imp.root for imp in imports} == {"os"}


def test_collect_imports_accepts_a_subnode() -> None:
  # collect_imports can be pointed at a single statement (used for guard blocks).
  stmt = ast.parse("try:\n    import ujson\nexcept ImportError:\n    import json\n").body[0]
  imports = collect_imports(stmt, "pkg.mod")
  assert {imp.module for imp in imports} == {"ujson", "json"}


def test_resolve_relative_function() -> None:
  # Level 1 = the module's own package, so 'pkg.a.b.mod' -> 'pkg.a.b'.
  assert resolve_relative("pkg.a.b.mod", False, 1, "helpers") == "pkg.a.b.helpers"
  assert resolve_relative("pkg.a.b.mod", False, 2, "x") == "pkg.a.x"
  # When the module *is* a package, '.' refers to a child of that package.
  assert resolve_relative("pkg.a.b.mod", True, 1, "x") == "pkg.a.b.mod.x"
  assert resolve_relative("pkg.a.b.mod", True, 1, "") == "pkg.a.b.mod"


def test_resolve_relative_escapes_raises() -> None:
  with pytest.raises(ResolutionError):
    resolve_relative("pkg.mod", False, 5, "")
