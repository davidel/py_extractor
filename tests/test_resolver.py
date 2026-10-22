"""Tests for resolving imports to concrete locations in the source tree."""

from __future__ import annotations

import ast

from py_extractor.analysis.imports import collect_imports
from py_extractor.analysis.resolver import resolve_import_name
from py_extractor.source.tree import SourceTree


def _imp(source: str, module: str = "pkg.mod", is_package: bool = False):
  return collect_imports(ast.parse(source), module, is_package)


def test_resolve_symbol_in_sibling_module() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "from .utils import SCALE\n",
      "pkg/utils.py": "SCALE = 2.0\n",
  })
  (imp,) = _imp("from .utils import SCALE\n", "pkg.mod")
  resolved = resolve_import_name(tree, imp, "SCALE", current_module="pkg.mod")
  assert resolved is not None
  assert resolved.path == "pkg/utils.py"
  assert resolved.symbol == "SCALE"


def test_resolve_submodule_as_name() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "from . import utils\n",
      "pkg/utils.py": "VALUE = 1\n",
  })
  (imp,) = _imp("from . import utils\n", "pkg.mod")
  resolved = resolve_import_name(tree, imp, "utils", current_module="pkg.mod")
  assert resolved is not None
  assert resolved.path == "pkg/utils.py"
  assert resolved.symbol is None


def test_resolve_reexport_via_init() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "from .core import Model\n",
      "pkg/core.py": "class Model:\n    pass\n",
  })
  (imp,) = _imp("from pkg import Model\n", "consumer")
  resolved = resolve_import_name(tree, imp, "Model", current_module="consumer")
  assert resolved is not None
  assert resolved.path == "pkg/core.py"
  assert resolved.symbol == "Model"


def test_resolve_import_target_whole_module() -> None:
  tree = SourceTree.from_mapping({"pkg/__init__.py": "", "pkg/utils.py": ""})
  (imp,) = _imp("import pkg.utils\n", "consumer")
  resolved = resolve_import_name(tree, imp, "pkg", current_module="consumer")
  assert resolved is not None
  assert resolved.path == "pkg/utils.py"


def test_unresolved_external_returns_none() -> None:
  tree = SourceTree.from_mapping({"pkg/__init__.py": ""})
  (imp,) = _imp("import numpy\n", "pkg.mod")
  assert resolve_import_name(tree, imp, "numpy", current_module="pkg.mod") is None


def test_resolve_relative_deep() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/a/__init__.py": "",
      "pkg/a/b/__init__.py": "",
      "pkg/a/b/mod.py": "from ..helpers import assist\n",
      "pkg/a/helpers.py": "def assist():\n    return 1\n",
  })
  (imp,) = _imp("from ..helpers import assist\n", "pkg.a.b.mod")
  resolved = resolve_import_name(tree, imp, "assist", current_module="pkg.a.b.mod")
  assert resolved is not None
  assert resolved.path == "pkg/a/helpers.py"
  assert resolved.symbol == "assist"


def test_src_layout_resolution() -> None:
  tree = SourceTree.from_mapping({
      "src/mypkg/__init__.py": "",
      "src/mypkg/core.py": "def helper():\n    return 1\n",
      "src/mypkg/user.py": "from .core import helper\n",
  })
  (imp,) = _imp("from .core import helper\n", "mypkg.user")
  resolved = resolve_import_name(tree, imp, "helper", current_module="mypkg.user")
  assert resolved is not None
  assert resolved.path == "src/mypkg/core.py"
