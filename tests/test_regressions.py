"""Regression tests for concrete bugs found in the original exa.py sketch.

Each test names the bug it guards against, so a future regression is obvious.
"""

from __future__ import annotations

import ast

from py_extractor.analysis.imports import collect_imports
from py_extractor.analysis.resolver import ResolvedSymbol
from py_extractor.analysis.symbols import free_names
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.extract.closure.builder import compute_closure
from py_extractor.extract.closure.models import FlagKind
from py_extractor.extract.emitter import emit
from py_extractor.pypi import PYPI_JSON_URL
from py_extractor.source.tree import SourceTree


def _options() -> ExtractOptions:
  return ExtractOptions(cache_dir=__import__("pathlib").Path("/tmp/x"), trust=TrustPolicy())


def _closure(tree: SourceTree, path: str, module: str, symbol: str | None):
  target = ResolvedSymbol(path, module, symbol, path.endswith("__init__.py"))
  return compute_closure(tree, target, _options(), repo_id="demo/repo")


def test_bug1_pypi_url_is_not_malformed() -> None:
  """exa.py built 'https://pypi.orgrequests/json' (missing '/pypi/')."""
  assert PYPI_JSON_URL.format(dist="requests") == "https://pypi.org/pypi/requests/json"


def test_bug2_whitelisted_from_import_name_is_preserved() -> None:
  """exa.py dropped 'from torch import nn' (kept only 'import torch') -> NameError."""
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "from torch import nn\n\nclass C(nn.Module):\n    pass\n",
  })
  source = emit(_closure(tree, "pkg/mod.py", "pkg.mod", "C"))
  assert "from torch import nn" in source
  ast.parse(source)


def test_bug3_import_aliases_are_preserved() -> None:
  """exa.py emitted 'import numpy' but the body used 'np'."""
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "import numpy as np\n\nclass C:\n    x = np.zeros(3)\n",
  })
  source = emit(_closure(tree, "pkg/mod.py", "pkg.mod", "C"))
  assert "import numpy as np" in source
  ast.parse(source)


def test_bug4_dependency_finder_excludes_locals_and_builtins() -> None:
  """exa.py collected {'i','len','range','total','x'} instead of just free names."""
  src = "def helper(x):\n    total = 0\n    for i in range(len(x)):\n        return total + x[i]\n"
  assert free_names(ast.parse(src).body[0]) == set()


def test_bug5_bare_dot_import_resolves_to_package() -> None:
  """exa.py resolved 'from . import X' to a directory, never considering __init__."""
  imports = collect_imports(ast.parse("from . import sub\n"), "pkg.a.b.mod", False)
  assert imports[0].module == "pkg.a.b"  # not 'pkg/a/b'


def test_bug6_same_name_in_two_files_does_not_collide() -> None:
  """exa.py keyed fetched nodes by bare name -> second 'Conv2d' silently dropped."""
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/one.py": "class Conv2d:\n    tag = 'one'\n\nclass User:\n    x = Conv2d\n",
      "pkg/two.py": "class Conv2d:\n    tag = 'two'\n",
  })
  closure = _closure(tree, "pkg/one.py", "pkg.one", "User")
  # Only one.py's Conv2d is reachable from User, and it is inlined correctly.
  symbols = {(u.path, u.symbol) for u in closure.units}
  assert ("pkg/one.py", "Conv2d") in symbols
  assert all(path != "pkg/two.py" for path, _ in symbols)


def test_bug7_resolution_uses_real_tree_not_path_guessing() -> None:
  """exa.py guessed 'module/path.py' by string-joining; here a src/ layout works."""
  tree = SourceTree.from_mapping({
      "src/mypkg/__init__.py": "",
      "src/mypkg/core.py": "def helper():\n    return 1\n",
      "src/mypkg/user.py": "from .core import helper\n\nclass C:\n    x = helper\n",
  })
  closure = _closure(tree, "src/mypkg/user.py", "mypkg.user", "C")
  modules = {u.module for u in closure.units}
  assert "mypkg.core" in modules


def test_bug8_no_hardcoded_repo_sniffing() -> None:
  """exa.py hardcoded 'repo == DAT or author == w-e-w' to detect Hugging Face."""
  import inspect
  import re

  import py_extractor
  from py_extractor.source import repo as repo_mod

  src = inspect.getsource(repo_mod) + inspect.getsource(py_extractor)
  assert "w-e-w" not in src
  # Whole-word match: a hardcoded ``DAT`` repo/author token must not appear, but
  # unrelated identifiers that merely embed those letters (e.g. ``DATASET``) are
  # fine.
  assert not re.search(r"\bDAT\b", src)


def test_bug_stdlib_auto_whitelisted() -> None:
  """The tool must auto-keep stdlib imports (os, re, sys, ...)."""
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py":
          "import os\nimport re\n\nclass C:\n    path = os.sep\n    pat = re.compile('x')\n",
  })
  closure = _closure(tree, "pkg/mod.py", "pkg.mod", "C")
  kept = {i.module for i in closure.kept_imports}
  assert {"os", "re"} <= kept
  # No warnings for stdlib deps.
  assert not any(f.kind is FlagKind.UNRESOLVED_THIRD_PARTY for f in closure.flags)
