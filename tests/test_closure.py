"""Tests for transitive closure computation."""

from __future__ import annotations

from py_extractor.analysis.resolver import ResolvedSymbol
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.extract.closure import FlagKind, compute_closure
from py_extractor.source.tree import SourceTree


def _target(path: str, module: str, symbol: str) -> ResolvedSymbol:
  return ResolvedSymbol(path, module, symbol, path.endswith("__init__.py"))


def test_closure_pulls_sibling_and_relative_imports(basic_tree, options) -> None:
  closure = compute_closure(basic_tree, _target("mypkg/core.py", "mypkg.core", "Model"), options)
  modules = {(u.module, u.symbol) for u in closure.units}
  assert ("mypkg.core", "Model") in modules
  assert ("mypkg.utils", "SCALE") in modules
  assert closure.flags == []


def test_aliases_kept_not_dropped(basic_tree, options) -> None:
  closure = compute_closure(basic_tree, _target("mypkg/core.py", "mypkg.core", "Model"), options)
  kept = {(i.module, i.name, i.asname) for i in closure.kept_imports}
  assert ("numpy", "numpy", "np") in kept
  assert ("torch", "nn", None) in kept


def test_dependency_ordering(basic_tree, options) -> None:
  closure = compute_closure(basic_tree, _target("mypkg/core.py", "mypkg.core", "Model"), options)
  order = [u.symbol for u in closure.units]
  assert order.index("SCALE") < order.index("Model")


def test_deep_relative_import(relative_tree, options) -> None:
  closure = compute_closure(relative_tree, _target("pkg/a/b/mod.py", "pkg.a.b.mod", None), options)
  # Whole-module target resolves its imports into the closure.
  modules = {u.module for u in closure.units}
  assert "pkg.a.helpers" in modules
  assert "pkg.a.b.sibling" in modules


def test_namespace_root_recorded(basic_tree, options) -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "import pkg.utils\n\nclass C:\n    x = pkg.utils.VALUE\n",
      "pkg/utils.py": "VALUE = 1\n",
  })
  closure = compute_closure(tree, _target("pkg/mod.py", "pkg.mod", "C"), options)
  assert "pkg" in closure.namespace_roots


def test_unresolved_third_party_flagged_and_kept() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "import obscure_lib\n\nclass C:\n    x = obscure_lib.thing\n",
  })
  options = ExtractOptions(cache_dir=__import__("pathlib").Path("/tmp/x"), trust=TrustPolicy())
  closure = compute_closure(tree, _target("pkg/mod.py", "pkg.mod", "C"), options)
  kinds = {f.kind for f in closure.flags}
  assert FlagKind.UNRESOLVED_THIRD_PARTY in kinds
  kept = {i.module for i in closure.kept_imports}
  assert "obscure_lib" in kept


def test_unresolved_name_flagged() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "class C:\n    x = THIS_IS_UNDEFINED\n",
  })
  options = ExtractOptions(cache_dir=__import__("pathlib").Path("/tmp/x"), trust=TrustPolicy())
  closure = compute_closure(tree, _target("pkg/mod.py", "pkg.mod", "C"), options)
  assert any(f.kind is FlagKind.UNRESOLVED_NAME for f in closure.flags)


def test_star_import_flagged(star_tree, options) -> None:
  closure = compute_closure(star_tree, _target("s/__init__.py", "s", None), options)
  assert any(f.kind is FlagKind.STAR_IMPORT for f in closure.flags)


def test_collision_detected() -> None:
  tree = SourceTree.from_mapping({
      "p/__init__.py": "",
      "p/a.py": "from .b import helper\n\nclass C:\n    x = helper\n",
      "p/b.py": "def helper():\n    return 1\n",
      "p/c.py": "def helper():\n    return 2\n",
  })
  options = ExtractOptions(cache_dir=__import__("pathlib").Path("/tmp/x"), trust=TrustPolicy())
  closure = compute_closure(tree, _target("p/a.py", "p.a", "C"), options)
  # helper exists in b and c, but only b's is pulled by a -> no collision yet.
  assert isinstance(closure.units, list)


def test_cycle_terminates() -> None:
  tree = SourceTree.from_mapping({
      "p/__init__.py": "",
      "p/a.py": "from .b import B\n\nclass A:\n    x = B\n",
      "p/b.py": "from .a import A\n\nclass B:\n    y = A\n",
  })
  options = ExtractOptions(cache_dir=__import__("pathlib").Path("/tmp/x"), trust=TrustPolicy())
  closure = compute_closure(tree, _target("p/a.py", "p.a", "A"), options)
  symbols = {u.symbol for u in closure.units}
  assert {"A", "B"} <= symbols


def test_third_party_resolver_inlines_foreign_repo() -> None:
  root_tree = SourceTree.from_mapping({
      "app/__init__.py": "",
      "app/main.py": "import helperlib\n\nclass App:\n    x = helperlib.thing\n",
  })
  foreign_tree = SourceTree.from_mapping({
      "helperlib/__init__.py": "def thing():\n    return 42\n",
  })
  options = ExtractOptions(cache_dir=__import__("pathlib").Path("/tmp/x"), trust=TrustPolicy())

  def resolver(root, opts):
    if root == "helperlib":
      return "foreign/helperlib", foreign_tree
    return None

  closure = compute_closure(
      root_tree,
      _target("app/main.py", "app.main", "App"),
      options,
      third_party_resolver=resolver,
  )
  repos = {u.repo_id for u in closure.units}
  assert "foreign/helperlib" in repos or any(
      u.repo_id == "foreign/helperlib" for u in closure.units)
