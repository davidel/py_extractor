"""Local-name bindings: inlined definitions must stay reachable at call sites.

When a file imports a name and uses it through an alias (``from .core import
Widget as W``) or a module object (``from . import core`` / ``import pkg.core``),
inlining the definition alone leaves the call site dangling. These tests execute
the generated output to prove the binding is recreated.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from py_extractor.analysis.resolver import ResolvedSymbol
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.extract.closure.builder import compute_closure
from py_extractor.extract.emitter import emit
from py_extractor.source.tree import SourceTree

CORE = "class Widget:\n  def tag(self):\n    return 'w'\n"


def _options() -> ExtractOptions:
  return ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy())


def _closure(tree: SourceTree, symbol: str | None):
  target = ResolvedSymbol("pkg/mod.py", "pkg.mod", symbol, False)
  return compute_closure(tree, target, _options(), repo_id="demo/pkg")


def _run(source: str) -> dict:
  ns: dict = {}
  exec(compile(source, "<out>", "exec"), ns)  # noqa: S102 - intentional in test
  return ns


@pytest.mark.parametrize(
    ("mod_source", "call"),
    [
        ("from .core import Widget as W\n\ndef build():\n  return W().tag()\n", "build"),
        ("from . import core\n\ndef build():\n  return core.Widget().tag()\n", "build"),
        ("import pkg.core\n\ndef build():\n  return pkg.core.Widget().tag()\n", "build"),
        ("import pkg.core as c\n\ndef build():\n  return c.Widget().tag()\n", "build"),
    ],
)
def test_alias_and_module_bindings_execute(mod_source: str, call: str) -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/core.py": CORE,
      "pkg/mod.py": mod_source,
  })
  source = emit(_closure(tree, None), include_report=False)
  ast.parse(source)
  assert _run(source)[call]() == "w"


def test_alias_binding_on_named_symbol_target() -> None:
  """The alias fix applies to the normal (non-whole-module) path too."""
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/core.py":
          "class Helper:\n  def hi(self):\n    return 'hi'\n",
      "pkg/mod.py": ("from .core import Helper as H\n\n\nclass Widget:\n"
                     "  def f(self):\n    return H().hi()\n"),
  })
  source = emit(_closure(tree, "Widget"), include_report=False)
  assert "H = Helper" in source
  assert _run(source)["Widget"]().f() == "hi"


def test_bare_from_import_needs_no_alias_binding() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/core.py": CORE,
      "pkg/mod.py": "from .core import Widget\n\ndef build():\n  return Widget().tag()\n",
  })
  closure = _closure(tree, None)
  assert all(b.kind != "alias" for b in closure.bindings)
  assert _run(emit(closure, include_report=False))["build"]() == "w"


def test_module_binding_with_no_members_is_skipped() -> None:
  # An empty module has nothing to expose; no binding should be synthesised.
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/empty.py": "# nothing here\n",
      "pkg/mod.py": "from . import empty\n\ndef build():\n  return 1\n",
  })
  closure = _closure(tree, None)
  assert all(b.kind != "module" for b in closure.bindings)


def test_module_binding_members_exclude_dunder() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/core.py": "__all__ = ['Widget']\n\nclass Widget:\n  pass\n\nX = 1\n",
      "pkg/mod.py": "from . import core\n\ndef build():\n  return (core.Widget, core.X)\n",
  })
  closure = _closure(tree, None)
  module_bindings = [b for b in closure.bindings if b.kind == "module"]
  assert len(module_bindings) == 1
  assert "__all__" not in module_bindings[0].members
  assert set(module_bindings[0].members) >= {"Widget", "X"}


def test_duplicate_alias_collision_is_flagged() -> None:
  # Two inlined files each alias a different definition to the same local name;
  # in the flattened single file only one can win, so it must be flagged.
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/a.py":
          "class A:\n  pass\n",
      "pkg/b.py":
          "class B:\n  pass\n",
      "pkg/one.py":
          "from .a import A as W\n\ndef use1():\n  return W()\n",
      "pkg/two.py":
          "from .b import B as W\n\ndef use2():\n  return W()\n",
      "pkg/mod.py": ("from .one import use1\nfrom .two import use2\n\n"
                     "def build():\n  return use1(), use2()\n"),
  })
  closure = _closure(tree, None)
  assert any(f.kind.value == "collision" for f in closure.flags)


def test_whole_module_relative_import_still_flattens() -> None:
  """Item 1 baseline: a plain relative import resolves and flattens (no alias)."""
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/core.py": CORE,
      "pkg/mod.py": "from .core import Widget\n\ndef build():\n  return Widget()\n",
  })
  closure = _closure(tree, None)
  modules = {u.module for u in closure.units}
  assert "pkg.core" in modules
  assert closure.bindings == []
  assert _run(emit(closure, include_report=False))["build"]().tag() == "w"


def test_bindings_render_in_text_and_json_reports() -> None:
  from py_extractor.report import render_json, render_text

  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/core.py": CORE,
      "pkg/mod.py": "from .core import Widget as W\n\ndef build():\n  return W()\n",
  })
  closure = _closure(tree, None)
  assert "W = Widget" in render_text(closure)
  payload = json.loads(render_json(closure))
  assert payload["bindings"][0]["local"] == "W"
  assert payload["bindings"][0]["kind"] == "alias"
