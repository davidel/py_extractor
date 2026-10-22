"""Optional/conditional imports: guards are preserved, locals are not hoisted.

Hoisting an import out of its ``try``/``except ImportError`` guard turns an
optional dependency into a hard one (and, when the fallback rebinds the same
name, silently drops the fallback). Function/class-local imports must likewise
stay with their definition instead of being lifted to module scope, which would
duplicate imports and break in-repo references. These tests execute the
generated output to prove the idioms survive.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from py_extractor.analysis.imports import GUARD_TYPES, collect_imports
from py_extractor.analysis.resolver import ResolvedSymbol
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.extract.closure import compute_closure
from py_extractor.extract.emitter import emit
from py_extractor.source.tree import SourceTree

CORE = "class Widget:\n  def tag(self):\n    return 'w'\n"


def _options() -> ExtractOptions:
  return ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy())


def _closure(tree: SourceTree, symbol: str | None, path: str = "pkg/mod.py"):
  module = path[:-3].replace("/", ".")
  target = ResolvedSymbol(path, module, symbol, False)
  return compute_closure(tree, target, _options(), repo_id="demo/pkg")


def _run(source: str) -> dict:
  ns: dict = {"__name__": "extracted"}
  exec(compile(source, "<out>", "exec"), ns)  # noqa: S102 - intentional in test
  return ns


# --- collect_imports records scope and guard --------------------------------


def test_collect_imports_records_module_scope_without_guard() -> None:
  (imp,) = collect_imports(ast.parse("import os\n"), "pkg.mod")
  assert imp.scope == "module"
  assert imp.guard is None


def test_collect_imports_records_guard_for_try_block() -> None:
  imports = collect_imports(ast.parse("try:\n  import ujson\nexcept ImportError:\n  import json\n"),
                            "pkg.mod")
  assert all(imp.scope == "module" for imp in imports)
  assert all(isinstance(imp.guard, GUARD_TYPES) for imp in imports)
  assert {imp.module for imp in imports} == {"ujson", "json"}


def test_collect_imports_records_nested_scope_for_local_import() -> None:
  (imp,) = collect_imports(ast.parse("def f():\n  import os\n"), "pkg.mod")
  assert imp.scope == "nested"
  assert imp.guard is None


def test_collect_imports_multi_alias_yields_one_record_per_alias() -> None:
  imports = collect_imports(ast.parse("import os, sys\n"), "pkg.mod")
  assert {imp.target for imp in imports} == {"os", "sys"}


# --- the try/except optional-import idiom is preserved verbatim -------------


def test_try_except_fallback_is_preserved_verbatim() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("try:\n  import ujson as json\nexcept ImportError:\n  import json\n\n"
                     "def dumps():\n  return json.dumps({'a': 1})\n"),
  })
  closure = _closure(tree, "dumps")
  source = emit(closure, include_report=False)
  # The guard is copied, not hoisted: 'ujson' must not become a bare import.
  assert "try:" in source
  assert "except ImportError:" in source
  assert closure.guarded_blocks
  assert _run(source)["dumps"]() == '{"a": 1}'


def test_preserved_guard_binds_names_for_dependents() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("try:\n  import ujson as json\nexcept ImportError:\n  import json\n\n"
                     "class C:\n  x = json.dumps([])\n"),
  })
  closure = _closure(tree, "C")
  # 'json' is provided by the preserved block; it must not be hoisted or flagged.
  assert all(i.module not in ("ujson", "json") for i in closure.kept_imports)
  assert not any(f.kind.value == "unresolved_name" for f in closure.flags)
  assert _run(emit(closure, include_report=False))["C"].x == "[]"


def test_version_gate_imports_are_preserved() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("import sys\n"
                     "if sys.version_info >= (3, 8):\n"
                     "  import importlib.metadata as md\n"
                     "else:\n"
                     "  import importlib_metadata as md\n\n"
                     "def version():\n  return md\n"),
  })
  closure = _closure(tree, "version")
  source = emit(closure, include_report=False)
  assert "if sys.version_info" in source
  # 'sys' is still needed by the preserved condition and must be hoisted.
  assert any(i.module == "sys" for i in closure.kept_imports)
  assert _run(source)["version"]() is not None


def test_guard_with_in_repo_import_is_not_preserved() -> None:
  # The guard imports an in-repo module, which must be flattened; the guard
  # cannot be copied as-is, so the import is resolved normally instead.
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/core.py":
          CORE,
      "pkg/mod.py": ("try:\n  from .core import Widget\nexcept ImportError:\n"
                     "  Widget = None\n\ndef build():\n  return Widget().tag()\n"),
  })
  closure = _closure(tree, "build")
  assert not closure.guarded_blocks
  assert "pkg.core" in {u.module for u in closure.units}
  assert any(f.kind.value == "guarded_import" for f in closure.flags)
  assert _run(emit(closure, include_report=False))["build"]() == "w"


def test_guard_referencing_sibling_definition_is_not_preserved() -> None:
  # The guard's condition uses a sibling definition that is inlined *after* the
  # guarded block; emitting the block verbatim would raise NameError.
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("FLAG = True\n"
                     "if FLAG:\n"
                     "  import os\n"
                     "\n"
                     "def use():\n  return os.sep\n"),
  })
  closure = _closure(tree, "use")
  assert not closure.guarded_blocks
  assert any(i.module == "os" for i in closure.kept_imports)
  assert _run(emit(closure, include_report=False))["use"]()


def test_guard_binding_unneeded_name_is_skipped() -> None:
  # The optional import binds a name the extracted code never uses, so the whole
  # block is dropped rather than preserved.
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("try:\n  import ujson\nexcept ImportError:\n  ujson = None\n\n"
                     "def use():\n  return 1\n"),
  })
  closure = _closure(tree, "use")
  assert not closure.guarded_blocks
  assert _run(emit(closure, include_report=False))["use"]() == 1


def test_guard_with_absolute_in_repo_import_is_not_preserved() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/core.py":
          CORE,
      "pkg/mod.py": ("try:\n  from pkg.core import Widget\nexcept ImportError:\n"
                     "  Widget = None\n\ndef build():\n  return Widget().tag()\n"),
  })
  closure = _closure(tree, "build")
  assert not closure.guarded_blocks
  assert "pkg.core" in {u.module for u in closure.units}
  assert _run(emit(closure, include_report=False))["build"]() == "w"


def test_guard_without_imports_is_ignored() -> None:
  # A conditional statement with no imports is not a guard and is left alone.
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": ("if True:\n  _X = 1\n\n"
                     "def use():\n  return 2\n"),
  })
  closure = _closure(tree, "use")
  assert not closure.guarded_blocks
  assert _run(emit(closure, include_report=False))["use"]() == 2


# --- function/class-local imports are not hoisted ---------------------------


def test_function_local_import_is_not_hoisted() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("def helper():\n  import os\n  return os.sep\n\n"
                     "class C:\n  x = helper()\n"),
  })
  closure = _closure(tree, "C")
  assert closure.kept_imports == []
  source = emit(closure, include_report=False)
  assert source.count("import os") == 1
  assert _run(source)["C"].x  # non-empty separator


def test_class_body_import_is_not_hoisted() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/mod.py": "class C:\n  import os\n  x = os.sep\n",
  })
  closure = _closure(tree, "C")
  assert closure.kept_imports == []
  assert _run(emit(closure, include_report=False))["C"].x


def test_function_local_in_repo_import_is_flagged() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/core.py":
          CORE,
      "pkg/mod.py": ("def helper():\n  from .core import Widget\n  return Widget().tag()\n\n"
                     "class C:\n  x = helper()\n"),
  })
  closure = _closure(tree, "C")
  assert any(f.kind.value == "nested_import" for f in closure.flags)


def test_duplicate_import_across_inlined_modules_is_emitted_once() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/core.py":
          "import os\n\nclass Widget:\n  sep = os.sep\n",
      "pkg/user.py": ("import os\nfrom .core import Widget\n\n"
                      "class User:\n  w = Widget\n  s = os.sep\n"),
  })
  closure = _closure(tree, "User", path="pkg/user.py")
  source = emit(closure, include_report=False)
  assert source.count("import os") == 1


# --- reporting --------------------------------------------------------------


def test_guarded_blocks_render_in_text_and_json_reports() -> None:
  from py_extractor.report import render_json, render_text

  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("try:\n  import ujson as json\nexcept ImportError:\n  import json\n\n"
                     "def dumps():\n  return json.dumps([])\n"),
  })
  closure = _closure(tree, "dumps")
  assert "Optional-import blocks preserved (1)" in render_text(closure)
  payload = json.loads(render_json(closure))
  assert payload["guarded_blocks"][0]["module"] == "pkg.mod"
