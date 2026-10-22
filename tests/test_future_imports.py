"""``from __future__`` imports must survive and stay first in the output.

Future statements are special: they bind no runtime names (so they never
surface as free names) and the compiler requires them to appear before every
other statement. Hoisting them among the ordinary imports either dropped them
or produced a ``SyntaxError``. These tests execute/compile the generated output
to prove they are preserved and emitted first.
"""

from __future__ import annotations

import json
from pathlib import Path

from py_extractor.analysis.resolver import ResolvedSymbol
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.extract.closure import compute_closure
from py_extractor.extract.emitter import emit
from py_extractor.source.tree import SourceTree

CORE = "class Widget:\n  def __init__(self, *a):\n    pass\n"


def _closure(tree: SourceTree, symbol: str | None, path: str = "pkg/mod.py"):
  module = path[:-3].replace("/", ".")
  target = ResolvedSymbol(path, module, symbol, False)
  return compute_closure(tree,
                         target,
                         ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy()),
                         repo_id="demo/pkg")


def _first_code_line(source: str) -> str:
  return next(line for line in source.splitlines() if line and not line.startswith("#"))


def test_future_import_is_preserved_and_first() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/core.py":
          CORE,
      "pkg/mod.py": ("from __future__ import annotations\n"
                     "import os\n"
                     "from .core import Widget\n\n"
                     "def use() -> Widget:\n  return Widget(os.sep)\n"),
  })
  closure = _closure(tree, "use")
  assert [i.name for i in closure.future_imports] == ["annotations"]
  source = emit(closure, include_report=False)
  assert _first_code_line(source) == "from __future__ import annotations"
  compile(source, "<out>", "exec")


def test_future_import_not_duplicated_among_kept_imports() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("from __future__ import annotations\n"
                     "import os\n\n"
                     "def use():\n  return os.sep\n"),
  })
  closure = _closure(tree, "use")
  assert all(i.module != "__future__" for i in closure.kept_imports)
  source = emit(closure, include_report=False)
  assert source.count("from __future__ import annotations") == 1


def test_future_import_precedes_alphabetically_earlier_import() -> None:
  # A trusted import whose name sorts before '__future__' must still come after.
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("from __future__ import annotations\n"
                     "import numpy as np\n\n"
                     "def use():\n  return np\n"),
  })
  closure = _closure(tree, "use")
  source = emit(closure, include_report=False)
  assert _first_code_line(source) == "from __future__ import annotations"
  assert "import numpy as np" in source
  compile(source, "<out>", "exec")


def test_multiple_future_names_are_all_preserved() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("from __future__ import annotations, division\n"
                     "import os\n\n"
                     "def use():\n  return os.sep\n"),
  })
  closure = _closure(tree, "use")
  assert {i.name for i in closure.future_imports} == {"annotations", "division"}
  source = emit(closure, include_report=False)
  assert "from __future__ import annotations" in source
  assert "from __future__ import division" in source
  compile(source, "<out>", "exec")


def test_future_import_preserved_for_whole_module_target() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("from __future__ import annotations\n"
                     "import os\n\n"
                     "def use(x: int) -> str:\n  return os.sep\n"),
  })
  closure = _closure(tree, None)
  source = emit(closure, include_report=False)
  assert source.count("from __future__ import annotations") == 1
  assert _first_code_line(source) == "from __future__ import annotations"
  compile(source, "<out>", "exec")


def test_future_import_alone_still_compiles() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/core.py":
          CORE,
      "pkg/mod.py": ("from __future__ import annotations\n"
                     "from .core import Widget\n\n"
                     "def use() -> Widget:\n  return Widget()\n"),
  })
  closure = _closure(tree, "use")
  source = emit(closure, include_report=False)
  compile(source, "<out>", "exec")
  ns: dict = {"__name__": "x"}
  exec(compile(source, "<out>", "exec"), ns)  # noqa: S102 - intentional
  assert isinstance(ns["use"](), ns["Widget"])


def test_future_imports_render_in_reports() -> None:
  from py_extractor.report import render_json, render_text

  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/mod.py": ("from __future__ import annotations\n"
                     "import os\n\n"
                     "def use():\n  return os.sep\n"),
  })
  closure = _closure(tree, "use")
  assert "Future imports (1):" in render_text(closure)
  payload = json.loads(render_json(closure))
  assert payload["future_imports"][0]["name"] == "annotations"
