"""Output fidelity: comments, blank lines and formatting survive extraction.

``ast.unparse`` re-serialises a node and silently discards comments, blank
lines and the author's formatting.  The emitter instead copies the original
lines verbatim and only falls back to ``ast.unparse`` when the text cannot be
reproduced exactly (guarded by a reparse + AST-equality check).
"""

from __future__ import annotations

import ast
from pathlib import Path

from py_extractor.analysis.resolver import ResolvedSymbol
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.extract.closure.builder import compute_closure
from py_extractor.extract.emitter import emit
from py_extractor.source.tree import SourceTree

_COMMENTED = SourceTree.from_mapping({
    "pkg/__init__.py":
        "",
    "pkg/core.py": ("# module constant used below\n"
                    "DEFAULT = 3  # trailing note\n"
                    "\n"
                    "\n"
                    "@decorate\n"
                    "class Model:\n"
                    "  \"\"\"A model.\n"
                    "\n"
                    "  Longer description.\n"
                    "  \"\"\"\n"
                    "\n"
                    "  # class-level comment\n"
                    "  scale = DEFAULT\n"
                    "\n"
                    "  def run(self):\n"
                    "    # body comment\n"
                    "    return self.scale  # trailing\n"),
})


def _emit(tree: SourceTree, symbol: str | None, include_report: bool = False) -> str:
  options = ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy())
  target = ResolvedSymbol("pkg/core.py", "pkg.core", symbol, False)
  return emit(compute_closure(tree, target, options, repo_id="demo/repo"),
              include_report=include_report)


def test_comments_and_docstrings_are_preserved() -> None:
  source = _emit(_COMMENTED, "Model")
  for needle in [
      "# module constant used below",
      "DEFAULT = 3  # trailing note",
      "@decorate",
      "Longer description.",
      "# class-level comment",
      "# body comment",
      "return self.scale  # trailing",
  ]:
    assert needle in source, needle


def test_original_indentation_is_kept() -> None:
  source = _emit(_COMMENTED, "Model")
  assert "\n  # class-level comment\n" in source
  assert "\n    # body comment\n" in source
  # 2-space Google style from the source, not ast.unparse's 4-space default.
  assert "class Model:\n  " in source


def test_output_still_compiles() -> None:
  ast.parse(_emit(_COMMENTED, "Model"))


def test_whole_module_preserves_comments() -> None:
  source = _emit(_COMMENTED, None)
  assert "# module constant used below" in source
  assert "# class-level comment" in source
  ast.parse(source)


def test_dependency_constant_keeps_its_comment() -> None:
  source = _emit(_COMMENTED, "Model")
  # DEFAULT is inlined as a dependency and must carry its leading comment.
  assert "# module constant used below\nDEFAULT = 3  # trailing note" in source


def test_falls_back_to_unparse_when_dedent_would_corrupt_a_string() -> None:
  """A nested method whose docstring is indented can't be dedented losslessly.

  The guard must detect the corruption and fall back to ``ast.unparse`` (which
  drops comments but is always correct), so the output still compiles.
  """
  tree = SourceTree.from_mapping({
      "pkg/__init__.py":
          "",
      "pkg/core.py": ("class Model:\n"
                      "  def run(self):\n"
                      "    \"\"\"Doc.\n"
                      "\n"
                      "    Indented content.\n"
                      "    \"\"\"\n"
                      "    return 1\n"),
  })
  options = ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy())
  target = ResolvedSymbol("pkg/core.py", "pkg.core", "Model.run", False)
  source = emit(compute_closure(tree, target, options, repo_id="demo/repo"), include_report=False)
  ast.parse(source)
  assert "def run" in source


def test_leading_comment_of_previous_statement_is_not_stolen() -> None:
  tree = SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/core.py": ("A = 1  # for A\n"
                      "# for B\n"
                      "B = 2\n"),
  })
  options = ExtractOptions(cache_dir=Path("/tmp/x"), trust=TrustPolicy())
  target = ResolvedSymbol("pkg/core.py", "pkg.core", "B", False)
  source = emit(compute_closure(tree, target, options, repo_id="demo/repo"), include_report=False)
  assert "# for B" in source
  assert "# for A" not in source
