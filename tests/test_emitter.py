"""Tests for the single-file emitter and output validity."""

from __future__ import annotations

import ast

from py_extractor.analysis.resolver import ResolvedSymbol
from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.extract.closure import compute_closure
from py_extractor.extract.emitter import emit


def _closure(tree, path, module, symbol):
  options = ExtractOptions(cache_dir=__import__("pathlib").Path("/tmp/x"), trust=TrustPolicy())
  target = ResolvedSymbol(path, module, symbol, path.endswith("__init__.py"))
  return compute_closure(tree, target, options, repo_id="demo/repo")


def test_output_is_valid_python(basic_tree) -> None:
  source = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"))
  ast.parse(source)  # must not raise


def test_output_contains_target_and_alias_imports(basic_tree) -> None:
  source = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"))
  assert "class Model" in source
  assert "import numpy as np" in source
  assert "from torch import nn" in source


def test_output_contains_inlined_sibling(basic_tree) -> None:
  source = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"))
  assert "SCALE = 2.0" in source


def test_output_has_provenance_markers(basic_tree) -> None:
  source = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"))
  assert "mypkg.utils" in source
  assert "Resolution report" in source


def test_report_can_be_disabled(basic_tree) -> None:
  source = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"), include_report=False)
  assert "Resolution report" not in source
  ast.parse(source)


def test_custom_header(basic_tree) -> None:
  source = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"), header="My banner")
  assert "# My banner" in source


def test_golden_output_stable(basic_tree) -> None:
  source1 = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"))
  source2 = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"))
  assert source1 == source2


def test_output_ends_with_single_newline(basic_tree) -> None:
  source = emit(_closure(basic_tree, "mypkg/core.py", "mypkg.core", "Model"))
  assert source.endswith("\n")
  assert not source.endswith("\n\n")
