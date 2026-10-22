"""Integration tests against real repositories.

These require outbound network access and are deselected by default. Run with::

    pytest -m network

They are deliberately tolerant: GitHub layout changes should be caught, but the
assertions focus on invariants (valid Python output, target present) rather than
exact bytes.
"""

from __future__ import annotations

import ast

import pytest

from py_extractor import extract_symbol
from py_extractor.config import ExtractOptions, TrustPolicy

pytestmark = pytest.mark.network

# The original sketch referenced the (now deleted) fork ``w-e-w/DAT``; the
# upstream project is ``zhengchen1999/DAT`` (ICCV'23 Dual Aggregation Transformer).
DAT_REPO = "github:zhengchen1999/DAT"
DAT_FILE = "basicsr/archs/dat_arch.py"


def _options(tmp_path) -> ExtractOptions:
  return ExtractOptions(cache_dir=tmp_path / "cache",
                        trust=TrustPolicy.build(trusted={"torch", "numpy", "timm"}))


def test_extract_dat_from_github(tmp_path) -> None:
  """The reference example from the original sketch: DAT -> DAT."""
  result = extract_symbol(
      DAT_REPO,
      "DAT",
      file_hint=DAT_FILE,
      options=_options(tmp_path),
      fetch_third_party=False,
  )
  ast.parse(result.source)  # generated file must be valid Python
  assert "class DAT" in result.source
  assert result.target.symbol == "DAT"


def test_extract_small_pure_python_function(tmp_path) -> None:
  """A class from a small, well-known pure-python library."""
  result = extract_symbol(
      "github:psf/requests",
      "RequestEncodingMixin",
      options=_options(tmp_path),
      fetch_third_party=False,
  )
  ast.parse(result.source)


def test_output_never_drops_used_names(tmp_path) -> None:
  """Every module-level free name in the output is bound, imported or flagged."""
  import builtins

  from py_extractor.analysis.symbols import collect_module_symbols, free_names

  result = extract_symbol(
      DAT_REPO,
      "DAT",
      file_hint=DAT_FILE,
      options=_options(tmp_path),
      fetch_third_party=False,
  )
  module = ast.parse(result.source)

  # Scope-aware: this correctly excludes method parameters, `self`, locals, etc.
  referenced: set[str] = set()
  for stmt in module.body:
    referenced |= free_names(stmt)

  symbols = collect_module_symbols(module)
  # Kept imports bind either their alias (``import numpy as np``) or their name.
  imported: set[str] = set()
  for imp in result.closure.kept_imports:
    if imp.asname:
      imported.add(imp.asname)
    elif imp.kind == "import":
      imported.add(imp.module.split(".")[0])
    else:
      imported.add(imp.name)
  bound = set(symbols.defs) | set(symbols.assignments) | imported | set(vars(builtins))

  unresolved = referenced - bound
  assert not unresolved, f"unresolved free names in output: {sorted(unresolved)}"
  compile(result.source, "<extracted>", "exec")  # must actually compile


def test_generated_file_compiles_and_symbol_is_importable(tmp_path) -> None:
  """End-to-end: the written file passes compile_file and import_symbol."""
  from py_extractor.verify import compile_file, import_symbol

  result = extract_symbol(
      DAT_REPO,
      "DAT",
      file_hint=DAT_FILE,
      options=_options(tmp_path),
      fetch_third_party=False,
  )
  out = tmp_path / "dat_out.py"
  out.write_text(result.source, encoding="utf-8")

  assert compile_file(out).ok
  # torch/timm etc. are not installed in CI, so the import smoke test is
  # expected to fail gracefully (reported, never raising).
  outcome = import_symbol("DAT", path=out, timeout=60)
  assert outcome.kind == "import"
  if not outcome.ok:
    assert outcome.detail  # a readable reason is always provided
