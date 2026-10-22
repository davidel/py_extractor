"""Tests for post-extraction verification (compile + out-of-process import)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from py_extractor.verify import (
    VerifyResult,
    compile_file,
    compile_source,
    compile_source_result,
    import_symbol,
)

# --- compile_source ---------------------------------------------------------


def test_compile_source_accepts_valid_code() -> None:
  compile_source("x = 1\nclass C:\n    pass\n")  # must not raise


def test_compile_source_catches_syntax_error() -> None:
  with pytest.raises(SyntaxError):
    compile_source("def f(:\n  pass\n")


@pytest.mark.parametrize("snippet", [
    "return 5\n",
    "break\n",
    "yield 5\n",
    "nonlocal x\n",
    "def f(a, a):\n  pass\n",
])
def test_compile_source_catches_errors_ast_parse_misses(snippet: str) -> None:
  """These all parse fine but must fail full compilation."""
  import ast

  ast.parse(snippet)  # ast.parse accepts them
  with pytest.raises(SyntaxError):
    compile_source(snippet)


def test_compile_source_result_ok() -> None:
  result = compile_source_result("x = 1\n")
  assert isinstance(result, VerifyResult)
  assert result.ok and result.kind == "compile"


def test_compile_source_result_failure() -> None:
  result = compile_source_result("return 1\n")
  assert not result.ok
  assert "not valid Python" in result.detail


# --- compile_file -----------------------------------------------------------


def test_compile_file_ok(tmp_path: Path) -> None:
  path = tmp_path / "good.py"
  path.write_text("class C:\n    pass\n", encoding="utf-8")
  result = compile_file(path)
  assert result.ok and result.kind == "compile"


def test_compile_file_failure(tmp_path: Path) -> None:
  path = tmp_path / "bad.py"
  path.write_text("return 1\n", encoding="utf-8")
  result = compile_file(path)
  assert not result.ok
  assert "SyntaxError" in result.detail or "outside function" in result.detail


def test_compile_file_missing(tmp_path: Path) -> None:
  result = compile_file(tmp_path / "nope.py")
  assert not result.ok


# --- import_symbol ----------------------------------------------------------

GOOD_SOURCE = textwrap.dedent("""
  class C:
      def f(self):
          return 1
  def top():
      return 2
""")


def test_import_symbol_present_class() -> None:
  result = import_symbol("C", source=GOOD_SOURCE)
  assert result.ok and result.kind == "import"


def test_import_symbol_present_function() -> None:
  assert import_symbol("top", source=GOOD_SOURCE).ok


def test_import_symbol_present_method() -> None:
  assert import_symbol("C.f", source=GOOD_SOURCE).ok


def test_import_symbol_absent() -> None:
  result = import_symbol("Nope", source=GOOD_SOURCE)
  assert not result.ok
  assert "not defined" in result.detail


def test_import_symbol_absent_method() -> None:
  result = import_symbol("C.missing", source=GOOD_SOURCE)
  assert not result.ok


def test_import_symbol_module_level_error() -> None:
  result = import_symbol("C", source="raise RuntimeError('boom')\nclass C:\n    pass\n")
  assert not result.ok
  assert "failed" in result.detail


def test_import_symbol_from_path(tmp_path: Path) -> None:
  path = tmp_path / "gen.py"
  path.write_text(GOOD_SOURCE, encoding="utf-8")
  assert import_symbol("C", path=path).ok


def test_import_symbol_requires_path_or_source() -> None:
  with pytest.raises(ValueError):
    import_symbol("C")


def test_import_symbol_timeout() -> None:
  slow = "import time\ntime.sleep(30)\nclass C:\n    pass\n"
  result = import_symbol("C", source=slow, timeout=1.0)
  assert not result.ok
  assert "timed out" in result.detail
