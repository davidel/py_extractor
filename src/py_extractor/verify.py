"""Post-extraction verification helpers.

Two levels of assurance are provided:

* **Compile verification** -- :func:`compile_source` / :func:`compile_file` run
  the *full* CPython compiler (``compile()`` / ``py_compile``), which is strictly
  stronger than ``ast.parse`` and catches errors such as a stray top-level
  ``return``/``break``/``yield`` or a duplicate argument.

* **Import smoke test** -- :func:`import_symbol` executes the generated file in
  a fresh, isolated *subprocess* and reports whether the requested symbol can
  actually be materialised. Running out-of-process means untrusted module-level
  code cannot affect the current interpreter. This is opt-in because importing
  arbitrary code has side effects and may require the target's dependencies.
"""

from __future__ import annotations

import json
import py_compile
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_RESULT_MARKER = "__PY_EXTRACTOR_RESULT__"

_RUNNER = f"""
import json, runpy, sys
path, symbol = sys.argv[1], sys.argv[2]
namespace = runpy.run_path(path)
head, _, tail = symbol.partition(".")
present = head in namespace
if present and tail:
    obj = namespace.get(head)
    for part in tail.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            break
    present = obj is not None
print({_RESULT_MARKER!r} + json.dumps({{"present": bool(present)}}))
"""


@dataclass
class VerifyResult:
  """Outcome of a verification step.

  Attributes:
    ok: Whether the check passed.
    kind: The check performed (``"compile"`` or ``"import"``).
    detail: Human-readable explanation (error text or a short summary).
  """

  ok: bool
  kind: str
  detail: str = ""


def compile_source(source: str, filename: str = "<py_extractor output>") -> None:
  """Compile *source* with the full compiler, raising ``SyntaxError`` on failure.

  Unlike ``ast.parse`` this performs symbol-table and flow validation, catching
  errors such as top-level ``return``/``break``/``yield``.
  """
  compile(source, filename, "exec")


def compile_source_result(source: str, filename: str = "<py_extractor output>") -> VerifyResult:
  """Compile *source* and return a :class:`VerifyResult` instead of raising."""
  try:
    compile_source(source, filename)
  except SyntaxError as exc:
    return VerifyResult(False, "compile", f"not valid Python: {exc}")
  return VerifyResult(True, "compile", "source compiled successfully")


def compile_file(path: str | Path) -> VerifyResult:
  """Byte-compile a file on disk, returning a :class:`VerifyResult`.

  Never raises for a compilation error; the failure is reported in the result.
  """
  path = Path(path)
  try:
    py_compile.compile(str(path), doraise=True)
  except py_compile.PyCompileError as exc:
    return VerifyResult(False, "compile", str(exc))
  except OSError as exc:
    return VerifyResult(False, "compile", f"could not read {path}: {exc}")
  return VerifyResult(True, "compile", f"{path} compiled successfully")


def import_symbol(
    symbol: str,
    *,
    path: str | Path | None = None,
    source: str | None = None,
    timeout: float = 60.0,
) -> VerifyResult:
  """Execute the generated code in a subprocess and check *symbol* is usable.

  Args:
    symbol: The target symbol name (optionally dotted ``Class.method``).
    path: Path to an already-written generated file.
    source: Alternatively, the generated source (written to a temp file first).
    timeout: Subprocess timeout in seconds.

  Returns:
    A :class:`VerifyResult`. ``ok`` is ``True`` only when the module imported
    cleanly *and* *symbol* was materialised.
  """
  temp_dir: tempfile.TemporaryDirectory | None = None
  if path is None:
    if source is None:
      raise ValueError("import_symbol requires either 'path' or 'source'")
    temp_dir = tempfile.TemporaryDirectory(prefix="py_extractor_verify_")
    path = Path(temp_dir.name) / "extracted.py"
    Path(path).write_text(source, encoding="utf-8")

  try:
    try:
      completed = subprocess.run(
          [sys.executable, "-c", _RUNNER, str(path), symbol],
          capture_output=True,
          text=True,
          timeout=timeout,
      )
    except subprocess.TimeoutExpired:
      return VerifyResult(False, "import", f"timed out after {timeout:g}s")

    if completed.returncode != 0:
      detail = (completed.stderr or completed.stdout or "").strip()
      return VerifyResult(False, "import", f"importing the module failed:\n{detail}")

    for line in reversed(completed.stdout.splitlines()):
      if line.startswith(_RESULT_MARKER):
        payload = json.loads(line[len(_RESULT_MARKER):])
        if payload.get("present"):
          return VerifyResult(True, "import", f"symbol {symbol!r} is importable")
        return VerifyResult(False, "import", f"symbol {symbol!r} was not defined on import")
    return VerifyResult(False, "import", "verification runner produced no result")
  finally:
    if temp_dir is not None:
      temp_dir.cleanup()


__all__ = [
    "VerifyResult",
    "compile_file",
    "compile_source",
    "compile_source_result",
    "import_symbol",
]
