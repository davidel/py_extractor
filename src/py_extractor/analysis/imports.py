"""Import statement parsing: preserves aliases and resolves relative levels."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..errors import ResolutionError


@dataclass(frozen=True)
class ImportedName:
  """A single bound name from an import statement.

  Attributes:
    name: The imported attribute / module token (e.g. ``nn``, ``path``).
    asname: The local alias if any (e.g. ``np`` in ``import numpy as np``).
  """

  name: str
  asname: str | None = None

  @property
  def bound_name(self) -> str:
    """The name actually bound in the local namespace."""
    return self.asname or self.name.split(".")[0]


@dataclass
class ImportInfo:
  """A parsed import statement.

  Attributes:
    kind: ``"import"`` for ``import x`` or ``"from"`` for ``from x import y``.
    module: The absolute dotted module being imported from (relative resolved).
    level: The original relative level (0 for absolute imports).
    names: The imported names (empty for bare ``import x`` handled via ``target``).
    target: For ``import a.b as c`` the dotted module; ``None`` for from-imports.
    lineno: Source line number for reporting.
    node: The originating AST node.
    scope: ``"module"`` for a module-level import, ``"nested"`` for one inside a
      function/class body (which binds a local name and must not be hoisted).
    guard: For a module-level import inside a conditional statement, the
      enclosing statement (``try``/``if``/...); ``None`` otherwise.
  """

  kind: str
  module: str
  level: int = 0
  names: list[ImportedName] = field(default_factory=list)
  target: str | None = None
  lineno: int = 0
  node: ast.AST | None = None
  scope: str = "module"
  guard: ast.stmt | None = None

  @property
  def root(self) -> str:
    """The top-level module token (used for classification)."""
    source = self.target if self.kind == "import" else self.module
    return (source or "").split(".")[0]

  @property
  def is_star(self) -> bool:
    """Whether this is a ``from x import *`` statement."""
    return any(imported.name == "*" for imported in self.names)

  @property
  def explicit_names(self) -> list[ImportedName]:
    """Imported names excluding the ``*`` wildcard."""
    return [imported for imported in self.names if imported.name != "*"]


def resolve_relative(current_module: str, is_package: bool, level: int, module: str) -> str:
  """Resolve a relative import to an absolute dotted module name.

  Args:
    current_module: The dotted name of the module containing the import.
    is_package: Whether the current module is itself a package (``__init__``).
    level: The relative level (``1`` = ``.``, ``2`` = ``..`` ...).
    module: The (possibly empty) module part after the dots.

  Returns:
    The absolute dotted module name.

  Raises:
    ResolutionError: If the relative import rises above the top-level package.
  """
  base = current_module.split(".") if current_module else []
  if not is_package and base:
    base = base[:-1]
  drop = level - 1
  if drop > len(base):
    raise ResolutionError(f"Relative import level {level} escapes package in {current_module!r}")
  prefix = base[:len(base) - drop] if drop else base
  if module:
    return ".".join(prefix + module.split("."))
  return ".".join(prefix)


# Statements whose body is a conditional/optional context for an import: an
# import directly inside one of these at module level is *guarded* (it may not
# execute), so it must not be hoisted to an unconditional import.
GUARD_TYPES = (
    ast.If,
    ast.Try,
    ast.TryStar,
    ast.With,
    ast.AsyncWith,
    ast.For,
    ast.AsyncFor,
    ast.While,
)

# Statements that open a new (function/class) scope: imports inside them bind a
# local name and must stay where they are rather than be hoisted.
SCOPE_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _make_imports(node: ast.AST, current_module: str, is_package: bool, scope: str,
                  guard: ast.stmt | None) -> list[ImportInfo]:
  """Build the :class:`ImportInfo` list for *node* (one per alias).

  One record per bound name keeps ``target``/``module`` unambiguous for
  multi-alias statements such as ``import os, sys`` or ``from x import a, b``.
  """
  out: list[ImportInfo] = []
  if isinstance(node, ast.Import):
    for alias in node.names:
      out.append(
          ImportInfo(
              kind="import",
              module=alias.name,
              target=alias.name,
              names=[ImportedName(alias.name, alias.asname)],
              lineno=node.lineno,
              node=node,
              scope=scope,
              guard=guard,
          ))
    return out
  module = node.module or ""
  if node.level:
    try:
      module = resolve_relative(current_module, is_package, node.level, module)
    except ResolutionError:
      # A relative import that escapes the top-level package is invalid;
      # skip it rather than aborting analysis of an otherwise-fine file.
      return out
  names = [ImportedName(alias.name, alias.asname) for alias in node.names]
  out.append(
      ImportInfo(
          kind="from",
          module=module,
          level=node.level,
          names=names,
          lineno=node.lineno,
          node=node,
          scope=scope,
          guard=guard,
      ))
  return out


def _walk_imports(node: ast.AST, current_module: str, is_package: bool, scope: str,
                  guard: ast.stmt | None, results: list[ImportInfo]) -> None:
  """Recurse *node*'s statement children, recording imports with their context."""
  for child in ast.iter_child_nodes(node):
    if isinstance(child, (ast.Import, ast.ImportFrom)):
      results.extend(_make_imports(child, current_module, is_package, scope, guard))
      continue
    if isinstance(child, SCOPE_TYPES):
      # A local import stays with the definition; it is neither guarded nor
      # hoisted (the definition body is copied verbatim).
      _walk_imports(child, current_module, is_package, "nested", None, results)
      continue
    if isinstance(child, ast.stmt):
      # Entering a conditional/optional context makes any nested import guarded.
      # Only the outermost such statement is recorded as the guard.
      new_guard = guard if guard is not None else (
          child if isinstance(child, GUARD_TYPES) else None)
      _walk_imports(child, current_module, is_package, scope, new_guard, results)
      continue
    _walk_imports(child, current_module, is_package, scope, guard, results)


def collect_imports(tree: ast.AST,
                    current_module: str,
                    is_package: bool = False) -> list[ImportInfo]:
  """Walk *tree* and return every import statement (including nested ones).

  *tree* may be a whole :class:`ast.Module` or any sub-node (e.g. a single
  guarding ``try``/``if`` statement). Nested imports (inside
  ``if``/``try``/``TYPE_CHECKING`` blocks or functions) are collected too,
  because they still bind names the extracted code may use. Each result records
  the ``scope`` it lives in (``"module"`` or ``"nested"``) and, for
  module-level imports inside a conditional statement, the enclosing ``guard``
  so callers can preserve the optional-import idiom.
  """
  results: list[ImportInfo] = []
  _walk_imports(tree, current_module, is_package, "module", None, results)
  return results


__all__ = [
    "GUARD_TYPES", "SCOPE_TYPES", "ImportInfo", "ImportedName", "collect_imports",
    "resolve_relative"
]
