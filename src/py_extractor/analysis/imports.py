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
  """

  kind: str
  module: str
  level: int = 0
  names: list[ImportedName] = field(default_factory=list)
  target: str | None = None
  lineno: int = 0
  node: ast.AST | None = None

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


def collect_imports(tree: ast.Module,
                    current_module: str,
                    is_package: bool = False) -> list[ImportInfo]:
  """Walk *tree* and return every import statement (including nested ones).

  Nested imports (inside ``if``/``try``/``TYPE_CHECKING`` blocks or functions)
  are collected too, because they still bind names the extracted code may use.
  """
  results: list[ImportInfo] = []
  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        results.append(
            ImportInfo(
                kind="import",
                module=alias.name,
                target=alias.name,
                names=[ImportedName(alias.name, alias.asname)],
                lineno=node.lineno,
                node=node,
            ))
    elif isinstance(node, ast.ImportFrom):
      module = node.module or ""
      if node.level:
        try:
          module = resolve_relative(current_module, is_package, node.level, module)
        except ResolutionError:
          # A relative import that escapes the top-level package is invalid;
          # skip it rather than aborting analysis of an otherwise-fine file.
          continue
      names = [ImportedName(alias.name, alias.asname) for alias in node.names]
      results.append(
          ImportInfo(
              kind="from",
              module=module,
              level=node.level,
              names=names,
              lineno=node.lineno,
              node=node,
          ))
  return results


__all__ = ["ImportInfo", "ImportedName", "collect_imports", "resolve_relative"]
