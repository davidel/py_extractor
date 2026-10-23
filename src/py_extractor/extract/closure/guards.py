"""Preservation of optional-import guards (``try: import x`` / ``if cond:``).

Hoisting a guarded import out of its guard would turn an optional dependency
into a hard one, or silently drop a fallback branch. When every import inside a
top-level guarding statement can be kept as-is, the whole statement is copied
verbatim so the original fallback logic survives.

The helpers here are pure: they inspect the AST and return the blocks to keep
(plus the names they handle / rely on) and any flags to report, rather than
mutating builder state.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ...analysis.imports import GUARD_TYPES, ImportInfo, collect_imports
from ...analysis.resolver import ResolvedSymbol, resolve_import_name
from ...analysis.symbols import ModuleSymbols, collect_statement_bindings, free_names
from ...source.tree import SourceTree
from .models import Flag, FlagKind, GuardedBlock


@dataclass
class PreservedGuards:
  """The outcome of scanning a module for preservable optional-import guards.

  Attributes:
    handled: Names bound by preserved blocks (skip them in the dependency loop).
    extra: Additional free names those blocks rely on (e.g. ``sys`` in
      ``if sys.version_info:``), which must still be resolved and hoisted.
    blocks: The verbatim statements to emit, keyed by ``(module, lineno)``.
    flags: Warnings raised for guards that could not be preserved.
  """

  handled: set[str] = field(default_factory=set)
  extra: set[str] = field(default_factory=set)
  blocks: dict[tuple, GuardedBlock] = field(default_factory=dict)
  flags: list[Flag] = field(default_factory=list)


def preserve_guards(
    tree: SourceTree,
    resolved: ResolvedSymbol,
    module_ast: ast.Module,
    is_package: bool,
    dep_names: set[str],
    symbols: ModuleSymbols,
) -> PreservedGuards:
  """Copy optional-import blocks verbatim; report names handled and needed.

  When every import inside a top-level guarding statement can be kept as-is
  (absolute and not in-repo), the whole statement is copied verbatim; the names
  it binds are reported so they are neither hoisted nor flagged.
  """
  result = PreservedGuards()
  text = tree.get(resolved.path) or ""
  for stmt in module_ast.body:
    if not isinstance(stmt, GUARD_TYPES):
      continue
    imports = collect_imports(stmt, resolved.module, is_package)
    if not imports:
      continue
    bound = collect_statement_bindings([stmt])
    if not bound & dep_names:
      # The block binds nothing the extracted code needs: skip it entirely.
      continue
    if not _guard_is_preservable(tree, resolved, stmt, imports, symbols):
      # The guard cannot be copied as-is (it imports an in-repo/relative module,
      # or its condition needs a sibling definition that is inlined *after* the
      # block). The import is resolved normally, but the optional-import idiom
      # is lost, so record it for the report.
      result.flags.append(
          Flag(
              FlagKind.GUARDED_IMPORT,
              "optional-import guard could not be preserved (its import must be "
              "flattened); the dependency is now unconditional",
              f"{resolved.module}:{stmt.lineno}",
          ))
      continue
    source = ast.get_source_segment(text, stmt)
    if source is None:  # pragma: no cover - only if locations are missing
      continue
    key = (resolved.module, stmt.lineno)
    result.blocks.setdefault(key, GuardedBlock(source, resolved.module, stmt.lineno))
    result.handled |= bound
    result.extra |= free_names(stmt)
  result.extra -= result.handled
  return result


def _guard_is_preservable(tree: SourceTree, resolved: ResolvedSymbol, stmt: ast.stmt,
                          imports: list[ImportInfo], symbols: ModuleSymbols) -> bool:
  """Whether a guarding statement can be copied verbatim into the output.

  Two cases block preservation, both of which force the caller to fall back to
  resolving the import normally (and flagging it):

  * The guard imports a relative or in-repo module, which must be flattened
    rather than kept.
  * The guard's own condition (or other non-import statements) reference a
    module-level definition from the same file. That definition is inlined
    *after* the guarded block, so emitting the block verbatim would raise
    ``NameError``.
  """
  for imp in imports:
    if imp.level:
      return False
    for imported in imp.explicit_names:
      if resolve_import_name(tree, imp, imported.name, current_module=resolved.module) is not None:
        return False
  return all(not symbols.has(name) for name in free_names(stmt))


__all__ = ["PreservedGuards", "preserve_guards"]
