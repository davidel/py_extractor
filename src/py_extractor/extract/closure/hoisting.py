"""Import collection helpers for the closure walk.

These are pure functions (no builder state): they inspect a module AST and
return the data the closure builder needs -- the ``__future__`` imports to emit
first, the module-scope imports eligible for hoisting, the nested imports that
travel with their enclosing definition, and any flags raised along the way.
"""

from __future__ import annotations

import ast

from ...analysis.imports import ImportedName, ImportInfo, collect_imports
from ...analysis.resolver import ResolvedSymbol, resolve_import_name
from ...source.tree import SourceTree
from .models import Flag, FlagKind, KeptImport

# ``name -> (import statement, imported token)`` for a module's imports.
ImportBindingMap = dict[str, tuple[ImportInfo, ImportedName]]


def collect_future_imports(module_ast: ast.Module) -> dict[tuple, KeptImport]:
  """Return every ``from __future__ import ...`` in *module_ast*, keyed uniquely.

  Future statements must appear at the very top of the file, before any other
  statement. They bind no runtime names (so they never surface as free names)
  and would otherwise be dropped or sorted among the other imports; they are
  therefore collected separately and always emitted first.
  """
  future: dict[tuple, KeptImport] = {}
  for imp in collect_imports(module_ast, "__main__"):
    if imp.kind == "from" and imp.module == "__future__":
      for imported in imp.names:
        key = (imp.module, imported.name, imported.asname)
        future.setdefault(key, KeptImport("from", imp.module, imported.name, imported.asname))
  return future


def build_import_maps(module_ast: ast.Module, module: str,
                      is_package: bool) -> tuple[ImportBindingMap, ImportBindingMap, list[Flag]]:
  """Split imports into hoistable (module-scope) and local (nested) maps.

  Only *module-scope* imports are hoisted; imports inside a function or class
  body are bound in that local scope and are copied verbatim with the
  definition that contains them, so they are returned separately (as
  ``nested``) rather than added to the hoisting map.

  Returns:
    A triple ``(mapping, nested, flags)``: the hoisting map, the nested map and
    any flags raised (currently only unexpandable star imports).
  """
  mapping: ImportBindingMap = {}
  nested: ImportBindingMap = {}
  flags: list[Flag] = []
  for imp in collect_imports(module_ast, module, is_package):
    if imp.kind == "from" and imp.module == "__future__":
      # Future statements bind no runtime names and are emitted first by the
      # emitter (via ``collect_future_imports``); never hoist them here.
      continue
    if imp.scope == "nested":
      for imported in imp.explicit_names:
        nested.setdefault(imported.bound_name, (imp, imported))
      continue
    for imported in imp.explicit_names:
      mapping.setdefault(imported.bound_name, (imp, imported))
    if imp.is_star:
      flags.append(
          Flag(FlagKind.STAR_IMPORT, f"'from {imp.module} import *' cannot be expanded",
               f"{module}:{imp.lineno}"))
  return mapping, nested, flags


def needs_flattening(tree: SourceTree, resolved: ResolvedSymbol, imp: ImportInfo,
                     imported: ImportedName) -> bool:
  """Whether *imp* points at an in-repo (or relative) module to be inlined."""
  if imp.level:
    return True
  return resolve_import_name(tree, imp, imported.name, current_module=resolved.module) is not None


__all__ = ["build_import_maps", "collect_future_imports", "needs_flattening"]
