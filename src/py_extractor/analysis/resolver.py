"""Resolve an imported name to a concrete (path, symbol) inside the source tree.

This is what replaces the fragile string-join path guessing of the original
sketch: imports are resolved against the *real* file layout, following package
``__init__`` re-exports.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..errors import ResolutionError
from ..source.tree import SourceTree, is_package_file
from .imports import ImportInfo, collect_imports
from .symbols import collect_module_symbols

_MAX_REEXPORT_DEPTH = 12


@dataclass(frozen=True)
class ResolvedSymbol:
  """A concrete location of code inside the source tree.

  Attributes:
    path: The repo-relative file path hosting the definition.
    module: The dotted module name of that file.
    symbol: The top-level symbol to extract, or ``None`` to take the whole module.
    is_package: Whether the hosting file is an ``__init__.py``.
  """

  path: str
  module: str
  symbol: str | None
  is_package: bool


def resolve_import_name(
    tree: SourceTree,
    import_info: ImportInfo,
    name: str,
    *,
    current_module: str,
    _seen: set[str] | None = None,
) -> ResolvedSymbol | None:
  """Resolve a single imported *name* to a location in *tree*.

  Handles:
    * ``from pkg.sub import Name`` where ``Name`` is defined in ``pkg/sub.py``.
    * ``from pkg import sub`` where ``sub`` is a submodule.
    * ``from pkg import Name`` where ``Name`` is re-exported by ``pkg/__init__``.
    * ``import pkg.sub`` (whole-module target).

  Args:
    tree: The repository source tree.
    import_info: The parsed import statement.
    name: The specific imported name to resolve.
    current_module: Dotted name of the importing module (for context).
    _seen: Internal recursion guard for re-export chains (do not pass).

  Returns:
    A :class:`ResolvedSymbol`, or ``None`` if it is not an in-repo symbol.
  """
  if import_info.kind == "import":
    # ``import pkg.sub`` / ``import pkg.sub as s`` -> the target module itself.
    target = import_info.target or import_info.module
    if tree.has_module(target):
      path = tree.resolve_module(target)
      if path:
        return ResolvedSymbol(path, target, None, is_package_file(path))
    return None

  base_module = import_info.module
  # 1) Is the name itself a submodule (e.g. ``from pkg import core``)?
  submodule = f"{base_module}.{name}" if base_module else name
  if tree.has_module(submodule):
    path = tree.resolve_module(submodule)
    if path:
      return ResolvedSymbol(path, submodule, None, is_package_file(path))

  # 2) Resolve the base module file and look for the symbol there.
  if not base_module or not tree.has_module(base_module):
    return None
  path = tree.resolve_module(base_module)
  if path is None:
    return None
  seen = _seen if _seen is not None else set()
  key = f"{path}:{name}"
  if key in seen:
    # Re-export cycle: we are already trying to resolve this exact occurrence.
    return None
  seen.add(key)
  try:
    return _resolve_symbol_in_file(tree, path, base_module, name, depth=0, seen=seen)
  finally:
    seen.discard(key)


def _resolve_symbol_in_file(
    tree: SourceTree,
    path: str,
    module: str,
    name: str,
    *,
    depth: int,
    seen: set[str],
) -> ResolvedSymbol | None:
  """Find *name* in the file at *path*, following re-exports if necessary."""
  text = tree.get(path)
  if text is None:
    return None
  try:
    parsed = ast.parse(text)
  except SyntaxError:
    return None

  symbols = collect_module_symbols(parsed)
  if symbols.has(name):
    return ResolvedSymbol(path, module, name, is_package_file(path))

  # Not defined here: maybe re-exported via ``from .x import name`` / ``import x as name``.
  if depth >= _MAX_REEXPORT_DEPTH:
    raise ResolutionError(f"Re-export chain too deep for {name!r} in {path}")
  for imp in collect_imports(parsed, module, is_package_file(path)):
    for imported in imp.explicit_names:
      if imported.bound_name != name:
        continue
      submodule = f"{imp.module}.{imported.name}" if imp.kind == "from" and imp.module else None
      # Follow a re-exported submodule.
      if submodule and tree.has_module(submodule):
        sub_path = tree.resolve_module(submodule)
        if sub_path and f"{sub_path}:None" not in seen:
          return ResolvedSymbol(sub_path, submodule, None, is_package_file(sub_path))
      nested = resolve_import_name(tree, imp, imported.name, current_module=module, _seen=seen)
      if nested is not None and nested.path != path:
        key = f"{nested.path}:{nested.symbol}"
        if key in seen:
          continue
        return ResolvedSymbol(nested.path, nested.module, nested.symbol,
                              is_package_file(nested.path))
  return None


__all__ = ["ResolvedSymbol", "resolve_import_name"]
