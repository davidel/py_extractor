"""Transitive dependency closure.

Starting from a target symbol, walk its free names, pull in sibling module-level
definitions from the same file, resolve imports into the source tree, and
recurse. Trusted/stdlib imports are recorded (to be kept as imports); unresolved
third-party imports are flagged (and, when a resolver is supplied, inlined from
the foreign repository).
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from ..analysis.imports import ImportedName, ImportInfo, collect_imports
from ..analysis.resolver import ResolvedSymbol, resolve_import_name
from ..analysis.symbols import collect_module_symbols, find_definition, free_names
from ..config import ExtractOptions
from ..errors import ResolutionError, SymbolNotFoundError
from ..source.tree import SourceTree
from .classifier import Route, classify


class FlagKind(Enum):
  """Categories of things that could not be cleanly vendored."""

  UNRESOLVED_THIRD_PARTY = "unresolved_third_party"
  UNRESOLVED_NAME = "unresolved_name"
  STAR_IMPORT = "star_import"
  NAMESPACE_OBJECT = "namespace_object"
  COLLISION = "collision"


@dataclass
class Flag:
  """A single unresolved / risky item, surfaced in the report."""

  kind: FlagKind
  detail: str
  location: str = ""


@dataclass
class ExtractedUnit:
  """One inlined definition (or whole module) destined for the output."""

  key: str
  repo_id: str
  path: str
  module: str
  symbol: str | None
  node: ast.AST
  source: str


@dataclass(frozen=True)
class KeptImport:
  """An import that is kept as-is (stdlib or trusted)."""

  kind: str
  module: str
  name: str
  asname: str | None = None


@dataclass
class ClosureResult:
  """The full result of the closure computation."""

  target: ExtractedUnit
  units: list[ExtractedUnit]
  kept_imports: list[KeptImport]
  namespace_roots: dict[str, str] = field(default_factory=dict)
  flags: list[Flag] = field(default_factory=list)
  repo_trees: dict[str, SourceTree] = field(default_factory=dict)


# A resolver maps an unresolved third-party import root to a (repo_id, tree).
ThirdPartyResolver = Callable[[str, ExtractOptions], "tuple[str, SourceTree] | None"]


@dataclass
class _Context:
  repo_id: str
  tree: SourceTree


class _ClosureBuilder:
  """Accumulates the closure over potentially multiple source trees."""

  def __init__(
      self,
      root_repo_id: str,
      root_tree: SourceTree,
      options: ExtractOptions,
      third_party_resolver: ThirdPartyResolver | None,
  ) -> None:
    self.options = options
    self.policy = options.trust
    self.resolver = third_party_resolver
    self.units: dict[str, ExtractedUnit] = {}
    self.order: list[str] = []
    self.kept_imports: dict[tuple, KeptImport] = {}
    self.namespace_roots: dict[str, str] = {}
    self.flags: list[Flag] = []
    self.repo_trees: dict[str, SourceTree] = {root_repo_id: root_tree}
    self._trees: dict[str, SourceTree] = {root_repo_id: root_tree}
    self._foreign: dict[str, _Context | None] = {}
    self._edges: dict[str, set[str]] = {}
    self._whole_module_paths: set[tuple[str, str]] = set()
    self._collision_names: dict[str, str] = {}

  # -- public -------------------------------------------------------------

  def build(self, target: ResolvedSymbol, repo_id: str) -> ClosureResult:
    target_unit = self._process(repo_id, target)
    ordered = self._topological_order()
    return ClosureResult(
        target=target_unit,
        units=[self.units[key] for key in ordered],
        kept_imports=list(self.kept_imports.values()),
        namespace_roots=dict(self.namespace_roots),
        flags=self.flags,
        repo_trees=dict(self.repo_trees),
    )

  # -- core walk ----------------------------------------------------------

  @staticmethod
  def _key(repo_id: str, resolved: ResolvedSymbol) -> str:
    return f"{repo_id}::{resolved.path}::{resolved.symbol or '*'}"

  def _process(self, repo_id: str, resolved: ResolvedSymbol) -> ExtractedUnit:
    key = self._key(repo_id, resolved)
    if key in self.units:
      return self.units[key]

    tree = self._trees.get(repo_id)
    if tree is None:
      raise ResolutionError(f"Unknown repo context {repo_id!r}")
    text = tree.get(resolved.path)
    if text is None:
      raise ResolutionError(f"File disappeared from tree: {resolved.path}")
    module_ast = ast.parse(text)

    if resolved.symbol is None:
      # Whole-module target: inline every top-level definition and honour every
      # import as a dependency (imports don't surface as free names).
      node: ast.AST = module_ast
      self._whole_module_paths.add((repo_id, resolved.path))
      dep_names = self._module_free_names(module_ast, resolved.module)
      is_package = resolved.path.rsplit("/", 1)[-1] == "__init__.py"
      for imp in collect_imports(module_ast, resolved.module, is_package):
        dep_names.update(imported.bound_name for imported in imp.explicit_names)
    else:
      try:
        node = find_definition(module_ast, resolved.symbol)
      except SymbolNotFoundError:
        raise
      dep_names = free_names(node)

    unit = ExtractedUnit(
        key=key,
        repo_id=repo_id,
        path=resolved.path,
        module=resolved.module,
        symbol=resolved.symbol,
        node=node,
        source=text,
    )
    # Register before recursing so cycles terminate.
    self.units[key] = unit
    self.order.append(key)
    self._edges.setdefault(key, set())

    # Collision check for a whole-inlined top-level symbol name.
    if resolved.symbol is not None:
      clash = self._collision_names.get(resolved.symbol)
      if clash is not None and clash != key:
        self.flags.append(
            Flag(
                FlagKind.COLLISION,
                f"two definitions named {resolved.symbol!r}: {clash} and {key}",
                f"{resolved.module}:{resolved.symbol}",
            ))
      else:
        self._collision_names[resolved.symbol] = key

    self._resolve_dependencies(repo_id, tree, resolved, module_ast, dep_names, key)
    return unit

  def _module_free_names(self, module_ast: ast.Module, module: str) -> set[str]:
    names: set[str] = set()
    for stmt in module_ast.body:
      names |= free_names(stmt)
    return names

  def _resolve_dependencies(
      self,
      repo_id: str,
      tree: SourceTree,
      resolved: ResolvedSymbol,
      module_ast: ast.Module,
      dep_names: set[str],
      key: str,
  ) -> None:
    symbols = collect_module_symbols(module_ast)
    is_package = resolved.path.rsplit("/", 1)[-1] == "__init__.py"
    import_map = self._build_import_map(module_ast, resolved.module, is_package)

    for name in sorted(dep_names):
      # 1) A module-level definition in the same file -> inline it too.
      if symbols.has(name):
        if name == resolved.symbol:
          # A self-reference (e.g. ``super(C, self)``): already inlined, no edge.
          continue
        sibling = ResolvedSymbol(resolved.path, resolved.module, name, is_package)
        self._process(repo_id, sibling)
        self._edges[key].add(self._key(repo_id, sibling))
        continue

      # 2) A name bound by an import statement.
      entry = import_map.get(name)
      if entry is not None:
        imp, imported = entry
        self._handle_import(repo_id, tree, resolved, imp, imported, key)
        continue

      # 3) Dangling name: not a sibling, not imported, not builtin.
      self.flags.append(
          Flag(
              FlagKind.UNRESOLVED_NAME,
              f"name {name!r} is neither defined nor imported",
              f"{resolved.module}:{resolved.symbol or '<module>'}",
          ))

  def _build_import_map(self, module_ast: ast.Module, module: str,
                        is_package: bool) -> dict[str, tuple[ImportInfo, ImportedName]]:
    mapping: dict[str, tuple[ImportInfo, ImportedName]] = {}
    for imp in collect_imports(module_ast, module, is_package):
      for imported in imp.explicit_names:
        mapping.setdefault(imported.bound_name, (imp, imported))
      if imp.is_star:
        self.flags.append(
            Flag(FlagKind.STAR_IMPORT, f"'from {imp.module} import *' cannot be expanded",
                 f"{module}:{imp.lineno}"))
    return mapping

  # -- import handling ----------------------------------------------------

  def _handle_import(
      self,
      repo_id: str,
      tree: SourceTree,
      resolved: ResolvedSymbol,
      imp: ImportInfo,
      imported: ImportedName,
      key: str,
  ) -> None:
    root = imp.root
    decision = classify(root, self.policy)

    if decision.route in (Route.STDLIB, Route.TRUSTED):
      self._keep_import(imp, imported)
      return

    # Not stdlib/trusted: is it an in-repo module?
    internal = resolve_import_name(tree, imp, imported.name, current_module=resolved.module)
    if internal is not None:
      if internal.symbol is None:
        # A module referenced as a namespace -> remember the alias for flattening.
        self.namespace_roots[imported.bound_name] = internal.module
      self._process(repo_id, internal)
      self._edges[key].add(self._key(repo_id, internal))
      return

    # Try an external repository resolver (PyPI -> GitHub).
    foreign = self._resolve_foreign(root)
    if foreign is not None:
      foreign_id, foreign_tree = foreign
      internal_foreign = self._resolve_in_foreign(foreign_id, foreign_tree, imp, imported)
      if internal_foreign is not None:
        self._trees.setdefault(foreign_id, foreign_tree)
        if internal_foreign.symbol is None:
          self.namespace_roots[imported.bound_name] = internal_foreign.module
        self._process(foreign_id, internal_foreign)
        self._edges[key].add(self._key(foreign_id, internal_foreign))
        return

    # Keep as an import + flag (matches the "warn but still emit" policy).
    self.flags.append(
        Flag(
            FlagKind.UNRESOLVED_THIRD_PARTY,
            f"third-party '{root}' could not be inlined (kept as import)"
            if self.options.allow_third_party else f"third-party '{root}' denied (kept as import)",
            f"{resolved.module}:{resolved.symbol or '<module>'}",
        ))
    if decision.route is Route.UNKNOWN:
      self._keep_import(imp, imported)

  def _keep_import(self, imp: ImportInfo, imported: ImportedName) -> None:
    if imp.kind == "import":
      target = imp.target or imp.module
      record = KeptImport("import", target, target, imported.asname)
    else:
      record = KeptImport("from", imp.module, imported.name, imported.asname)
    key = (record.kind, record.module, record.name, record.asname)
    self.kept_imports.setdefault(key, record)

  def _resolve_foreign(self, root: str) -> tuple[str, SourceTree] | None:
    if root in self._foreign:
      cached = self._foreign[root]
      return None if cached is None else (cached.repo_id, cached.tree)
    if self.resolver is None:
      self._foreign[root] = None
      return None
    try:
      found = self.resolver(root, self.options)
    except Exception:  # noqa: BLE001 - resolver failures degrade to a flag
      found = None
    if found is None:
      self._foreign[root] = None
      return None
    repo_id, tree = found
    context = _Context(repo_id, tree)
    self._foreign[root] = context
    self.repo_trees.setdefault(repo_id, tree)
    return (repo_id, tree)

  def _resolve_in_foreign(
      self,
      repo_id: str,
      tree: SourceTree,
      imp: ImportInfo,
      imported: ImportedName,
  ) -> ResolvedSymbol | None:
    return resolve_import_name(tree, imp, imported.name, current_module=imp.module)

  # -- ordering -----------------------------------------------------------

  def _topological_order(self) -> list[str]:
    """Order units so that dependencies precede dependents (cycle-safe)."""
    visited: set[str] = set()
    order: list[str] = []

    def visit(key: str) -> None:
      if key in visited or key not in self.units:
        return
      visited.add(key)
      for dependency in sorted(self._edges.get(key, ())):
        visit(dependency)
      order.append(key)

    for key in self.order:
      visit(key)
    return order


def compute_closure(
    tree: SourceTree,
    target: ResolvedSymbol,
    options: ExtractOptions | None = None,
    *,
    repo_id: str = "root",
    third_party_resolver: ThirdPartyResolver | None = None,
) -> ClosureResult:
  """Compute the transitive closure for *target* within *tree*.

  Args:
    tree: The source tree containing the target.
    target: The resolved location of the target symbol.
    options: Extraction options (trust policy, third-party policy).
    repo_id: Identifier for the root repo (used to namespace unit keys).
    third_party_resolver: Optional callable to fetch foreign repositories.

  Returns:
    A :class:`ClosureResult` describing units, kept imports and flags.
  """
  options = options or ExtractOptions()
  builder = _ClosureBuilder(repo_id, tree, options, third_party_resolver)
  return builder.build(target, repo_id)


__all__ = [
    "ClosureResult",
    "ExtractedUnit",
    "Flag",
    "FlagKind",
    "KeptImport",
    "ThirdPartyResolver",
    "compute_closure",
]
