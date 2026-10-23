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

from ..analysis.imports import GUARD_TYPES, ImportedName, ImportInfo, collect_imports
from ..analysis.resolver import ResolvedSymbol, resolve_import_name
from ..analysis.symbols import (
    ModuleSymbols,
    collect_module_symbols,
    collect_statement_bindings,
    find_definition,
    free_names,
)
from ..config import ExtractOptions
from ..errors import ResolutionError, SymbolNotFoundError
from ..licensing import LicenseInfo, detect_license
from ..source.tree import SourceTree


class FlagKind(Enum):
  """Categories of things that could not be cleanly vendored."""

  UNRESOLVED_THIRD_PARTY = "unresolved_third_party"
  UNRESOLVED_NAME = "unresolved_name"
  STAR_IMPORT = "star_import"
  COLLISION = "collision"
  GUARDED_IMPORT = "guarded_import"
  NESTED_IMPORT = "nested_import"


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

  def render(self) -> str:
    """Return the ``import ...`` / ``from ... import ...`` source line."""
    if self.kind == "import":
      if self.asname and self.asname != self.module.split(".")[0]:
        return f"import {self.module} as {self.asname}"
      return f"import {self.module}"
    if self.asname:
      return f"from {self.module} import {self.name} as {self.asname}"
    return f"from {self.module} import {self.name}"


@dataclass(frozen=True)
class Binding:
  """A local name the extracted code needs, bound to inlined definitions.

  Importing a name into a file and then inlining the *definition* leaves the
  call site dangling (``from .core import Widget as W`` -> ``W()`` fails with
  ``NameError`` once ``Widget`` is flattened). A binding re-creates the missing
  local name so the flattened code still resolves.

  Attributes:
    kind: ``"alias"`` (``W = Widget``) or ``"module"`` (a namespace object
      exposing a module's inlined members).
    local: The name used at the call site (``W``, ``core``, ``pkg``).
    target: For ``"alias"``, the inlined top-level name to alias.
    unit_key: The unit this binding must be emitted after (the definition it
      depends on), so ordering matches dependency order.
    parts: For ``"module"``, the dotted path under *local* (``("core",)`` for
      ``import pkg.core``); empty when *local* names the module directly.
    members: For ``"module"``, the inlined top-level names to expose.
  """

  kind: str
  local: str
  target: str | None = None
  unit_key: str | None = None
  parts: tuple[str, ...] = ()
  members: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardedBlock:
  """A verbatim module-level statement preserved to keep an optional import.

  Hoisting ``import ujson as json`` out of a ``try: ... except ImportError:``
  block turns an *optional* dependency into a hard one (or, when the fallback
  binds the same name, silently drops it). Instead the whole guarding statement
  is copied verbatim so the original fallback logic is preserved.

  Attributes:
    source: The unparsed source of the guarding statement.
    module: The dotted module the block came from (for provenance).
    lineno: Line number in the origin file.
  """

  source: str
  module: str
  lineno: int


@dataclass
class ClosureResult:
  """The full result of the closure computation."""

  target: ExtractedUnit
  units: list[ExtractedUnit]
  kept_imports: list[KeptImport]
  bindings: list[Binding] = field(default_factory=list)
  guarded_blocks: list[GuardedBlock] = field(default_factory=list)
  future_imports: list[KeptImport] = field(default_factory=list)
  flags: list[Flag] = field(default_factory=list)
  licenses: list[LicenseInfo] = field(default_factory=list)
  license_ids: dict[str, str] = field(default_factory=dict)


# A resolver maps an unresolved third-party import root to a (repo_id, tree).
ThirdPartyResolver = Callable[[str, ExtractOptions], "tuple[str, SourceTree] | None"]


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
    self.bindings: dict[tuple, Binding] = {}
    self._binding_locals: dict[str, tuple] = {}
    self.guarded_blocks: dict[tuple, GuardedBlock] = {}
    self.future_imports: dict[tuple, KeptImport] = {}
    self.flags: list[Flag] = []
    self._trees: dict[str, SourceTree] = {root_repo_id: root_tree}
    self._foreign: dict[str, tuple[str, SourceTree] | None] = {}
    self._edges: dict[str, set[str]] = {}
    self._collision_names: dict[str, str] = {}

  # -- public -------------------------------------------------------------

  def build(self, target: ResolvedSymbol, repo_id: str) -> ClosureResult:
    target_unit = self._process(repo_id, target)
    ordered = self._topological_order()
    units = [self.units[key] for key in ordered]
    licenses, license_ids = self._collect_licenses(units)
    return ClosureResult(
        target=target_unit,
        units=units,
        kept_imports=list(self.kept_imports.values()),
        bindings=list(self.bindings.values()),
        guarded_blocks=list(self.guarded_blocks.values()),
        future_imports=list(self.future_imports.values()),
        flags=self.flags,
        licenses=licenses,
        license_ids=license_ids,
    )

  def _collect_licenses(self,
                        units: list[ExtractedUnit]) -> tuple[list[LicenseInfo], dict[str, str]]:
    """Assign a stable ``[L#]`` id to each contributing repository.

    Repositories are numbered in the order they first appear in the emitted
    (topologically ordered) units, so the ids are deterministic for a given
    extraction.
    """
    licenses: list[LicenseInfo] = []
    license_ids: dict[str, str] = {}
    for unit in units:
      repo_id = unit.repo_id
      if repo_id in license_ids:
        continue
      tree = self._trees.get(repo_id)
      if tree is None:
        continue
      license_ids[repo_id] = f"L{len(licenses) + 1}"
      licenses.append(detect_license(tree, repo_id))
    return licenses, license_ids

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
      dep_names = self._module_free_names(module_ast)
      is_package = resolved.path.rsplit("/", 1)[-1] == "__init__.py"
      for imp in collect_imports(module_ast, resolved.module, is_package):
        if imp.kind == "from" and imp.module == "__future__":
          continue
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

  def _module_free_names(self, module_ast: ast.Module) -> set[str]:
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
    self._collect_future_imports(module_ast)
    import_map, nested = self._build_import_map(module_ast, resolved.module, is_package)
    handled, extra = self._preserve_guards(tree, resolved, module_ast, is_package, dep_names,
                                           symbols)
    dep_names |= extra

    for name in sorted(dep_names):
      # 0) A name bound by a preserved optional-import block: already handled.
      if name in handled:
        continue

      # 1) A module-level definition in the same file -> inline it too.
      if symbols.has(name):
        if name == resolved.symbol:
          # A self-reference (e.g. ``super(C, self)``): already inlined, no edge.
          continue
        sibling = ResolvedSymbol(resolved.path, resolved.module, name, is_package)
        self._process(repo_id, sibling)
        self._edges[key].add(self._key(repo_id, sibling))
        continue

      # 2) A name bound by a module-level import statement.
      entry = import_map.get(name)
      if entry is not None:
        imp, imported = entry
        self._handle_import(repo_id, tree, resolved, imp, imported, key)
        continue

      # 3) A name bound by a function/class-local import: it stays with the
      #    copied definition body. A local import of an in-repo/relative module
      #    cannot be flattened there, so warn that it will not resolve.
      nested_entry = nested.get(name)
      if nested_entry is not None:
        imp, imported = nested_entry
        if self._needs_flattening(tree, resolved, imp, imported):
          self.flags.append(
              Flag(
                  FlagKind.NESTED_IMPORT,
                  f"import of {imp.module!r} inside a nested scope cannot be "
                  f"flattened; {name!r} may not resolve",
                  f"{resolved.module}:{imp.lineno}",
              ))
        continue

      # 4) Dangling name: not a sibling, not imported, not builtin.
      self.flags.append(
          Flag(
              FlagKind.UNRESOLVED_NAME,
              f"name {name!r} is neither defined nor imported",
              f"{resolved.module}:{resolved.symbol or '<module>'}",
          ))

  def _needs_flattening(self, tree: SourceTree, resolved: ResolvedSymbol, imp: ImportInfo,
                        imported: ImportedName) -> bool:
    """Whether *imp* points at an in-repo (or relative) module to be inlined."""
    if imp.level:
      return True
    return resolve_import_name(tree, imp, imported.name, current_module=resolved.module) is not None

  def _collect_future_imports(self, module_ast: ast.Module) -> None:
    """Record every ``from __future__ import ...`` for the emitter.

    Future statements must appear at the very top of the file, before any other
    statement. They bind no runtime names (so they never surface as free names)
    and would otherwise be dropped or sorted among the other imports; they are
    therefore collected separately and always emitted first.
    """
    for imp in collect_imports(module_ast, "__main__"):
      if imp.kind == "from" and imp.module == "__future__":
        for imported in imp.names:
          key = (imp.module, imported.name, imported.asname)
          self.future_imports.setdefault(
              key, KeptImport("from", imp.module, imported.name, imported.asname))

  def _preserve_guards(
      self,
      tree: SourceTree,
      resolved: ResolvedSymbol,
      module_ast: ast.Module,
      is_package: bool,
      dep_names: set[str],
      symbols: ModuleSymbols,
  ) -> tuple[set[str], set[str]]:
    """Copy optional-import blocks verbatim; return names handled and needed.

    Hoisting a guarded import (``try: import x`` / ``if cond: import x``) out of
    its guard would turn an optional dependency into a hard one, or silently
    drop a fallback branch. When every import inside a top-level guarding
    statement can be kept as-is (absolute and not in-repo), the whole statement
    is copied verbatim; the names it binds are reported so they are neither
    hoisted nor flagged.

    Returns:
      A pair ``(handled, extra)``: the names bound by preserved blocks (skip
      them in the dependency loop) and the additional free names those blocks
      rely on (e.g. ``sys`` in ``if sys.version_info:``), which must still be
      resolved and hoisted.
    """
    handled: set[str] = set()
    extra: set[str] = set()
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
      if not self._guard_is_preservable(tree, resolved, stmt, imports, symbols):
        # The guard cannot be copied as-is (it imports an in-repo/relative
        # module, or its condition needs a sibling definition that is inlined
        # *after* the block). The import is resolved normally, but the
        # optional-import idiom is lost, so record it for the report.
        self.flags.append(
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
      self.guarded_blocks.setdefault(key, GuardedBlock(source, resolved.module, stmt.lineno))
      handled |= bound
      extra |= free_names(stmt)
    return handled, extra - handled

  def _guard_is_preservable(self, tree: SourceTree, resolved: ResolvedSymbol, stmt: ast.stmt,
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
        if resolve_import_name(tree, imp, imported.name,
                               current_module=resolved.module) is not None:
          return False
    return all(not symbols.has(name) for name in free_names(stmt))

  def _build_import_map(
      self, module_ast: ast.Module, module: str, is_package: bool
  ) -> tuple[dict[str, tuple[ImportInfo, ImportedName]], dict[str, tuple[ImportInfo,
                                                                         ImportedName]]]:
    """Split imports into hoistable (module-scope) and local (nested) maps.

    Only *module-scope* imports are hoisted; imports inside a function or class
    body are bound in that local scope and are copied verbatim with the
    definition that contains them, so they are returned separately (as
    ``nested``) rather than added to the hoisting map.
    """
    mapping: dict[str, tuple[ImportInfo, ImportedName]] = {}
    nested: dict[str, tuple[ImportInfo, ImportedName]] = {}
    for imp in collect_imports(module_ast, module, is_package):
      if imp.kind == "from" and imp.module == "__future__":
        # Future statements bind no runtime names and are emitted first by the
        # emitter (via ``_collect_future_imports``); never hoist them here.
        continue
      if imp.scope == "nested":
        for imported in imp.explicit_names:
          nested.setdefault(imported.bound_name, (imp, imported))
        continue
      for imported in imp.explicit_names:
        mapping.setdefault(imported.bound_name, (imp, imported))
      if imp.is_star:
        self.flags.append(
            Flag(FlagKind.STAR_IMPORT, f"'from {imp.module} import *' cannot be expanded",
                 f"{module}:{imp.lineno}"))
    return mapping, nested

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

    # stdlib/trusted: keep as an import and never descend into it. Whitelisting
    # a package means "it and its dependencies are installed", so the closure
    # engine must not inline it (or any of its submodules).
    if self.policy.is_trusted(root):
      self._keep_import(imp, imported)
      return

    # Is it an in-repo module?
    internal = resolve_import_name(tree, imp, imported.name, current_module=resolved.module)
    if internal is not None:
      self._process(repo_id, internal)
      self._edges[key].add(self._key(repo_id, internal))
      self._record_binding(repo_id, internal, imported)
      return

    # Try an external repository resolver (PyPI -> GitHub).
    foreign = self._resolve_foreign(root)
    if foreign is not None:
      foreign_id, foreign_tree = foreign
      internal_foreign = resolve_import_name(foreign_tree,
                                             imp,
                                             imported.name,
                                             current_module=imp.module)
      if internal_foreign is not None:
        self._trees.setdefault(foreign_id, foreign_tree)
        self._process(foreign_id, internal_foreign)
        self._edges[key].add(self._key(foreign_id, internal_foreign))
        self._record_binding(foreign_id, internal_foreign, imported)
        return

    # Keep as an import + flag (matches the "warn but still emit" policy).
    self.flags.append(
        Flag(
            FlagKind.UNRESOLVED_THIRD_PARTY,
            f"third-party '{root}' could not be inlined (kept as import)"
            if self.options.allow_third_party else f"third-party '{root}' denied (kept as import)",
            f"{resolved.module}:{resolved.symbol or '<module>'}",
        ))
    self._keep_import(imp, imported)

  def _keep_import(self, imp: ImportInfo, imported: ImportedName) -> None:
    if imp.kind == "import":
      target = imp.target or imp.module
      record = KeptImport("import", target, target, imported.asname)
    else:
      record = KeptImport("from", imp.module, imported.name, imported.asname)
    key = (record.kind, record.module, record.name, record.asname)
    self.kept_imports.setdefault(key, record)

  # -- local bindings -----------------------------------------------------

  def _add_binding(self, binding: Binding, dedupe_key: tuple) -> None:
    """Store *binding*, flagging a local name bound to two different targets."""
    existing = self.bindings.get(dedupe_key)
    if existing is not None:
      return
    clash = self._binding_locals.get(binding.local)
    if clash is not None and clash != dedupe_key:
      self.flags.append(
          Flag(
              FlagKind.COLLISION,
              f"local name {binding.local!r} is bound to two different inlined "
              f"definitions; only one is emitted",
              f"binding {binding.local}",
          ))
      return
    self._binding_locals[binding.local] = dedupe_key
    self.bindings[dedupe_key] = binding

  def _record_binding(self, repo_id: str, internal: ResolvedSymbol, imported: ImportedName) -> None:
    """Record the local name the call site needs for an inlined definition.

    Resolving and inlining the *definition* is not enough: the importing file
    bound it under a local name (an alias, or a module object). Without a
    binding the flattened code raises ``NameError`` at the call site.
    """
    unit_key = self._key(repo_id, internal)
    if internal.symbol is not None:
      # ``from .core import Widget as W`` -> ``W = Widget`` (only when aliased;
      # a bare ``from .core import Widget`` already binds ``Widget``).
      if imported.asname:
        binding = Binding("alias", imported.asname, target=internal.symbol, unit_key=unit_key)
        self._add_binding(binding, ("alias", binding.local, binding.target))
      return

    # Module-object binding: ``from . import core`` / ``import pkg.core``.
    members = self._module_members(repo_id, internal)
    if not members:
      return
    if imp_target := (imported.asname or None):
      # ``import pkg.core as c`` / ``from . import core as c`` -> ``c``.
      local, parts = imp_target, ()
    elif "." in imported.name:
      # ``import pkg.core`` binds ``pkg``; access is ``pkg.core.Widget``.
      local, parts = imported.name.split(".")[0], tuple(imported.name.split(".")[1:])
    else:
      # ``from . import core`` -> ``core``.
      local, parts = imported.name, ()
    binding = Binding("module", local, unit_key=unit_key, parts=parts, members=members)
    self._add_binding(binding, ("module", binding.local, binding.parts))

  def _module_members(self, repo_id: str, internal: ResolvedSymbol) -> tuple[str, ...]:
    """Return the inlined top-level names exposed by a whole-module unit."""
    unit = self.units.get(self._key(repo_id, internal))
    if unit is None or not isinstance(unit.node, ast.Module):
      return ()
    symbols = collect_module_symbols(unit.node)
    names = set(symbols.defs) | set(symbols.assignments)
    names.discard("__all__")
    return tuple(sorted(names))

  def _resolve_foreign(self, root: str) -> tuple[str, SourceTree] | None:
    if root in self._foreign:
      return self._foreign[root]
    if self.resolver is None:
      self._foreign[root] = None
      return None
    try:
      found = self.resolver(root, self.options)
    except Exception:  # noqa: BLE001 - resolver failures degrade to a flag
      found = None
    self._foreign[root] = found
    return found

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
    "Binding",
    "ClosureResult",
    "ExtractedUnit",
    "Flag",
    "FlagKind",
    "GuardedBlock",
    "KeptImport",
    "ThirdPartyResolver",
    "compute_closure",
]
