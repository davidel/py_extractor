"""Top-level symbol discovery and scope-aware free-name (dependency) analysis."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

from ..config import is_builtin_name
from ..errors import SymbolNotFoundError

_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_COMP_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


@dataclass
class ModuleSymbols:
  """Top-level definitions found in a module.

  Attributes:
    defs: Mapping of name -> definition node (functions, classes).
    assignments: Mapping of name -> assignment node (module-level constants).
    all_exports: The contents of ``__all__`` if declared.
  """

  defs: dict[str, ast.AST] = field(default_factory=dict)
  assignments: dict[str, ast.AST] = field(default_factory=dict)
  all_exports: list[str] = field(default_factory=list)

  def has(self, name: str) -> bool:
    """Return ``True`` if *name* is defined at module level."""
    return name in self.defs or name in self.assignments

  def get(self, name: str) -> ast.AST | None:
    """Return the defining node for *name*, or ``None``."""
    return self.defs.get(name) or self.assignments.get(name)


def collect_module_symbols(tree: ast.Module) -> ModuleSymbols:
  """Collect module-level definitions and ``__all__`` from *tree*."""
  symbols = ModuleSymbols()
  for node in tree.body:
    _record(node, symbols)
  return symbols


def _record(node: ast.AST, symbols: ModuleSymbols) -> None:
  if isinstance(node, _DEF_TYPES):
    symbols.defs[node.name] = node
  elif isinstance(node, ast.Assign):
    for target in node.targets:
      if isinstance(target, ast.Name):
        if target.id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
          for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
              symbols.all_exports.append(element.value)
        symbols.assignments[target.id] = node
  elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
    symbols.assignments[node.target.id] = node


def _member_names(node: ast.AST) -> list[str]:
  """Return the names bound directly inside *node* (defs and assignments)."""
  names: list[str] = []
  for child in getattr(node, "body", []):
    if isinstance(child, _DEF_TYPES):
      names.append(child.name)
    elif isinstance(child, ast.Assign):
      names.extend(t.id for t in child.targets if isinstance(t, ast.Name))
    elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
      names.append(child.target.id)
  return names


def _suggest(label: str, names: list[str], limit: int = 8) -> str:
  """Return an `` Available <label>: a, b, ...`` hint, or ``""`` when empty."""
  unique = sorted(set(names))
  if not unique:
    return ""
  shown = unique[:limit]
  more = len(unique) - len(shown)
  tail = f", ... (+{more} more)" if more else ""
  return f" Available {label}: {', '.join(shown)}{tail}."


def find_definition(tree: ast.Module, name: str) -> ast.AST:
  """Locate a top-level definition by *name* (optionally ``Class.method``).

  Args:
    tree: The parsed module.
    name: Either a top-level name, or a dotted ``Class.attr`` path.

  Returns:
    The defining AST node.

  Raises:
    SymbolNotFoundError: If the name cannot be located.
  """
  head, _, tail = name.partition(".")
  symbols = collect_module_symbols(tree)
  node = symbols.get(head)
  if node is None:
    hint = "" if tail else " Nested members must be referenced as 'Class.member'."
    raise SymbolNotFoundError(f"Symbol {head!r} not found at module level.{hint}" +
                              _suggest("top-level names",
                                       list(symbols.defs) + list(symbols.assignments)))
  if not tail:
    return node
  current: ast.AST = node
  for part in tail.split("."):
    body = getattr(current, "body", [])
    match = None
    for child in body:
      if isinstance(child, _DEF_TYPES) and child.name == part:
        match = child
        break
      if isinstance(child, ast.Assign):
        for target in child.targets:
          if isinstance(target, ast.Name) and target.id == part:
            match = child
            break
      if match is not None:
        break
    if match is None:
      raise SymbolNotFoundError(f"Symbol {part!r} not found inside {name!r}." +
                                _suggest("members", _member_names(current)))
    current = match
  return current


# ---------------------------------------------------------------------------
# Scope-aware free-variable analysis
# ---------------------------------------------------------------------------


def _arg_names(node: ast.AST) -> set[str]:
  """Return the parameter names of a function/lambda node."""
  args = getattr(node, "args", None)
  names: set[str] = set()
  if args is None:
    return names
  for group in (args.posonlyargs, args.args, args.kwonlyargs):
    names.update(arg.arg for arg in group)
  if args.vararg:
    names.add(args.vararg.arg)
  if args.kwarg:
    names.add(args.kwarg.arg)
  return names


def _block_bindings(stmts: list[ast.stmt]) -> set[str]:
  """Return names bound at the *immediate* scope of *stmts*.

  Nested function/class bodies and comprehension scopes are excluded (their
  bindings belong to child scopes), but nested def/class *names* are included
  because they bind in the enclosing scope.
  """
  bound: set[str] = set()

  def scan(node: ast.AST) -> None:
    for child in ast.iter_child_nodes(node):
      if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(child.name)
        continue
      if isinstance(child, (ast.Lambda, *_COMP_TYPES)):
        continue
      if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
        bound.add(child.id)
      elif isinstance(child, (ast.Import, ast.ImportFrom)):
        for alias in child.names:
          bound.add(alias.asname or alias.name.split(".")[0])
      elif isinstance(child, (ast.Global, ast.Nonlocal)):
        bound.update(child.names)
      elif isinstance(child, (ast.comprehension,)):
        continue
      scan(child)

  for stmt in stmts:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      # The def/class *name* binds here; its body is a child scope.
      bound.add(stmt.name)
      continue
    scan(stmt)
  return bound


def _comp_targets(comp: ast.AST) -> set[str]:
  """Return the names bound by a comprehension's ``for`` targets."""
  targets: set[str] = set()
  for generator in comp.generators:
    for node in ast.walk(generator.target):
      if isinstance(node, ast.Name):
        targets.add(node.id)
  return targets


def _walk_expr(node: ast.AST, visible: frozenset[str], free: set[str]) -> None:
  """Walk *node* recording free ``Name`` loads not present in *visible*.

  New scopes (functions, lambdas, classes, comprehensions) are entered when
  encountered as either a child *or* the root node.
  """
  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
    for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
      _walk_expr(default, visible, free)
    for decorator in node.decorator_list:
      _walk_expr(decorator, visible, free)
    _walk_function(node, visible, free)
    return

  if isinstance(node, ast.ClassDef):
    for decorator in node.decorator_list:
      _walk_expr(decorator, visible, free)
    for base in node.bases:
      _walk_expr(base, visible, free)
    for keyword in node.keywords:
      _walk_expr(keyword.value, visible, free)
    _walk_block(node.body, visible, free)
    return

  if isinstance(node, ast.Lambda):
    for default in node.args.defaults:
      _walk_expr(default, visible, free)
    for default in node.args.kw_defaults:
      if default is not None:
        _walk_expr(default, visible, free)
    inner = visible | frozenset(_arg_names(node))
    _walk_expr(node.body, inner, free)
    return

  if isinstance(node, _COMP_TYPES):
    inner = visible | frozenset(_comp_targets(node))
    for generator in node.generators:
      _walk_expr(generator.iter, visible, free)
      for condition in generator.ifs:
        _walk_expr(condition, inner, free)
    for element in _comp_elements(node):
      _walk_expr(element, inner, free)
    return

  if isinstance(node, ast.Name):
    if (isinstance(node.ctx, ast.Load) and node.id not in visible and not is_builtin_name(node.id)):
      free.add(node.id)
    return

  for child in ast.iter_child_nodes(node):
    _walk_expr(child, visible, free)


def _comp_elements(comp: ast.AST) -> list[ast.AST]:
  """Return the value element(s) of a comprehension."""
  if isinstance(comp, ast.DictComp):
    return [comp.key, comp.value]
  return [comp.elt]


def _walk_block(stmts: list[ast.stmt], outer: frozenset[str], free: set[str]) -> None:
  """Walk a statement block with its own scope layered on *outer*.

  Nested ``def``/``class`` statements are handed to :func:`_walk_expr`, which
  applies their own parameter/local scopes correctly (so class method bodies
  do not leak ``self`` / parameters as free names).
  """
  visible = outer | frozenset(_block_bindings(stmts))
  for stmt in stmts:
    _walk_expr(stmt, visible, free)


def _walk_function(node: ast.AST, outer: frozenset[str], free: set[str]) -> None:
  """Walk a function body with its parameters and locals layered on *outer*."""
  visible = outer | frozenset(_arg_names(node)) | frozenset(_block_bindings(node.body))
  for stmt in node.body:
    _walk_expr(stmt, visible, free)


def free_names(node: ast.AST) -> set[str]:
  """Return the free names *node* depends on, with proper scope handling.

  Excludes parameters, local variables, loop/comprehension variables, nested-def
  parameters and builtins; includes names used in decorators, base classes,
  annotations, defaults, f-strings and ``AnnAssign`` values.
  """
  free: set[str] = set()
  outer: frozenset[str] = frozenset()
  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
    for default in list(node.args.defaults) + [d for d in node.args.kw_defaults if d]:
      _walk_expr(default, outer, free)
    for decorator in node.decorator_list:
      _walk_expr(decorator, outer, free)
    _walk_function(node, outer, free)
  elif isinstance(node, ast.ClassDef):
    for decorator in node.decorator_list:
      _walk_expr(decorator, outer, free)
    for base in node.bases:
      _walk_expr(base, outer, free)
    for keyword in node.keywords:
      _walk_expr(keyword.value, outer, free)
    _walk_block(node.body, outer, free)
  else:
    _walk_expr(node, outer, free)
  return free


def collect_bound_names(node: ast.AST) -> set[str]:
  """Return the set of names bound by *node* (module-level convenience)."""
  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
    return frozenset(_arg_names(node)) | frozenset(_block_bindings(node.body))
  if isinstance(node, ast.ClassDef):
    return frozenset(_block_bindings(node.body))
  return frozenset()


def collect_statement_bindings(stmts: list[ast.stmt]) -> set[str]:
  """Return names bound at the immediate scope of *stmts*.

  Unlike :func:`collect_bound_names` this accepts arbitrary statements (e.g. a
  guarding ``try``/``if`` block) and returns the names they bind in the
  enclosing (module) scope, without descending into nested scopes.
  """
  return _block_bindings(stmts)


def collect_loaded_names(node: ast.AST) -> set[str]:
  """Return every non-builtin name loaded within *node* (scopeless)."""
  loaded = {
      child.id
      for child in ast.walk(node)
      if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
  }
  return {name for name in loaded if not is_builtin_name(name)}


def collect_dependency_names(node: ast.AST) -> set[str]:
  """Return the free names *node* depends on (scope-aware). See :func:`free_names`."""
  return free_names(node)


__all__ = [
    "ModuleSymbols",
    "collect_bound_names",
    "collect_dependency_names",
    "collect_loaded_names",
    "collect_module_symbols",
    "collect_statement_bindings",
    "find_definition",
    "free_names",
]
