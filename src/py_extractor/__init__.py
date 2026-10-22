"""The high-level extraction API tying everything together."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .analysis.resolver import ResolvedSymbol
from .analysis.symbols import find_definition
from .config import ExtractOptions, TrustPolicy
from .errors import SourceError, SymbolNotFoundError
from .extract.closure import ClosureResult, ThirdPartyResolver, compute_closure
from .extract.emitter import emit
from .pypi import extract_github_repo, fetch_pypi_json, guess_distribution_names
from .source import load_tree, parse_repo
from .source.fetcher import Fetcher
from .source.repo import RepoRef
from .source.tree import SourceTree

__version__ = "0.1.0"


@dataclass
class ExtractionResult:
  """The outcome of a full extraction.

  Attributes:
    source: The generated single-file Python source.
    closure: The closure result backing it.
    target: The located target symbol.
  """

  source: str
  closure: ClosureResult
  target: ResolvedSymbol


def _module_of(tree: SourceTree, path: str) -> str:
  """Return the best dotted module name for *path* within *tree*."""
  candidates = tree._candidate_names(path)
  if not candidates:
    return ""
  candidates.sort(key=lambda pair: (-pair[0], len(pair[1]), pair[1]))
  return candidates[0][1]


def _is_package_file(path: str) -> bool:
  return path.rsplit("/", 1)[-1] == "__init__.py"


def locate_symbol(tree: SourceTree, symbol: str, file_hint: str | None = None) -> ResolvedSymbol:
  """Locate *symbol* within *tree*, returning its concrete location.

  Args:
    tree: The source tree to search.
    symbol: A class/function name (optionally dotted ``Class.method``).
    file_hint: Optional repo-relative path to restrict the search to.

  Returns:
    The resolved location.

  Raises:
    SymbolNotFoundError: If the symbol cannot be found.
  """
  head = symbol.split(".")[0]
  if file_hint:
    path = _resolve_hint(tree, file_hint)
    if path is None:
      raise SymbolNotFoundError(f"File {file_hint!r} not found in repository")
    text = tree.get(path) or ""
    try:
      find_definition(ast.parse(text), symbol)  # raises if missing
    except SyntaxError as exc:
      raise SymbolNotFoundError(f"{file_hint!r} is not valid Python: {exc}") from exc
    return ResolvedSymbol(path, _module_of(tree, path), symbol, _is_package_file(path))

  matches: list[ResolvedSymbol] = []
  for path in tree.paths():
    text = tree.get(path) or ""
    try:
      parsed = ast.parse(text)
    except SyntaxError:
      continue
    for node in parsed.body:
      if getattr(node, "name", None) == head:
        matches.append(ResolvedSymbol(path, _module_of(tree, path), symbol, _is_package_file(path)))
        break
  if not matches:
    raise SymbolNotFoundError(f"Symbol {symbol!r} not found anywhere in repository")
  if len(matches) > 1:
    preferred = [m for m in matches if m.module.split(".")[-1].lower() == head.lower()]
    if len(preferred) == 1:
      return preferred[0]
  return matches[0]


def _resolve_hint(tree: SourceTree, file_hint: str) -> str | None:
  """Resolve a user-supplied file hint to an existing path, if possible."""
  if tree.exists(file_hint):
    return file_hint
  if not file_hint.endswith(".py"):
    if tree.exists(file_hint + ".py"):
      return file_hint + ".py"
    resolved = tree.resolve_module(file_hint)
    if resolved is not None:
      return resolved
  return None


def make_third_party_resolver(fetcher: Fetcher) -> ThirdPartyResolver:
  """Build a resolver mapping a third-party root module to a fetched GitHub tree."""

  def resolver(root: str, options: ExtractOptions) -> tuple[str, SourceTree] | None:
    for dist in guess_distribution_names(root):
      try:
        payload = fetch_pypi_json(dist, fetcher)
      except SourceError:
        continue
      repo = extract_github_repo(payload.get("info", {}))
      if repo is None:
        continue
      repo_ref = RepoRef("github", repo.owner, repo.name, "main")
      try:
        tree = load_tree(repo_ref, options)
      except Exception:  # noqa: BLE001 - any failure degrades to "unresolved"
        continue
      if tree.has_module(root):
        return repo_ref.slug, tree
    return None

  return resolver


def extract_symbol(
    repo: str,
    symbol: str,
    *,
    file_hint: str | None = None,
    ref: str = "main",
    options: ExtractOptions | None = None,
    trusted: set[str] | frozenset[str] | None = None,
    untrusted: set[str] | frozenset[str] | None = None,
    registry_files: list[str | Path] | None = None,
    strict_trust: bool = False,
    offline: bool = False,
    allow_third_party: bool = True,
    cache_dir: str | Path = ".py_extractor_cache",
    fetch_third_party: bool = True,
) -> ExtractionResult:
  """Extract *symbol* from *repo* into a single-file source string.

  Args:
    repo: Repository spec (``github:owner/repo``, a URL, or a local path).
    symbol: Class/function name to extract.
    file_hint: Optional repo-relative file path containing the symbol.
    ref: Git ref (branch/tag/commit).
    options: Explicit extraction options (overrides the keyword arguments).
    trusted: Additional trusted top-level module names.
    untrusted: Trusted module names to revoke.
    registry_files: Extra TOML registry files to load.
    strict_trust: Ignore the built-in curated registry.
    offline: Disallow all network access.
    allow_third_party: Whether to attempt inlining third-party code.
    cache_dir: Cache directory for downloads.
    fetch_third_party: Whether to attempt PyPI->GitHub resolution at all.

  Returns:
    An :class:`ExtractionResult`.
  """
  if options is None:
    options = ExtractOptions(
        trust=TrustPolicy.build(trusted, untrusted, registry_files, strict_trust),
        allow_third_party=allow_third_party,
        cache_dir=Path(cache_dir),
        offline=offline,
    )

  repo_ref = parse_repo(repo, ref)
  root_tree = load_tree(repo_ref, options)
  target = locate_symbol(root_tree, symbol, file_hint=file_hint)

  # Honour the effective options (an explicit `options` object wins over the
  # individual keyword arguments, which keeps the CLI's `--no-third-party`
  # decision intact instead of silently reverting to the default).
  effective_third_party = options.allow_third_party and allow_third_party
  fetcher = Fetcher(options.cache_dir, offline=options.offline)
  resolver = make_third_party_resolver(fetcher) if (fetch_third_party and
                                                    effective_third_party) else None

  closure = compute_closure(
      root_tree,
      target,
      options,
      repo_id=repo_ref.slug,
      third_party_resolver=resolver,
  )
  return ExtractionResult(source=emit(closure), closure=closure, target=target)


__all__ = [
    "ExtractionResult",
    "__version__",
    "extract_symbol",
    "locate_symbol",
    "make_third_party_resolver",
]
