"""The high-level extraction API tying everything together."""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .analysis.resolver import ResolvedSymbol
from .analysis.symbols import find_definition
from .config import ExtractOptions, TrustPolicy
from .errors import AmbiguousSymbolError, SourceError, SymbolNotFoundError
from .extract.closure.builder import compute_closure
from .extract.closure.models import ClosureResult, ThirdPartyResolver
from .extract.emitter import emit
from .licensing import LicenseInfo, pypi_license
from .pypi import extract_github_repo, fetch_pypi_json, guess_distribution_names
from .source import load_tree, parse_repo
from .source.fetcher import Fetcher
from .source.repo import RepoRef
from .source.tree import SourceTree, is_package_file
from .verify import VerifyResult, compile_file, compile_source, import_symbol

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

  @property
  def licenses(self) -> list[LicenseInfo]:
    """License attribution for every contributing repository."""
    return self.closure.licenses


def locate_symbol(
    tree: SourceTree,
    symbol: str,
    file_hint: str | None = None,
    *,
    choose: Callable[[str, list[str]], str | None] | None = None,
) -> ResolvedSymbol:
  """Locate *symbol* within *tree*, returning its concrete location.

  Args:
    tree: The source tree to search.
    symbol: A class/function name (optionally dotted ``Class.method``).
    file_hint: Optional repo-relative path to use as the reference file. This
      overrides the repository-wide search entirely (useful for scripts/CI).
    choose: Optional callback invoked when the symbol is defined in more than
      one file. It receives the symbol and the ranked candidate paths and must
      return the chosen path (or ``None`` to decline).

  Returns:
    The resolved location.

  Raises:
    SymbolNotFoundError: If the symbol cannot be found.
    AmbiguousSymbolError: If the symbol is defined in multiple files and no
      ``choose`` callback resolves the ambiguity.
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
    return ResolvedSymbol(path, tree.module_name(path), symbol, is_package_file(path))

  matches: list[ResolvedSymbol] = []
  for path in tree.paths():
    text = tree.get(path) or ""
    try:
      parsed = ast.parse(text)
    except SyntaxError:
      continue
    for node in parsed.body:
      if getattr(node, "name", None) == head:
        matches.append(ResolvedSymbol(path, tree.module_name(path), symbol, is_package_file(path)))
        break
  if not matches:
    hint = "" if "." in symbol else (
        " Nested members must be referenced as 'Class.member'; the repository-wide "
        "search only matches top-level definitions.")
    raise SymbolNotFoundError(f"Symbol {symbol!r} not found anywhere in repository.{hint}")
  if len(matches) == 1:
    return matches[0]

  ranked = _rank_matches(matches, head)
  candidate_paths = [match.path for match in ranked]
  if choose is not None:
    chosen = choose(symbol, candidate_paths)
    if chosen is not None:
      for match in ranked:
        if match.path == chosen:
          return match
      raise SymbolNotFoundError(f"Chosen file {chosen!r} is not a candidate for {symbol!r}")
  raise AmbiguousSymbolError(symbol, candidate_paths)


def _rank_matches(matches: list[ResolvedSymbol], head: str) -> list[ResolvedSymbol]:
  """Order candidate matches best-first (module named after the symbol wins)."""

  def rank(match: ResolvedSymbol) -> tuple[int, int, str]:
    leaf = match.module.split(".")[-1].lower()
    named = 0 if leaf == head.lower() else 1
    return (named, len(match.path), match.path)

  return sorted(matches, key=rank)


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
      repo_ref = RepoRef("github", repo.owner, repo.name, "HEAD")
      try:
        tree = load_tree(repo_ref, options)
      except Exception:  # noqa: BLE001 - any failure degrades to "unresolved"
        continue
      if tree.has_module(root):
        _attach_pypi_license(tree, payload.get("info", {}))
        return repo_ref.slug, tree
    return None

  return resolver


def _attach_pypi_license(tree: SourceTree, info: dict) -> None:
  """Attach a PyPI-declared license, unless a better hint is already present."""
  if tree.license_hint:
    return
  hint, origin = pypi_license(info)
  if hint:
    tree.license_hint = hint
    tree.license_hint_origin = origin


def extract_symbol(
    repo: str,
    symbol: str,
    *,
    file_hint: str | None = None,
    ref: str = "HEAD",
    options: ExtractOptions | None = None,
    choose: Callable[[str, list[str]], str | None] | None = None,
    trusted: set[str] | frozenset[str] | None = None,
    untrusted: set[str] | frozenset[str] | None = None,
    registry_files: list[str | Path] | None = None,
    strict_trust: bool = False,
    offline: bool = False,
    allow_third_party: bool = True,
    cache_dir: str | Path = ".py_extractor_cache",
    fetch_third_party: bool = True,
    license_notice: bool = True,
    license_full_text: bool = False,
) -> ExtractionResult:
  """Extract *symbol* from *repo* into a single-file source string.

  Args:
    repo: Repository spec (``github:owner/repo``, a URL, or a local path).
    symbol: Class/function name to extract.
    file_hint: Optional repo-relative reference file. Overrides the
      repository-wide symbol search (useful for scripts/CI).
    ref: Git ref (branch/tag/commit).
    options: Explicit extraction options (overrides the keyword arguments).
    choose: Optional callback used to disambiguate a symbol defined in several
      files. Receives the symbol and candidate paths, returns the chosen path.
    trusted: Additional trusted top-level module names.
    untrusted: Trusted module names to revoke.
    registry_files: Extra TOML registry files to load.
    strict_trust: Ignore the built-in curated registry.
    offline: Disallow all network access.
    allow_third_party: Whether to attempt inlining third-party code.
    cache_dir: Cache directory for downloads.
    fetch_third_party: Whether to attempt PyPI->GitHub resolution at all.
    license_notice: Emit a consolidated license attribution table (on by
      default; licensing is a legal requirement for flattened code).
    license_full_text: Embed the full text of each license file as comments.

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
  target = locate_symbol(root_tree, symbol, file_hint=file_hint, choose=choose)

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
  source = emit(
      closure,
      license_notice=license_notice,
      license_full_text=license_full_text,
  )
  return ExtractionResult(source=source, closure=closure, target=target)


__all__ = [
    "ExtractionResult",
    "VerifyResult",
    "__version__",
    "compile_file",
    "compile_source",
    "extract_symbol",
    "import_symbol",
    "locate_symbol",
    "make_third_party_resolver",
]
