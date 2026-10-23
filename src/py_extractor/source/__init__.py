"""Source acquisition: fetch and materialise a repository into a :class:`SourceTree`."""

from __future__ import annotations

from ..config import ExtractOptions
from ..errors import RepoNotFoundError
from .fetcher import Fetcher
from .repo import RepoRef, parse_repo
from .tree import SourceTree, is_package_file

__all__ = ["Fetcher", "RepoRef", "SourceTree", "is_package_file", "load_tree", "parse_repo"]


def load_tree(repo: RepoRef, options: ExtractOptions | None = None) -> SourceTree:
  """Load the source tree for *repo*.

  GitHub repositories are downloaded as a tarball (one request, exact
  snapshot). Local repositories are read directly from disk. Hugging Face
  repositories fall back to the per-file resolve endpoint.

  Args:
    repo: The repository reference.
    options: Extraction options (cache directory, offline flag).

  Returns:
    A populated :class:`SourceTree`.

  Raises:
    RepoNotFoundError: If the repository (or its default ref) cannot be found.
  """
  options = options or ExtractOptions()
  fetcher = Fetcher(options.cache_dir, offline=options.offline)

  if repo.kind == "local":
    if repo.path is None or not repo.path.is_dir():
      raise RepoNotFoundError(f"Local repository not found: {repo.path}")
    return SourceTree.from_directory(str(repo.path))

  if repo.kind == "github":
    try:
      body = fetcher.fetch_tarball(repo.tarball_url())
    except RepoNotFoundError as exc:
      raise RepoNotFoundError(
          f"Could not download {repo.slug}@{repo.ref}. Check the repository name, or pass "
          f"--ref to select a specific branch/tag/commit (e.g. --ref v1.0).") from exc
    tree = SourceTree.from_tarball(body)
    _attach_github_license(repo, tree, fetcher)
    return tree

  if repo.kind == "hf":
    from .hf import load_hf_tree  # local import to avoid cycles

    tree = load_hf_tree(repo, fetcher)
    _attach_hf_license(repo, tree, fetcher)
    return tree

  raise RepoNotFoundError(f"Unsupported repository kind: {repo.kind!r}")


def _attach_github_license(repo: RepoRef, tree: SourceTree, fetcher: Fetcher) -> None:
  """Attach an authoritative GitHub SPDX id when one can be fetched."""
  from .metadata import fetch_github_license  # local import to avoid cycles

  spdx = fetch_github_license(repo, fetcher)
  if spdx:
    tree.license_hint = spdx
    tree.license_hint_origin = "github-api"


def _attach_hf_license(repo: RepoRef, tree: SourceTree, fetcher: Fetcher) -> None:
  """Attach the license declared on a Hugging Face repo card, when present."""
  from .metadata import fetch_hf_license  # local import to avoid cycles

  hint = fetch_hf_license(repo, fetcher)
  if hint:
    tree.license_hint = hint
    tree.license_hint_origin = "hf-model-card"
