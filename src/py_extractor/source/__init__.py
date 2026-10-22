"""Source acquisition: fetch and materialise a repository into a :class:`SourceTree`."""

from __future__ import annotations

from ..config import ExtractOptions
from ..errors import RepoNotFoundError
from .fetcher import Fetcher
from .repo import RepoRef, parse_repo
from .tree import SourceTree

__all__ = ["Fetcher", "RepoRef", "SourceTree", "load_tree", "parse_repo"]


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
          f"Could not download {repo.slug}@{repo.ref}. If the default branch is not "
          f"'main', pass --ref (e.g. --ref master).") from exc
    return SourceTree.from_tarball(body)

  if repo.kind == "hf":
    from .hf import load_hf_tree  # local import to avoid cycles

    return load_hf_tree(repo, fetcher)

  raise RepoNotFoundError(f"Unsupported repository kind: {repo.kind!r}")
