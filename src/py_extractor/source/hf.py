"""Hugging Face repository loading via the tree API + per-file resolve endpoint."""

from __future__ import annotations

from ..errors import RepoNotFoundError, SourceError
from .fetcher import Fetcher
from .repo import RepoRef
from .tree import SourceTree

_PY_EXTS = (".py", ".pyi")


def load_hf_tree(repo: RepoRef, fetcher: Fetcher) -> SourceTree:
  """Load a Hugging Face repository into a :class:`SourceTree`.

  The HF hub does not serve a single tarball for models, so we list the tree
  once (recursive) and fetch each Python file via the ``resolve`` endpoint.
  """
  try:
    entries = fetcher.fetch_json(repo.tree_api_url())
  except RepoNotFoundError as exc:
    raise RepoNotFoundError(f"Could not list {repo.slug}@{repo.ref}. If the default branch is not "
                            f"'main', pass --ref.") from exc
  if not isinstance(entries, list):
    raise SourceError(f"Unexpected HF tree response for {repo.slug}")

  tree = SourceTree()
  for entry in entries:
    if not isinstance(entry, dict):
      continue
    # The tree API tags entries; skip directories/submodules when present.
    if entry.get("type", "file") != "file":
      continue
    path = entry.get("path")
    if not path or not path.endswith(_PY_EXTS):
      continue
    text = fetcher.fetch_text(repo.resolve_url(path))
    tree.add(path, text)
  return tree


__all__ = ["load_hf_tree"]
