"""PyPI -> GitHub source discovery.

Fixes the original sketch's malformed URL (``https://pypi.org{pkg}/json``) and
scans ``project_urls`` rather than trusting a single ``home_page`` field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import SourceError
from .source.fetcher import Fetcher

PYPI_JSON_URL = "https://pypi.org/pypi/{dist}/json"

# Preference order for project_urls labels that point at source repositories.
_SOURCE_LABELS = (
    "source",
    "source code",
    "repository",
    "code",
    "github",
    "homepage",
    "home",
    "download",
    "downloads",
    "url",
)

_GITHUB_RE = re.compile(r"github\.com/([^/\s#?]+)/([^/\s#?]+)")


@dataclass(frozen=True)
class SourceRepo:
  """A discovered source repository."""

  owner: str
  name: str

  @property
  def slug(self) -> str:
    return f"{self.owner}/{self.name}"


def fetch_pypi_json(dist: str, fetcher: Fetcher) -> dict:
  """Fetch the PyPI JSON metadata for distribution *dist*.

  Args:
    dist: The PyPI distribution name (e.g. ``scikit-learn``).
    fetcher: The caching fetcher to use.

  Returns:
    The parsed JSON payload.

  Raises:
    SourceError: If the distribution does not exist or the payload is invalid.
  """
  url = PYPI_JSON_URL.format(dist=dist)
  try:
    payload = fetcher.fetch_json(url)
  except SourceError as exc:
    raise SourceError(f"PyPI lookup failed for {dist!r}: {exc}") from exc
  if not isinstance(payload, dict) or "info" not in payload:
    raise SourceError(f"Unexpected PyPI payload for {dist!r}")
  return payload


def extract_github_repo(info: dict) -> SourceRepo | None:
  """Extract the most likely GitHub source repo from PyPI ``info`` metadata.

  Scans ``project_urls`` (label-preferred) then ``home_page``.
  """
  candidates: list[tuple[int, str]] = []

  project_urls = info.get("project_urls") or {}
  if isinstance(project_urls, dict):
    items = project_urls.items()
  elif isinstance(project_urls, list):
    items = ((entry.get("label", ""), entry.get("url", ""))
             for entry in project_urls
             if isinstance(entry, dict))
  else:
    items = []

  for label, url in items:
    if not url:
      continue
    match = _GITHUB_RE.search(url)
    if not match:
      continue
    rank = _label_rank(label)
    candidates.append((rank, url))

  home = info.get("home_page")
  if home and _GITHUB_RE.search(home):
    candidates.append((_label_rank("homepage"), home))

  if not candidates:
    return None
  candidates.sort(key=lambda pair: pair[0])
  match = _GITHUB_RE.search(candidates[0][1])
  if not match:
    return None
  owner, name = match.group(1), match.group(2)
  if name.endswith(".git"):
    name = name[:-len(".git")]
  return SourceRepo(owner, name)


def _label_rank(label: str) -> int:
  label = (label or "").strip().lower()
  for index, preferred in enumerate(_SOURCE_LABELS):
    if preferred in label:
      return index
  return len(_SOURCE_LABELS)


def guess_distribution_names(import_name: str) -> list[str]:
  """Return candidate PyPI distribution names for an *import_name*.

  The import name and distribution name frequently differ only by separators
  (``scikit_learn`` vs ``scikit-learn``) or casing. We generate the obvious
  candidates; callers try them in order.
  """
  candidates = [
      import_name,
      import_name.replace("_", "-"),
      import_name.replace("-", "_"),
      import_name.lower(),
  ]
  # Preserve order while de-duplicating.
  seen: set[str] = set()
  ordered: list[str] = []
  for candidate in candidates:
    if candidate and candidate not in seen:
      seen.add(candidate)
      ordered.append(candidate)
  return ordered


__all__ = [
    "PYPI_JSON_URL",
    "SourceRepo",
    "extract_github_repo",
    "fetch_pypi_json",
    "guess_distribution_names",
]
