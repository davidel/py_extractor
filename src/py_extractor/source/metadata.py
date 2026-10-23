"""Best-effort authoritative license lookups for remote repositories.

These helpers query repository APIs for a structured license (GitHub's
``/license`` endpoint, Hugging Face's model/dataset info endpoint) and return a
license hint that :mod:`py_extractor.licensing` can resolve to an SPDX id.

Every function here is *best effort*: any network error, rate limit or missing
field degrades to ``None`` so extraction still succeeds using the license files
that were downloaded with the tree.
"""

from __future__ import annotations

from .fetcher import Fetcher
from .repo import RepoRef

_GITHUB_LICENSE_URL = "https://api.github.com/repos/{owner}/{name}/license"
_HF_INFO_URL = "https://huggingface.co/api/{kind}/{owner}/{name}"

# HF uses a handful of type names in its API path.
_HF_KIND = {"model": "models", "dataset": "datasets", "space": "spaces"}


def fetch_github_license(repo: RepoRef, fetcher: Fetcher) -> str | None:
  """Return the SPDX id GitHub reports for *repo*, or ``None``.

  The endpoint is rate-limited for unauthenticated callers; a failure simply
  yields ``None`` so the caller can fall back to text detection.
  """
  url = _GITHUB_LICENSE_URL.format(owner=repo.owner, name=repo.name)
  try:
    payload = fetcher.fetch_json(url)
  except Exception:  # noqa: BLE001 - metadata is optional, never fatal
    return None
  if not isinstance(payload, dict):
    return None
  license_field = payload.get("license")
  if isinstance(license_field, dict):
    spdx = license_field.get("spdx_id")
    if isinstance(spdx, str) and spdx and spdx.upper() != "NOASSERTION":
      return spdx
  return None


def fetch_hf_license(repo: RepoRef, fetcher: Fetcher) -> str | None:
  """Return the license declared on a Hugging Face repo card, or ``None``."""
  kind = _HF_KIND.get(repo.repo_type, "models")
  url = _HF_INFO_URL.format(kind=kind, owner=repo.owner, name=repo.name)
  try:
    payload = fetcher.fetch_json(url)
  except Exception:  # noqa: BLE001 - metadata is optional, never fatal
    return None
  if not isinstance(payload, dict):
    return None
  card = payload.get("cardData")
  if isinstance(card, dict):
    license_field = card.get("license")
    if isinstance(license_field, str) and license_field.strip():
      return license_field.strip()
    if isinstance(license_field, list) and license_field:
      first = license_field[0]
      if isinstance(first, str) and first.strip():
        return first.strip()
  return None


__all__ = ["fetch_github_license", "fetch_hf_license"]
