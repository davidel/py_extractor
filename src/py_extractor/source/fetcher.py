"""HTTP fetching with an on-disk cache.

All network access for the project funnels through this module, so tests can
monkeypatch :func:`fetch_bytes` / :class:`Fetcher` cleanly.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..errors import RepoNotFoundError, SourceError

DEFAULT_TIMEOUT = 30


def fetch_bytes(url: str, *, timeout: int = DEFAULT_TIMEOUT, offline: bool = False) -> bytes:
  """Fetch *url* and return the raw body.

  Args:
    url: The absolute URL to download.
    timeout: Socket timeout in seconds.
    offline: When ``True``, raise instead of performing any network call.

  Returns:
    The response body as bytes.

  Raises:
    RepoNotFoundError: On HTTP 404.
    SourceError: On any other network failure, or when *offline* is set.
  """
  if offline:
    raise SourceError(f"Offline mode: refused to fetch {url}")
  try:
    response = requests.get(url, timeout=timeout, allow_redirects=True)
  except requests.RequestException as exc:  # network-level failure
    raise SourceError(f"Failed to fetch {url}: {exc}") from exc
  if response.status_code == 404:
    raise RepoNotFoundError(f"Not found (404): {url}")
  if not response.ok:
    raise SourceError(f"HTTP {response.status_code} for {url}")
  return response.content


def fetch_text(url: str, **kwargs) -> str:
  """Fetch *url* and decode the body as UTF-8 text."""
  return fetch_bytes(url, **kwargs).decode("utf-8", errors="replace")


def fetch_json(url: str, **kwargs) -> dict:
  """Fetch *url* and parse the body as JSON."""
  return json.loads(fetch_text(url, **kwargs))


@dataclass
class Fetcher:
  """A caching HTTP fetcher bound to a cache directory.

  Attributes:
    cache_dir: Directory under which tarballs and payloads are cached.
    offline: When ``True``, only cache reads are permitted.
  """

  cache_dir: Path
  offline: bool = False

  def __post_init__(self) -> None:
    self.cache_dir = Path(self.cache_dir)

  def _cache_path(self, url: str, suffix: str = ".bin") -> Path:
    key = urlparse(url).netloc + urlparse(url).path
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key)
    return self.cache_dir / (safe[:180] + suffix)

  def fetch_bytes(self, url: str, *, use_cache: bool = True) -> bytes:
    """Fetch *url*, consulting and populating the on-disk cache."""
    cache_path = self._cache_path(url)
    if use_cache and cache_path.exists():
      return cache_path.read_bytes()
    body = fetch_bytes(url, offline=self.offline)
    if use_cache:
      self.cache_dir.mkdir(parents=True, exist_ok=True)
      tmp = cache_path.with_suffix(cache_path.suffix + ".part")
      tmp.write_bytes(body)
      tmp.replace(cache_path)
    return body

  def fetch_text(self, url: str, **kwargs) -> str:
    """Fetch *url* and decode as UTF-8 text."""
    return self.fetch_bytes(url, **kwargs).decode("utf-8", errors="replace")

  def fetch_json(self, url: str, **kwargs) -> dict:
    """Fetch *url* and parse as JSON."""
    return json.loads(self.fetch_text(url, **kwargs))

  def fetch_tarball(self, url: str, *, use_cache: bool = True) -> bytes:
    """Fetch a gzipped tarball, using a ``.tar.gz`` cache suffix."""
    cache_path = self._cache_path(url, suffix=".tar.gz")
    if use_cache and cache_path.exists():
      return cache_path.read_bytes()
    body = fetch_bytes(url, offline=self.offline)
    if use_cache:
      self.cache_dir.mkdir(parents=True, exist_ok=True)
      tmp = cache_path.with_suffix(cache_path.suffix + ".part")
      tmp.write_bytes(body)
      tmp.replace(cache_path)
    return body


def open_tarball(body: bytes) -> tarfile.TarFile:
  """Open an in-memory gzipped tarball."""
  try:
    return tarfile.open(fileobj=io.BytesIO(body), mode="r:gz")
  except tarfile.TarError as exc:
    raise SourceError(f"Invalid tarball: {exc}") from exc


__all__ = [
    "DEFAULT_TIMEOUT",
    "Fetcher",
    "fetch_bytes",
    "fetch_json",
    "fetch_text",
    "open_tarball",
]
