"""Tests for the caching fetcher (network calls are monkeypatched)."""

from __future__ import annotations

from pathlib import Path

import pytest

from py_extractor.errors import RepoNotFoundError, SourceError
from py_extractor.source import fetcher as fetcher_mod
from py_extractor.source.fetcher import Fetcher, fetch_bytes


def test_fetch_bytes_offline_refuses(monkeypatch) -> None:
  with pytest.raises(SourceError):
    fetch_bytes("https://example.com/x", offline=True)


def test_fetch_bytes_404_raises_repo_not_found(monkeypatch) -> None:

  class _Resp:
    status_code = 404
    ok = False
    content = b""

  monkeypatch.setattr(fetcher_mod.requests, "get", lambda *a, **k: _Resp())
  with pytest.raises(RepoNotFoundError):
    fetch_bytes("https://example.com/missing")


def test_fetch_bytes_ok(monkeypatch) -> None:

  class _Resp:
    status_code = 200
    ok = True
    content = b"hello"

  monkeypatch.setattr(fetcher_mod.requests, "get", lambda *a, **k: _Resp())
  assert fetch_bytes("https://example.com/ok") == b"hello"


def test_fetch_bytes_network_error(monkeypatch) -> None:

  def boom(*a, **k):
    raise fetcher_mod.requests.RequestException("nope")

  monkeypatch.setattr(fetcher_mod.requests, "get", boom)
  with pytest.raises(SourceError):
    fetch_bytes("https://example.com/err")


def test_fetcher_cache_hit_avoids_network(tmp_path: Path, monkeypatch) -> None:
  calls = {"n": 0}

  class _Resp:
    status_code = 200
    ok = True
    content = b"cached-body"

  def fake_get(*a, **k):
    calls["n"] += 1
    return _Resp()

  monkeypatch.setattr(fetcher_mod.requests, "get", fake_get)
  fetcher = Fetcher(tmp_path)
  url = "https://example.com/data.json"
  assert fetcher.fetch_bytes(url) == b"cached-body"
  assert fetcher.fetch_bytes(url) == b"cached-body"
  assert calls["n"] == 1  # second call served from cache


def test_fetcher_cache_key_is_filesystem_safe(tmp_path: Path, monkeypatch) -> None:

  class _Resp:
    status_code = 200
    ok = True
    content = b"x"

  monkeypatch.setattr(fetcher_mod.requests, "get", lambda *a, **k: _Resp())
  fetcher = Fetcher(tmp_path)
  fetcher.fetch_bytes("https://example.com/a/b/c?query=1&x=2")
  files = list(tmp_path.iterdir())
  assert files
  assert all("?" not in f.name and "&" not in f.name for f in files)


def test_fetcher_offline_reads_cache(tmp_path: Path, monkeypatch) -> None:

  class _Resp:
    status_code = 200
    ok = True
    content = b"seed"

  monkeypatch.setattr(fetcher_mod.requests, "get", lambda *a, **k: _Resp())
  url = "https://example.com/seed"
  Fetcher(tmp_path).fetch_bytes(url)  # populate cache

  offline = Fetcher(tmp_path, offline=True)
  assert offline.fetch_bytes(url) == b"seed"  # served from cache, no network


def test_fetcher_offline_cache_miss_raises(tmp_path: Path) -> None:
  offline = Fetcher(tmp_path, offline=True)
  with pytest.raises(SourceError):
    offline.fetch_bytes("https://example.com/not-cached")
