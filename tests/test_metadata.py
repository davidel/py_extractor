"""Tests for authoritative license metadata lookups."""

from __future__ import annotations

from py_extractor.errors import SourceError
from py_extractor.source.metadata import fetch_github_license, fetch_hf_license
from py_extractor.source.repo import RepoRef


class _FakeFetcher:
  """A fake fetcher returning canned JSON payloads per URL."""

  def __init__(self, responses: dict[str, object], *, error: Exception | None = None):
    self.responses = responses
    self.error = error
    self.urls: list[str] = []

  def fetch_json(self, url: str, **kwargs):
    self.urls.append(url)
    if self.error is not None:
      raise self.error
    return self.responses.get(url)


def test_github_license_returns_spdx() -> None:
  url = "https://api.github.com/repos/acme/thing/license"
  fetcher = _FakeFetcher({url: {"license": {"spdx_id": "MIT"}}})
  assert fetch_github_license(RepoRef("github", "acme", "thing"), fetcher) == "MIT"
  assert fetcher.urls == [url]


def test_github_license_ignores_noassertion() -> None:
  url = "https://api.github.com/repos/acme/thing/license"
  fetcher = _FakeFetcher({url: {"license": {"spdx_id": "NOASSERTION"}}})
  assert fetch_github_license(RepoRef("github", "acme", "thing"), fetcher) is None


def test_github_license_swallows_errors() -> None:
  fetcher = _FakeFetcher({}, error=SourceError("boom"))
  assert fetch_github_license(RepoRef("github", "acme", "thing"), fetcher) is None


def test_github_license_handles_non_dict_payload() -> None:
  url = "https://api.github.com/repos/acme/thing/license"
  fetcher = _FakeFetcher({url: ["not", "a", "dict"]})
  assert fetch_github_license(RepoRef("github", "acme", "thing"), fetcher) is None


def test_hf_license_from_card_data() -> None:
  url = "https://huggingface.co/api/models/acme/thing"
  fetcher = _FakeFetcher({url: {"cardData": {"license": "apache-2.0"}}})
  assert fetch_hf_license(RepoRef("hf", "acme", "thing"), fetcher) == "apache-2.0"


def test_hf_license_uses_dataset_endpoint() -> None:
  url = "https://huggingface.co/api/datasets/acme/thing"
  fetcher = _FakeFetcher({url: {"cardData": {"license": ["mit"]}}})
  ref = RepoRef("hf", "acme", "thing", repo_type="dataset")
  assert fetch_hf_license(ref, fetcher) == "mit"


def test_hf_license_swallows_errors() -> None:
  fetcher = _FakeFetcher({}, error=SourceError("boom"))
  assert fetch_hf_license(RepoRef("hf", "acme", "thing"), fetcher) is None
