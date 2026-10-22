"""Tests for the PyPI -> GitHub source discovery (regression for the sketch bug)."""

from __future__ import annotations

from py_extractor.errors import SourceError
from py_extractor.pypi import (
    PYPI_JSON_URL,
    extract_github_repo,
    fetch_pypi_json,
    guess_distribution_names,
)


class _FakeFetcher:
  """A stand-in Fetcher recording URLs and returning canned payloads."""

  def __init__(self, payloads: dict[str, dict]):
    self.payloads = payloads
    self.urls: list[str] = []

  def fetch_json(self, url: str, **kwargs) -> dict:
    self.urls.append(url)
    if url in self.payloads:
      return self.payloads[url]
    raise SourceError(f"404: {url}")


def _pypi_url(dist: str) -> str:
  return PYPI_JSON_URL.format(dist=dist)


def test_pypi_url_is_well_formed() -> None:
  """Regression: the sketch produced 'https://pypi.orgrequests/json'."""
  assert PYPI_JSON_URL.format(dist="requests") == "https://pypi.org/pypi/requests/json"


def test_fetch_pypi_json_uses_correct_url() -> None:
  fetcher = _FakeFetcher({_pypi_url("requests"): {"info": {"project_urls": {}}}})
  fetch_pypi_json("requests", fetcher)
  assert fetcher.urls == ["https://pypi.org/pypi/requests/json"]


def test_extract_github_from_project_urls_source() -> None:
  info = {"project_urls": {"Source": "https://github.com/psf/requests"}, "home_page": ""}
  repo = extract_github_repo(info)
  assert repo is not None
  assert repo.slug == "psf/requests"


def test_extract_github_prefers_source_over_docs() -> None:
  info = {
      "project_urls": {
          "Documentation": "https://github.com/psf/requests-docs",
          "Source": "https://github.com/psf/requests",
      },
  }
  repo = extract_github_repo(info)
  assert repo is not None
  assert repo.slug == "psf/requests"


def test_extract_github_from_home_page() -> None:
  info = {"home_page": "https://github.com/pallets/flask"}
  assert extract_github_repo(info).slug == "pallets/flask"  # type: ignore[union-attr]


def test_extract_github_strips_git_suffix() -> None:
  info = {"project_urls": {"Source": "https://github.com/psf/requests.git"}}
  assert extract_github_repo(info).name == "requests"  # type: ignore[union-attr]


def test_extract_github_none_when_absent() -> None:
  assert extract_github_repo({"project_urls": {}, "home_page": "https://example.com"}) is None


def test_extract_github_from_list_project_urls() -> None:
  info = {"project_urls": [{"label": "Repository", "url": "https://github.com/o/n"}]}
  assert extract_github_repo(info).slug == "o/n"  # type: ignore[union-attr]


def test_guess_distribution_names() -> None:
  assert "scikit-learn" in guess_distribution_names("scikit_learn") or \
         "scikit_learn" in guess_distribution_names("scikit-learn")
  assert guess_distribution_names("numpy") == ["numpy"]
