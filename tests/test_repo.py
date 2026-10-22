"""Tests for repository spec parsing and URL construction."""

from __future__ import annotations

import pytest

from py_extractor.errors import RepoNotFoundError
from py_extractor.source.repo import RepoRef, parse_repo


def test_parse_github_shorthand() -> None:
  ref = parse_repo("github:psf/requests", "main")
  assert ref.kind == "github"
  assert ref.owner == "psf"
  assert ref.name == "requests"
  assert ref.slug == "psf/requests"


def test_parse_hf_shorthand() -> None:
  ref = parse_repo("hf:owner/name", "main")
  assert ref.kind == "hf"
  assert (ref.owner, ref.name) == ("owner", "name")


def test_parse_hf_shorthand_requires_two_segments() -> None:
  with pytest.raises(RepoNotFoundError):
    parse_repo("hf:single-segment")


def test_parse_bare_owner_repo_defaults_github() -> None:
  ref = parse_repo("pallets/flask")
  assert ref.kind == "github"
  assert (ref.owner, ref.name) == ("pallets", "flask")


def test_parse_github_url() -> None:
  ref = parse_repo("https://github.com/w-e-w/DAT")
  assert ref.kind == "github"
  assert (ref.owner, ref.name) == ("w-e-w", "DAT")


def test_parse_github_url_with_tree_path() -> None:
  ref = parse_repo("https://github.com/w-e-w/DAT/tree/main")
  assert (ref.owner, ref.name) == ("w-e-w", "DAT")


def test_parse_hf_url() -> None:
  ref = parse_repo("https://huggingface.co/google-bert/bert-base-uncased")
  assert ref.kind == "hf"
  assert (ref.owner, ref.name) == ("google-bert", "bert-base-uncased")


def test_parse_hf_url_with_models_prefix() -> None:
  ref = parse_repo("https://huggingface.co/models/google-bert/bert-base-uncased")
  assert ref.kind == "hf"
  assert (ref.owner, ref.name) == ("google-bert", "bert-base-uncased")


def test_parse_local_directory(tmp_path) -> None:
  ref = parse_repo(str(tmp_path))
  assert ref.kind == "local"
  assert ref.path == tmp_path.resolve()


def test_parse_invalid_spec_raises() -> None:
  with pytest.raises(RepoNotFoundError):
    parse_repo("!!!not a repo!!!")


def test_github_tarball_url() -> None:
  ref = parse_repo("github:psf/requests", "v2.31.0")
  assert ref.tarball_url() == "https://codeload.github.com/psf/requests/tar.gz/v2.31.0"


def test_hf_urls() -> None:
  ref = RepoRef("hf", "owner", "name", "main")
  assert ref.tree_api_url().endswith("/api/models/owner/name/tree/main?recursive=true")
  assert ref.resolve_url("a/b.py").endswith("/owner/name/resolve/main/a/b.py")


def test_local_has_no_tarball_url(tmp_path) -> None:
  ref = parse_repo(str(tmp_path))
  with pytest.raises(RepoNotFoundError):
    ref.tarball_url()


def test_cache_key_distinct_per_ref() -> None:
  a = RepoRef("github", "o", "n", "main")
  b = RepoRef("github", "o", "n", "dev")
  assert a.cache_key != b.cache_key


def test_git_suffix_stripped() -> None:
  ref = parse_repo("https://github.com/psf/requests.git")
  assert ref.name == "requests"
