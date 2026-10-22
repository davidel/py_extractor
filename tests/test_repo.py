"""Tests for repository spec parsing and URL construction."""

from __future__ import annotations

import pytest

from py_extractor.errors import RepoNotFoundError
from py_extractor.source.repo import HFRepoType, RepoRef, parse_repo


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


def test_parse_invalid_github_url_raises() -> None:
  with pytest.raises(RepoNotFoundError):
    parse_repo("https://github.com/onlyone")


def test_parse_invalid_github_spec_raises() -> None:
  with pytest.raises(RepoNotFoundError):
    parse_repo("github:onlyone")


def test_parse_invalid_hf_url_raises() -> None:
  with pytest.raises(RepoNotFoundError):
    parse_repo("https://huggingface.co/onlyone")


def test_github_tarball_url() -> None:
  ref = parse_repo("github:psf/requests", "v2.31.0")
  assert ref.tarball_url() == "https://codeload.github.com/psf/requests/tar.gz/v2.31.0"


def test_hf_urls() -> None:
  ref = RepoRef("hf", "owner", "name", "main")
  assert ref.tree_api_url() == (
      "https://huggingface.co/api/models/owner/name/tree/main?recursive=true")
  assert ref.resolve_url("a/b.py") == ("https://huggingface.co/owner/name/resolve/main/a/b.py")


def test_hf_urls_use_custom_ref() -> None:
  ref = RepoRef("hf", "owner", "name", "v1.2")
  assert ref.tree_api_url() == (
      "https://huggingface.co/api/models/owner/name/tree/v1.2?recursive=true")
  assert ref.resolve_url("pkg/mod.py") == (
      "https://huggingface.co/owner/name/resolve/v1.2/pkg/mod.py")


@pytest.mark.parametrize(
    ("repo_type", "api_seg", "web_prefix"),
    [
        (HFRepoType.MODEL, "models", ""),
        (HFRepoType.DATASET, "datasets", "datasets/"),
        (HFRepoType.SPACE, "spaces", "spaces/"),
    ],
)
def test_hf_urls_per_repo_type(repo_type: str, api_seg: str, web_prefix: str) -> None:
  ref = RepoRef("hf", "owner", "name", "main", repo_type=repo_type)
  assert ref.tree_api_url() == (
      f"https://huggingface.co/api/{api_seg}/owner/name/tree/main?recursive=true")
  assert ref.resolve_url("a/b.py") == (
      f"https://huggingface.co/{web_prefix}owner/name/resolve/main/a/b.py")


@pytest.mark.parametrize(
    ("spec", "repo_type"),
    [
        ("hf:owner/name", HFRepoType.MODEL),
        ("hf:models/owner/name", HFRepoType.MODEL),
        ("hf:datasets/owner/name", HFRepoType.DATASET),
        ("hf:spaces/owner/name", HFRepoType.SPACE),
        ("https://huggingface.co/owner/name", HFRepoType.MODEL),
        ("https://huggingface.co/models/owner/name", HFRepoType.MODEL),
        ("https://huggingface.co/datasets/owner/name", HFRepoType.DATASET),
        ("https://huggingface.co/spaces/owner/name", HFRepoType.SPACE),
    ],
)
def test_parse_hf_repo_type(spec: str, repo_type: str) -> None:
  ref = parse_repo(spec)
  assert ref.kind == "hf"
  assert (ref.owner, ref.name) == ("owner", "name")
  assert ref.repo_type == repo_type


def test_parse_hf_owner_named_like_type_is_not_mistaken() -> None:
  # A two-segment spec whose owner happens to be "datasets" is still a model.
  ref = parse_repo("hf:datasets/some-model")
  assert (ref.owner, ref.name) == ("datasets", "some-model")
  assert ref.repo_type == HFRepoType.MODEL


def test_hf_cache_key_distinguishes_repo_type() -> None:
  model = RepoRef("hf", "o", "n", "main", repo_type=HFRepoType.MODEL)
  dataset = RepoRef("hf", "o", "n", "main", repo_type=HFRepoType.DATASET)
  assert model.cache_key != dataset.cache_key
  assert "dataset" in dataset.cache_key


def test_local_has_no_tree_api(tmp_path) -> None:
  ref = parse_repo(str(tmp_path))
  with pytest.raises(RepoNotFoundError):
    ref.tree_api_url()


def test_github_has_no_resolve_url() -> None:
  ref = RepoRef("github", "o", "n", "main")
  with pytest.raises(RepoNotFoundError):
    ref.resolve_url("a.py")


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


def test_default_ref_is_head() -> None:
  """The default ref resolves to the repository's real default branch.

  ``HEAD`` is symbolic: GitHub (codeload) and Hugging Face both resolve it to
  the default branch, so ``master``-era repositories work without the caller
  knowing the branch name.
  """
  assert parse_repo("github:psf/requests").ref == "HEAD"
  assert parse_repo("hf:owner/name").ref == "HEAD"
  assert parse_repo("pallets/flask").ref == "HEAD"
  assert RepoRef("github", "o", "n").ref == "HEAD"


def test_explicit_ref_overrides_default() -> None:
  assert parse_repo("github:psf/requests", "v2.31.0").ref == "v2.31.0"


def test_head_ref_builds_default_branch_urls() -> None:
  ref = parse_repo("github:google-research/bert")
  assert ref.tarball_url() == ("https://codeload.github.com/google-research/bert/tar.gz/HEAD")
