"""Tests for config: stdlib detection, trust policy and registry loading."""

from __future__ import annotations

import ast

import pytest

from py_extractor.config import (
    TrustPolicy,
    is_builtin_name,
    is_stdlib,
    load_registry_file,
    stdlib_module_names,
)


@pytest.mark.parametrize("name", [
    "os", "re", "sys", "json", "collections", "typing", "dataclasses", "argparse", "itertools",
    "functools", "pathlib", "ast", "urllib"
])
def test_stdlib_recognised(name: str) -> None:
  assert is_stdlib(name)


@pytest.mark.parametrize("name", ["numpy", "torch", "requests", "mypkg", "flask"])
def test_non_stdlib_not_recognised(name: str) -> None:
  assert not is_stdlib(name)


def test_stdlib_names_include_builtins_and_future() -> None:
  names = stdlib_module_names()
  assert "__future__" in names
  assert "sys" in names
  assert "os" in names


@pytest.mark.parametrize("name", ["len", "range", "print", "isinstance", "super"])
def test_builtin_detection(name: str) -> None:
  assert is_builtin_name(name)


def test_builtin_detection_negative() -> None:
  assert not is_builtin_name("numpy")


def test_default_policy_trusts_curated_and_stdlib() -> None:
  policy = TrustPolicy()
  assert policy.is_trusted("torch")
  assert policy.is_trusted("numpy")
  assert policy.is_trusted("os")  # stdlib always trusted
  assert not policy.is_trusted("some_obscure_pkg")


def test_strict_trust_keeps_only_stdlib() -> None:
  policy = TrustPolicy(strict=True)
  assert policy.is_trusted("os")
  assert not policy.is_trusted("torch")


def test_trust_policy_build_add_and_revoke() -> None:
  policy = TrustPolicy.build(trusted={"my_local_pkg"}, untrusted={"torch"})
  assert policy.is_trusted("my_local_pkg")
  assert not policy.is_trusted("torch")
  assert policy.is_trusted("os")


def test_registry_keys_are_valid_import_identifiers() -> None:
  """Every registry key must be usable in an ``import`` statement."""
  policy = TrustPolicy()
  for root in policy.builtin_registry:
    assert root.isidentifier(), f"{root!r} is not a valid identifier"
    ast.parse(f"import {root}")  # must not raise


def test_import_name_vs_distribution_name_coverage() -> None:
  """Alias-style import names must be present, not just distribution names."""
  policy = TrustPolicy()
  for import_name in ("PIL", "sklearn", "cv2", "yaml", "bs4", "dateutil", "pkg_resources"):
    assert import_name in policy.builtin_registry, f"{import_name} missing from registry"


def test_load_registry_file(tmp_path) -> None:
  path = tmp_path / "extra.toml"
  path.write_text("[packages]\nmy_thing = 'my thing'\n", encoding="utf-8")
  loaded = load_registry_file(path)
  assert loaded == {"my_thing": "my thing"}


def test_registry_file_used_by_policy(tmp_path) -> None:
  path = tmp_path / "extra.toml"
  path.write_text("[packages]\nmy_thing = 'my thing'\n", encoding="utf-8")
  policy = TrustPolicy.build(registry_files=[path])
  assert policy.is_trusted("my_thing")


def test_load_registry_file_flat_table(tmp_path) -> None:
  path = tmp_path / "extra.toml"
  path.write_text("plain = 'flat table'\n", encoding="utf-8")
  assert load_registry_file(path) == {"plain": "flat table"}
