"""Tests for the stdlib/trusted/unknown classifier."""

from __future__ import annotations

from py_extractor.config import TrustPolicy
from py_extractor.extract.classifier import Route, classify


def test_classify_stdlib() -> None:
  assert classify("os", TrustPolicy()).route is Route.STDLIB
  assert classify("json", TrustPolicy()).route is Route.STDLIB


def test_classify_trusted() -> None:
  assert classify("torch", TrustPolicy()).route is Route.TRUSTED
  assert classify("numpy", TrustPolicy()).route is Route.TRUSTED


def test_classify_unknown() -> None:
  assert classify("totally_obscure_pkg", TrustPolicy()).route is Route.UNKNOWN


def test_classify_strict_still_stdlib() -> None:
  policy = TrustPolicy(strict=True)
  assert classify("os", policy).route is Route.STDLIB
  assert classify("torch", policy).route is Route.UNKNOWN


def test_classify_extra_trusted() -> None:
  policy = TrustPolicy.build(trusted={"my_local"})
  assert classify("my_local", policy).route is Route.TRUSTED
