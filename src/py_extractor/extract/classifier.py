"""Classify an imported root module into a routing decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import TrustPolicy, is_stdlib


class Route(Enum):
  """How a required dependency should be handled."""

  STDLIB = "stdlib"  # keep as an import (standard library)
  TRUSTED = "trusted"  # keep as an import (curated allow-list)
  INTERNAL = "internal"  # in-repo: recurse and inline
  THIRD_PARTY = "third_party"  # external: inline if resolvable, else flag
  UNKNOWN = "unknown"  # could not be resolved at all -> flag


@dataclass(frozen=True)
class Classification:
  """The routing decision for a dependency root module."""

  root: str
  route: Route
  reason: str = ""


def classify(root: str, policy: TrustPolicy) -> Classification:
  """Classify a dependency *root* module name.

  Precedence: stdlib -> trusted registry -> (caller decides internal vs third
  party). This function deliberately does not know about the source tree; the
  closure engine upgrades ``UNKNOWN`` to ``INTERNAL`` when the module exists
  in-repo.
  """
  if is_stdlib(root):
    return Classification(root, Route.STDLIB, "python standard library")
  if policy.is_trusted(root):
    return Classification(root, Route.TRUSTED, "curated trusted package")
  return Classification(root, Route.UNKNOWN, "not stdlib and not trusted")


__all__ = ["Classification", "Route", "classify"]
