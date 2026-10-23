"""Classify an imported root module into a routing decision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..config import TrustPolicy, is_stdlib


class Route(Enum):
  """How a required dependency should be handled."""

  STDLIB = "stdlib"  # keep as an import (standard library)
  TRUSTED = "trusted"  # keep as an import (curated allow-list)
  UNKNOWN = "unknown"  # not stdlib/trusted -> the caller decides in-repo vs external


@dataclass(frozen=True)
class Classification:
  """The routing decision for a dependency root module."""

  root: str
  route: Route


def classify(root: str, policy: TrustPolicy) -> Classification:
  """Classify a dependency *root* module name.

  Precedence: stdlib -> trusted registry -> unknown. This function deliberately
  does not know about the source tree; the closure engine only needs to know
  whether the import is one to keep (stdlib/trusted) or one it must try to
  resolve in-repo / externally.
  """
  if is_stdlib(root):
    return Classification(root, Route.STDLIB)
  if policy.is_trusted(root):
    return Classification(root, Route.TRUSTED)
  return Classification(root, Route.UNKNOWN)


__all__ = ["Classification", "Route", "classify"]
