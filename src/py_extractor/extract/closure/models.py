"""Data carriers for the transitive dependency closure.

These are the plain value objects shared by the closure walk and the emitter:
the flags surfaced in the report, the extracted units, the kept imports, the
local-name bindings, the preserved optional-import guards and the aggregate
:class:`ClosureResult`.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from ...config import ExtractOptions
from ...licensing import LicenseInfo
from ...source.tree import SourceTree


class FlagKind(Enum):
  """Categories of things that could not be cleanly vendored."""

  UNRESOLVED_THIRD_PARTY = "unresolved_third_party"
  UNRESOLVED_NAME = "unresolved_name"
  STAR_IMPORT = "star_import"
  COLLISION = "collision"
  GUARDED_IMPORT = "guarded_import"
  NESTED_IMPORT = "nested_import"


@dataclass
class Flag:
  """A single unresolved / risky item, surfaced in the report."""

  kind: FlagKind
  detail: str
  location: str = ""


@dataclass
class ExtractedUnit:
  """One inlined definition (or whole module) destined for the output."""

  key: str
  repo_id: str
  path: str
  module: str
  symbol: str | None
  node: ast.AST
  source: str


@dataclass(frozen=True)
class KeptImport:
  """An import that is kept as-is (stdlib or trusted)."""

  kind: str
  module: str
  name: str
  asname: str | None = None

  def render(self) -> str:
    """Return the ``import ...`` / ``from ... import ...`` source line."""
    if self.kind == "import":
      if self.asname and self.asname != self.module.split(".")[0]:
        return f"import {self.module} as {self.asname}"
      return f"import {self.module}"
    if self.asname:
      return f"from {self.module} import {self.name} as {self.asname}"
    return f"from {self.module} import {self.name}"


@dataclass(frozen=True)
class Binding:
  """A local name the extracted code needs, bound to inlined definitions.

  Importing a name into a file and then inlining the *definition* leaves the
  call site dangling (``from .core import Widget as W`` -> ``W()`` fails with
  ``NameError`` once ``Widget`` is flattened). A binding re-creates the missing
  local name so the flattened code still resolves.

  Attributes:
    kind: ``"alias"`` (``W = Widget``) or ``"module"`` (a namespace object
      exposing a module's inlined members).
    local: The name used at the call site (``W``, ``core``, ``pkg``).
    target: For ``"alias"``, the inlined top-level name to alias.
    unit_key: The unit this binding must be emitted after (the definition it
      depends on), so ordering matches dependency order.
    parts: For ``"module"``, the dotted path under *local* (``("core",)`` for
      ``import pkg.core``); empty when *local* names the module directly.
    members: For ``"module"``, the inlined top-level names to expose.
  """

  kind: str
  local: str
  target: str | None = None
  unit_key: str | None = None
  parts: tuple[str, ...] = ()
  members: tuple[str, ...] = ()


@dataclass(frozen=True)
class GuardedBlock:
  """A verbatim module-level statement preserved to keep an optional import.

  Hoisting ``import ujson as json`` out of a ``try: ... except ImportError:``
  block turns an *optional* dependency into a hard one (or, when the fallback
  binds the same name, silently drops it). Instead the whole guarding statement
  is copied verbatim so the original fallback logic is preserved.

  Attributes:
    source: The unparsed source of the guarding statement.
    module: The dotted module the block came from (for provenance).
    lineno: Line number in the origin file.
  """

  source: str
  module: str
  lineno: int


@dataclass
class ClosureResult:
  """The full result of the closure computation."""

  target: ExtractedUnit
  units: list[ExtractedUnit]
  kept_imports: list[KeptImport]
  bindings: list[Binding] = field(default_factory=list)
  guarded_blocks: list[GuardedBlock] = field(default_factory=list)
  future_imports: list[KeptImport] = field(default_factory=list)
  flags: list[Flag] = field(default_factory=list)
  licenses: list[LicenseInfo] = field(default_factory=list)
  license_ids: dict[str, str] = field(default_factory=dict)


# A resolver maps an unresolved third-party import root to a (repo_id, tree).
ThirdPartyResolver = Callable[[str, ExtractOptions], "tuple[str, SourceTree] | None"]

__all__ = [
    "Binding",
    "ClosureResult",
    "ExtractedUnit",
    "Flag",
    "FlagKind",
    "GuardedBlock",
    "KeptImport",
    "ThirdPartyResolver",
]
