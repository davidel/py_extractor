"""Configuration: stdlib detection, the trusted-package registry and policy objects."""

from __future__ import annotations

import functools
import sys
import sysconfig
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


@functools.lru_cache(maxsize=1)
def stdlib_module_names() -> frozenset[str]:
  """Return the set of standard-library top-level module names.

  ``sys.stdlib_module_names`` (Python 3.10+) is authoritative; builtins and
  ``__future__`` are added for completeness. A defensive fallback is provided
  for exotic interpreters where the attribute is missing.
  """
  names: set[str] = set(getattr(sys, "stdlib_module_names", ()))
  names.update(getattr(sys, "builtin_module_names", ()))
  names.add("__future__")
  if not names:
    # Extremely defensive fallback: scan the stdlib and platstdlib dirs.
    for key in ("stdlib", "platstdlib"):
      stdlib_dir = sysconfig.get_paths().get(key)
      if not stdlib_dir:
        continue
      for entry in Path(stdlib_dir).iterdir():
        if entry.suffix == ".py" and entry.stem != "__init__":
          names.add(entry.stem)
        elif entry.is_dir():
          names.add(entry.name)
  return frozenset(names)


def is_stdlib(module_root: str) -> bool:
  """Return ``True`` if *module_root* is part of the Python standard library."""
  # Names like ``os.path`` are always stdlib regardless of the submodule.
  return module_root in stdlib_module_names() or module_root in {"os", "sys"}


def is_builtin_name(name: str) -> bool:
  """Return ``True`` if *name* is a builtin (e.g. ``len``, ``range``)."""
  import builtins

  return name in vars(builtins)


def _load_builtin_registry() -> dict[str, str]:
  """Load the shipped curated trusted-package registry (import name -> note)."""
  try:
    data = resources.files("py_extractor").joinpath("data/trusted_packages.toml").read_bytes()
  except (FileNotFoundError, ModuleNotFoundError):  # pragma: no cover - packaging guard
    return {}
  parsed = tomllib.loads(data.decode("utf-8"))
  return dict(parsed.get("packages", {}))


def load_registry_file(path: str | Path) -> dict[str, str]:
  """Load an additional trusted-package registry from a TOML *path*.

  The file may use either a ``[packages]`` table or a flat top-level table.
  """
  parsed = tomllib.loads(Path(path).read_text(encoding="utf-8"))
  table = parsed.get("packages", parsed)
  return {str(k): str(v) for k, v in table.items()}


@dataclass(frozen=True)
class TrustPolicy:
  """Decides whether an imported module should be kept as an import.

  Precedence (first match wins):
    1. Python standard library  -> always trusted
    2. explicit trusted registry -> trusting
    3. otherwise                 -> not trusted (inline or flag)
  """

  builtin_registry: frozenset[str] = field(
      default_factory=lambda: frozenset(_load_builtin_registry()))
  extra_trusted: frozenset[str] = frozenset()
  revoked_trusted: frozenset[str] = frozenset()
  strict: bool = False

  def trusted_modules(self) -> frozenset[str]:
    """Effective set of explicitly trusted top-level module names."""
    if self.strict:
      base: frozenset[str] = frozenset()
    else:
      base = self.builtin_registry
    return (base | self.extra_trusted) - self.revoked_trusted

  def is_trusted(self, module_root: str) -> bool:
    """Return ``True`` if *module_root* should be kept as an import."""
    if is_stdlib(module_root):
      return True
    return module_root in self.trusted_modules()

  @classmethod
  def build(
      cls,
      trusted: frozenset[str] | set[str] | None = None,
      untrusted: frozenset[str] | set[str] | None = None,
      registry_files: list[str | Path] | None = None,
      strict: bool = False,
  ) -> TrustPolicy:
    """Construct a policy from CLI-style arguments."""
    extra: set[str] = set(trusted or ())
    for path in registry_files or []:
      extra.update(load_registry_file(path))
    return cls(
        extra_trusted=frozenset(extra),
        revoked_trusted=frozenset(untrusted or ()),
        strict=strict,
    )


@dataclass
class ExtractOptions:
  """Options controlling a single extraction run."""

  trust: TrustPolicy = field(default_factory=TrustPolicy)
  allow_third_party: bool = True
  cache_dir: Path = Path(".py_extractor_cache")
  offline: bool = False
  verbosity: int = 0
