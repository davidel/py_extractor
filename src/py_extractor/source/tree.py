"""An in-memory view of a repository's files with a module-name index.

Instead of guessing per-file URLs, we materialise the whole (Python) source
tree once and resolve imports against the *real* file layout. This is what
makes relative imports, ``src/`` layouts, namespace packages and ``__init__``
re-exports work correctly.
"""

from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from ..errors import SourceError


@dataclass
class SourceTree:
  """A repository snapshot: repo-relative POSIX path -> source text.

  Attributes:
    files: Mapping of repo-relative POSIX path to decoded source text.
    package_dirs: Set of directories (POSIX strings) that contain ``__init__.py``.
  """

  files: dict[str, str] = field(default_factory=dict)
  package_dirs: set[str] = field(default_factory=set)
  _index: dict[str, list[tuple[int, str]]] | None = field(default=None, repr=False)

  # -- construction -------------------------------------------------------

  def add(self, path: str, text: str) -> None:
    """Add a single file to the tree.

    A leading UTF-8 BOM (``\\ufeff``) is stripped: it is a byte-order artifact
    rather than source, and ``ast.parse`` rejects it with "invalid non-printable
    character U+FEFF". Several real-world repositories ship BOM-prefixed files.
    """
    if text.startswith("\ufeff"):
      text = text[1:]
    rel = str(PurePosixPath(path))
    self.files[rel] = text
    if PurePosixPath(rel).name == "__init__.py":
      parent = str(PurePosixPath(rel).parent)
      self.package_dirs.add(parent if parent != "." else "")
    self._index = None

  @classmethod
  def from_mapping(cls, mapping: dict[str, str]) -> SourceTree:
    """Build a tree from a plain ``path -> text`` mapping (used in tests)."""
    tree = cls()
    for path, text in mapping.items():
      tree.add(path, text)
    return tree

  @classmethod
  def from_directory(cls, root: str, *, exts: tuple[str, ...] = (".py", ".pyi")) -> SourceTree:
    """Build a tree from a local directory."""
    from pathlib import Path

    tree = cls()
    base = Path(root)
    for file in sorted(base.rglob("*")):
      if file.is_file() and file.suffix in exts:
        rel = file.relative_to(base).as_posix()
        tree.add(rel, file.read_text(encoding="utf-8", errors="replace"))
    return tree

  @classmethod
  def from_tarball(cls, body: bytes, *, exts: tuple[str, ...] = (".py", ".pyi")) -> SourceTree:
    """Build a tree from an in-memory gzipped tarball, stripping the top dir."""
    entries: list[tuple[str, str]] = []
    try:
      with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
        for member in archive.getmembers():
          if not member.isfile() or not member.name.endswith(exts):
            continue
          extracted = archive.extractfile(member)
          if extracted is None:
            continue
          entries.append((member.name, extracted.read().decode("utf-8", errors="replace")))
    except tarfile.TarError as exc:
      raise SourceError(f"Invalid tarball: {exc}") from exc

    prefix = _common_prefix(name for name, _ in entries)
    tree = cls()
    for name, text in entries:
      tree.add(name[len(prefix):] if prefix else name, text)
    return tree

  # -- queries ------------------------------------------------------------

  def get(self, path: str) -> str | None:
    """Return the source text at *path*, or ``None``."""
    return self.files.get(str(PurePosixPath(path)))

  def exists(self, path: str) -> bool:
    """Return ``True`` if *path* is present."""
    return str(PurePosixPath(path)) in self.files

  def paths(self) -> list[str]:
    """Return all file paths, sorted."""
    return sorted(self.files)

  def is_package_dir(self, directory: str) -> bool:
    """Return ``True`` if *directory* contains an ``__init__.py``."""
    return str(PurePosixPath(directory)) in self.package_dirs

  # -- module index -------------------------------------------------------

  def _candidate_names(self, path: str) -> list[tuple[int, str]]:
    """Return ``(score, dotted_name)`` candidates for a file path.

    Under PEP 420 any directory chain is importable, so a file maps to several
    dotted names. ``score`` biases resolution towards regular packages (whose
    components all have ``__init__.py``).
    """
    parts = PurePosixPath(path).parts
    stem = PurePosixPath(path).stem
    candidates: list[tuple[int, str]] = []
    for start in range(len(parts)):
      tail = list(parts[start:-1])
      mod_parts = tail + ([] if stem == "__init__" else [stem])
      if not mod_parts:
        continue
      dotted = ".".join(mod_parts)
      # Score: +1 per leading component that is a regular package dir.
      score = 0
      for i in range(len(mod_parts) - 1):
        prefix = "/".join(parts[start:start + i + 1])
        if prefix in self.package_dirs:
          score += 1
        else:
          break
      candidates.append((score, dotted))
    return candidates

  def module_index(self) -> dict[str, list[tuple[int, str]]]:
    """Build (lazily) a mapping of dotted module name -> ``[(score, path)]``."""
    if self._index is None:
      index: dict[str, list[tuple[int, str]]] = {}
      for path in self.files:
        for score, dotted in self._candidate_names(path):
          index.setdefault(dotted, []).append((score, path))
      self._index = index
    return self._index

  def resolve_module(self, dotted: str) -> str | None:
    """Resolve a dotted module name to a repo file path, or ``None``.

    Prefers regular packages, then the shortest / lexicographically first path.
    """
    entries = self.module_index().get(dotted)
    if not entries:
      return None
    # Highest score wins; tie-break by path length then name for determinism.
    best_score = max(score for score, _ in entries)
    paths = sorted(path for score, path in entries if score == best_score)
    return paths[0] if paths else None

  def has_module(self, dotted: str) -> bool:
    """Return ``True`` if *dotted* names a module or package in the tree."""
    return dotted in self.module_index()

  def submodules_of(self, dotted: str) -> list[str]:
    """Return direct child module names of a package prefix (e.g. ``pkg`` -> ``core``)."""
    prefix = dotted + "."
    children: set[str] = set()
    for name in self.module_index():
      if name.startswith(prefix):
        rest = name[len(prefix):]
        children.add(rest.split(".", 1)[0])
    return sorted(children)


def _common_prefix(names) -> str:
  """Return the archive's single top-level directory (with trailing slash).

  GitHub (and most) source tarballs wrap all files under one directory named
  after the repo + ref; we strip exactly that first component. More than one
  top-level directory means the archive is already flat, so nothing is stripped.
  """
  names = list(names)
  if not names:
    return ""
  first_components = {name.split("/", 1)[0] for name in names if "/" in name}
  if len(first_components) != 1:
    return ""
  # Every file must actually be nested under the shared top directory.
  if any(name.count("/") == 0 for name in names):
    return ""
  return next(iter(first_components)) + "/"


__all__ = ["SourceTree"]
