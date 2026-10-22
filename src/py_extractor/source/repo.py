"""Repository references and archive/tree URL construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import RepoNotFoundError


@dataclass(frozen=True)
class RepoRef:
  """A reference to a remote or local repository.

  Attributes:
    kind: One of ``"github"``, ``"hf"`` or ``"local"``.
    owner: Repository owner (for github) or namespace (for hf).
    name: Repository name.
    ref: Git ref (branch, tag or commit). Ignored for ``local``.
    path: Filesystem path (only for ``local``).
  """

  kind: str
  owner: str
  name: str
  ref: str = "main"
  path: Path | None = None

  @property
  def slug(self) -> str:
    """Return ``owner/name`` (or the local path as a string)."""
    if self.kind == "local":
      return str(self.path)
    return f"{self.owner}/{self.name}"

  @property
  def cache_key(self) -> str:
    """A filesystem-safe cache key."""
    if self.kind == "local":
      return f"local-{self.name}"
    return f"{self.kind}-{self.owner}-{self.name}-{self.ref}"

  def tarball_url(self) -> str:
    """Return the archive URL for this repository (GitHub only)."""
    if self.kind == "github":
      return f"https://codeload.github.com/{self.owner}/{self.name}/tar.gz/{self.ref}"
    raise RepoNotFoundError(f"{self.kind!r} repositories have no tarball URL")

  def tree_api_url(self) -> str:
    """Return the tree-listing API URL (Hugging Face only)."""
    if self.kind == "hf":
      repo_id = f"{self.owner}/{self.name}"
      return f"https://huggingface.co/api/models/{repo_id}/tree/{self.ref}?recursive=true"
    raise RepoNotFoundError(f"{self.kind!r} repositories have no tree API")

  def resolve_url(self, path: str) -> str:
    """Return a per-file raw download URL (Hugging Face only)."""
    if self.kind == "hf":
      repo_id = f"{self.owner}/{self.name}"
      return f"https://huggingface.co/{repo_id}/resolve/{self.ref}/{path}"
    raise RepoNotFoundError(f"{self.kind!r} repositories have no per-file resolve URL")


def parse_repo(spec: str, ref: str = "main") -> RepoRef:
  """Parse a repository specification into a :class:`RepoRef`.

  Accepted forms::

    github:owner/repo
    hf:owner/repo
    https://github.com/owner/repo
    https://huggingface.co/owner/repo
    ./local/path        (or any existing directory)
    owner/repo          (defaults to github)

  Args:
    spec: The repository specification string.
    ref: Default git ref to use when the spec does not include one.

  Returns:
    A parsed :class:`RepoRef`.

  Raises:
    RepoNotFoundError: If the specification cannot be parsed.
  """
  spec = spec.strip()
  lowered = spec.lower()

  if lowered.startswith("github:"):
    parts = spec[len("github:"):].split("/")
    if len(parts) != 2 or not all(parts):
      raise RepoNotFoundError(f"Invalid github spec: {spec!r}")
    return RepoRef("github", parts[0], parts[1], ref)

  if lowered.startswith("hf:"):
    parts = spec[len("hf:"):].split("/")
    if len(parts) != 2 or not all(parts):
      raise RepoNotFoundError(f"Invalid hf spec: {spec!r}")
    return RepoRef("hf", parts[0], parts[1], ref)

  if lowered.startswith("https://github.com/") or lowered.startswith("http://github.com/"):
    return _parse_url(RepoRefKind.GITHUB, spec, ref)

  if lowered.startswith("https://huggingface.co/") or lowered.startswith("http://huggingface.co/"):
    return _parse_url(RepoRefKind.HF, spec, ref)

  # Local directory?
  candidate = Path(spec).expanduser()
  if candidate.exists() and candidate.is_dir():
    return RepoRef("local", "", candidate.name, ref, candidate.resolve())

  # Bare owner/repo -> assume github.
  if spec.count("/") == 1 and " " not in spec:
    owner, name = spec.split("/")
    if owner and name:
      return RepoRef("github", owner, name, ref)

  raise RepoNotFoundError(f"Could not parse repository spec: {spec!r}")


class RepoRefKind:
  """String constants for repository kinds."""

  GITHUB = "github"
  HF = "hf"
  LOCAL = "local"


def _parse_url(kind: str, url: str, ref: str) -> RepoRef:
  without_scheme = url.split("://", 1)[1]
  host, _, rest = without_scheme.partition("/")
  segments = [seg for seg in rest.split("/") if seg]
  if kind == RepoRefKind.GITHUB:
    segments = [seg for seg in segments if seg not in {"tree", "blob"}]
    if len(segments) < 2:
      raise RepoNotFoundError(f"Invalid GitHub URL: {url!r}")
    owner, name = segments[0], segments[1]
    if name.endswith(".git"):
      name = name[:-len(".git")]
    return RepoRef("github", owner, name, ref)
  # Hugging Face URLs may include /models/ or /datasets/.
  if segments and segments[0] in {"models", "datasets", "spaces"}:
    segments = segments[1:]
  if len(segments) < 2:
    raise RepoNotFoundError(f"Invalid Hugging Face URL: {url!r}")
  return RepoRef("hf", segments[0], segments[1], ref)


__all__ = ["RepoRef", "RepoRefKind", "parse_repo"]
