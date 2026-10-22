"""Repository references and archive/tree URL construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import RepoNotFoundError

# Maps a Hugging Face URL/API path segment to the internal repo-type name.
_HF_TYPE_SEGMENTS = {
    "models": "model",
    "datasets": "dataset",
    "spaces": "space",
}


@dataclass(frozen=True)
class RepoRef:
  """A reference to a remote or local repository.

  Attributes:
    kind: One of ``"github"``, ``"hf"`` or ``"local"``.
    owner: Repository owner (for github) or namespace (for hf).
    name: Repository name.
    ref: Git ref (branch, tag or commit). Ignored for ``local``. The default
      ``"HEAD"`` resolves to the repository's default branch.
    path: Filesystem path (only for ``local``).
    repo_type: Hugging Face repository type (``"model"``, ``"dataset"`` or
      ``"space"``). Ignored for github/local. Selects the correct API and
      resolve URL prefixes.
  """

  kind: str
  owner: str
  name: str
  ref: str = "HEAD"
  path: Path | None = None
  repo_type: str = "model"

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
    if self.kind == "hf":
      return f"hf-{self.repo_type}-{self.owner}-{self.name}-{self.ref}"
    return f"{self.kind}-{self.owner}-{self.name}-{self.ref}"

  def tarball_url(self) -> str:
    """Return the archive URL for this repository (GitHub only)."""
    if self.kind == "github":
      return f"https://codeload.github.com/{self.owner}/{self.name}/tar.gz/{self.ref}"
    raise RepoNotFoundError(f"{self.kind!r} repositories have no tarball URL")

  @property
  def _hf_api_segment(self) -> str:
    """Return the API path segment for this HF repo type (``models`` etc.)."""
    return {"model": "models", "dataset": "datasets", "space": "spaces"}[self.repo_type]

  @property
  def _hf_web_prefix(self) -> str:
    """Return the web path prefix for this HF repo type.

    Models are served at the root (no prefix); datasets and spaces are served
    under ``datasets/`` and ``spaces/`` respectively.
    """
    return {"model": "", "dataset": "datasets/", "space": "spaces/"}[self.repo_type]

  def tree_api_url(self) -> str:
    """Return the tree-listing API URL (Hugging Face only)."""
    if self.kind == "hf":
      repo_id = f"{self.owner}/{self.name}"
      return (f"https://huggingface.co/api/{self._hf_api_segment}/{repo_id}"
              f"/tree/{self.ref}?recursive=true")
    raise RepoNotFoundError(f"{self.kind!r} repositories have no tree API")

  def resolve_url(self, path: str) -> str:
    """Return a per-file raw download URL (Hugging Face only)."""
    if self.kind == "hf":
      repo_id = f"{self.owner}/{self.name}"
      return (f"https://huggingface.co/{self._hf_web_prefix}{repo_id}"
              f"/resolve/{self.ref}/{path}")
    raise RepoNotFoundError(f"{self.kind!r} repositories have no per-file resolve URL")


def parse_repo(spec: str, ref: str = "HEAD") -> RepoRef:
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
    ref: Default git ref to use when the spec does not include one. The default
      ``"HEAD"`` is a symbolic ref that GitHub and Hugging Face both resolve to
      the repository's *actual* default branch, so ``master``-era repositories
      work without the caller having to know the branch name.

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
    repo_type = "model"
    # A leading type segment (hf:datasets/owner/name) is only recognised when
    # an owner/name pair follows, so a 2-segment spec is never misread.
    if len(parts) == 3 and parts[0].lower() in _HF_TYPE_SEGMENTS:
      repo_type = _HF_TYPE_SEGMENTS[parts[0].lower()]
      parts = parts[1:]
    if len(parts) != 2 or not all(parts):
      raise RepoNotFoundError(f"Invalid hf spec: {spec!r}")
    return RepoRef("hf", parts[0], parts[1], ref, repo_type=repo_type)

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


class HFRepoType:
  """String constants for Hugging Face repository types."""

  MODEL = "model"
  DATASET = "dataset"
  SPACE = "space"


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
  # Hugging Face URLs may include /models/, /datasets/ or /spaces/.
  repo_type = "model"
  if len(segments) >= 3 and segments[0].lower() in _HF_TYPE_SEGMENTS:
    repo_type = _HF_TYPE_SEGMENTS[segments[0].lower()]
    segments = segments[1:]
  if len(segments) < 2:
    raise RepoNotFoundError(f"Invalid Hugging Face URL: {url!r}")
  return RepoRef("hf", segments[0], segments[1], ref, repo_type=repo_type)


__all__ = ["HFRepoType", "RepoRef", "RepoRefKind", "parse_repo"]
