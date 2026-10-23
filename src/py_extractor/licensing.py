"""License detection for vendored source.

When code is flattened out of one or more repositories, the resulting file is a
derivative work of every source repository it borrows from. This module
identifies the license of each contributing repository so the emitter can
surface a consolidated attribution table.

Detection prefers *authoritative* metadata (a GitHub API SPDX id, PyPI
``License ::`` classifiers or the ``info.license`` field, a Hugging Face
model-card ``license`` field) and only falls back to fingerprinting the text of
a ``LICENSE``/``COPYING``/``NOTICE`` file found in the downloaded tree. Nothing
here raises: an unrecognised license degrades to a plain "Unknown license"
entry rather than failing the extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Recognised license/notice file basenames (lower-cased). A path is treated as
# a license file when its basename matches one of these, or when its stem is a
# known prefix (so ``LICENSE-MIT``, ``LICENSE.apache`` and ``COPYING.LESSER``
# are all captured).
_LICENSE_BASENAMES = frozenset({
    "license",
    "license.txt",
    "license.md",
    "license.rst",
    "licence",
    "licence.txt",
    "licence.md",
    "copying",
    "copying.txt",
    "copying.md",
    "notice",
    "notice.txt",
    "notice.md",
    "unlicense",
    "unlicense.txt",
})

_LICENSE_STEMS = frozenset({"license", "licence", "copying", "notice"})

_COPYRIGHT_RE = re.compile(r"copyright[^\n]{0,200}", re.IGNORECASE)


def is_license_path(path: str) -> bool:
  """Return ``True`` if *path* looks like a license/notice file."""
  name = path.rsplit("/", 1)[-1].lower()
  if name in _LICENSE_BASENAMES:
    return True
  # Strip any extension and variant suffix (``LICENSE-MIT``, ``COPYING.LESSER``)
  # so ``license``/``copying``/``notice`` prefixes are all recognised.
  stem = re.split(r"[.\-]", name, maxsplit=1)[0]
  return stem in _LICENSE_STEMS


@dataclass(frozen=True)
class LicenseInfo:
  """The license attributed to one contributing repository.

  Attributes:
    repo_id: The repository slug the license belongs to.
    name: Human-readable license name (e.g. ``"MIT License"``).
    spdx: SPDX identifier when known (e.g. ``"MIT"``), else ``None``.
    origin: How the license was determined (``"github-api"``,
      ``"pypi-metadata"``, ``"classifier"``, ``"hf-model-card"``,
      ``"file-text"`` or ``"unknown"``).
    path: Repo-relative path of the license file, if one was found.
    copyright: A short copyright line extracted from the license text, if any.
    text: The full license text, when available (used for opt-in embedding).
  """

  repo_id: str
  name: str
  spdx: str | None = None
  origin: str = "unknown"
  path: str = ""
  copyright: str | None = None
  text: str = ""


# Ordered fingerprints: the first entry whose *every* needle appears in the
# (lower-cased) license text wins. More specific licenses must precede the more
# general ones they are built on (AGPL/LGPL before GPL, BSD-3 before BSD-2).
_FINGERPRINTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("AGPL-3.0-only", "GNU Affero General Public License v3.0", ("affero general public license",
                                                                 "version 3")),
    ("LGPL-3.0-only", "GNU Lesser General Public License v3.0", ("lesser general public license",
                                                                 "version 3")),
    ("LGPL-2.1-only", "GNU Lesser General Public License v2.1", ("lesser general public license",
                                                                 "version 2.1")),
    ("GPL-3.0-only", "GNU General Public License v3.0", ("gnu general public license",
                                                         "version 3, 29 june 2007")),
    ("GPL-2.0-only", "GNU General Public License v2.0", ("gnu general public license",
                                                         "version 2, june 1991")),
    ("Apache-2.0", "Apache License 2.0", ("apache license", "version 2.0, january 2004")),
    ("MPL-2.0", "Mozilla Public License 2.0", ("mozilla public license version 2.0",)),
    ("MIT", "MIT License", ("permission is hereby granted, free of charge",
                            'the software is provided "as is"')),
    ("BSD-3-Clause", "BSD 3-Clause License", ("redistribution and use in source and binary forms",
                                              "neither the name of")),
    ("BSD-2-Clause", "BSD 2-Clause License",
     ("redistribution and use in source and binary forms",)),
    ("ISC", "ISC License", ("permission to use, copy, modify, and/or distribute this software",)),
    ("Unlicense", "The Unlicense",
     ("this is free and unencumbered software released into the public domain",)),
    ("CC0-1.0", "Creative Commons Zero v1.0 Universal", ("cc0 1.0 universal",)),
)

_SPDX_NAMES: dict[str, str] = {spdx: name for spdx, name, _ in _FINGERPRINTS}

# Free-text license strings (PyPI ``info.license``, GitHub API, HF cards) mapped
# onto SPDX identifiers. Keys are lower-cased and stripped.
_SPDX_ALIASES: dict[str, str] = {
    "mit": "MIT",
    "mit license": "MIT",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "bsd": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "isc": "ISC",
    "isc license": "ISC",
    "mpl-2.0": "MPL-2.0",
    "mozilla public license 2.0": "MPL-2.0",
    "gpl-2.0": "GPL-2.0-only",
    "gpl-2.0-only": "GPL-2.0-only",
    "gpl-3.0": "GPL-3.0-only",
    "gpl-3.0-only": "GPL-3.0-only",
    "lgpl-3.0": "LGPL-3.0-only",
    "agpl-3.0": "AGPL-3.0-only",
    "unlicense": "Unlicense",
    "the unlicense": "Unlicense",
    "cc0-1.0": "CC0-1.0",
    "cc0 1.0 universal": "CC0-1.0",
}


def resolve_hint(hint: str) -> tuple[str | None, str]:
  """Map a free-text license *hint* to ``(spdx, name)``.

  Returns ``(None, hint)`` when the hint is not a recognised license string, so
  the caller can still display the raw value.
  """
  key = (hint or "").strip().lower()
  if not key:
    return None, "Unknown license"
  spdx = _SPDX_ALIASES.get(key)
  if spdx is None:
    for candidate, name in _SPDX_NAMES.items():
      if key == candidate.lower():
        spdx, key = candidate, name
        break
  if spdx is not None:
    return spdx, _SPDX_NAMES.get(spdx, hint.strip())
  return None, hint.strip()


def _match_license_text(text: str) -> tuple[str | None, str]:
  """Fingerprint *text* against the known licenses, best-effort."""
  lowered = text.lower()
  for spdx, name, needles in _FINGERPRINTS:
    if all(needle in lowered for needle in needles):
      return spdx, name
  return None, "Unknown license"


def _extract_copyright(text: str) -> str | None:
  """Return the first ``Copyright ...`` line in *text*, if any."""
  match = _COPYRIGHT_RE.search(text)
  if match is None:
    return None
  line = match.group(0).strip()
  return line or None


def _select_license_file(files: dict[str, str]) -> tuple[str, str]:
  """Return the ``(path, text)`` of the most representative license file.

  Prefers the shallowest path (a top-level ``LICENSE``) and breaks ties by path
  length then lexical order, so detection is deterministic.
  """
  if not files:
    return "", ""
  path = min(files, key=lambda p: (p.count("/"), len(p), p))
  return path, files[path]


def pypi_license(info: dict) -> tuple[str | None, str]:
  """Extract a license hint and its origin from a PyPI ``info`` mapping.

  Prefers the structured ``License ::`` classifier over the free-text
  ``license`` field. Returns ``(None, "")`` when nothing usable is present.
  """
  classifiers = info.get("classifiers") or []
  for classifier in classifiers:
    if isinstance(classifier, str) and classifier.startswith("License ::"):
      leaf = classifier.rsplit("::", 1)[-1].strip()
      if leaf and leaf.lower() != "unknown":
        return leaf, "classifier"
  license_field = (info.get("license") or "").strip()
  if license_field and license_field.lower() != "unknown" and "\n" not in license_field and \
      len(license_field) <= 120:
    return license_field, "pypi-metadata"
  return None, ""


def detect_license(tree, repo_id: str) -> LicenseInfo:
  """Determine the :class:`LicenseInfo` for *tree* (a ``SourceTree``).

  Authoritative metadata attached to the tree wins; otherwise the text of any
  license file is fingerprinted. Always returns a :class:`LicenseInfo`.
  """
  path, text = _select_license_file(getattr(tree, "license_files", {}) or {})
  copyright_line = _extract_copyright(text) if text else None

  hint = getattr(tree, "license_hint", None)
  origin = getattr(tree, "license_hint_origin", "") or "metadata"
  if hint:
    spdx, name = resolve_hint(hint)
    return LicenseInfo(
        repo_id=repo_id,
        name=name,
        spdx=spdx,
        origin=origin,
        path=path,
        copyright=copyright_line,
        text=text,
    )

  if text:
    spdx, name = _match_license_text(text)
    return LicenseInfo(
        repo_id=repo_id,
        name=name,
        spdx=spdx,
        origin="file-text",
        path=path,
        copyright=copyright_line,
        text=text,
    )

  return LicenseInfo(repo_id=repo_id, name="Unknown license", origin="unknown")


__all__ = [
    "LicenseInfo",
    "detect_license",
    "is_license_path",
    "pypi_license",
    "resolve_hint",
]
