"""Typed exceptions for :mod:`py_extractor`."""

from __future__ import annotations


class PyExtractorError(Exception):
  """Base class for all py_extractor errors."""


class SourceError(PyExtractorError):
  """Raised when a repository or file cannot be fetched / decoded."""


class RepoNotFoundError(SourceError):
  """The requested repository, ref or file does not exist."""


class SymbolNotFoundError(PyExtractorError):
  """The requested class/function could not be located in the repository."""


class ResolutionError(PyExtractorError):
  """An import could not be resolved to source code."""


class AmbiguousSymbolError(PyExtractorError):
  """A symbol name resolves to more than one definition and no hint disambiguates it."""


class ExtractionError(PyExtractorError):
  """The final single-file output could not be assembled/validated."""
