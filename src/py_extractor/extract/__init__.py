"""Extraction: classification, closure computation and emission."""

from __future__ import annotations

from .classifier import Classification, Route, classify
from .closure import (
    ClosureResult,
    ExtractedUnit,
    Flag,
    FlagKind,
    KeptImport,
    ThirdPartyResolver,
    compute_closure,
)
from .emitter import emit

__all__ = [
    "Classification",
    "ClosureResult",
    "ExtractedUnit",
    "Flag",
    "FlagKind",
    "KeptImport",
    "Route",
    "ThirdPartyResolver",
    "classify",
    "compute_closure",
    "emit",
]
