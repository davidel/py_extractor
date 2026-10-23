"""Extraction: closure computation and emission."""

from __future__ import annotations

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
    "ClosureResult",
    "ExtractedUnit",
    "Flag",
    "FlagKind",
    "KeptImport",
    "ThirdPartyResolver",
    "compute_closure",
    "emit",
]
