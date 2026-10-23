"""Static analysis: import parsing, symbol discovery and resolution."""

from __future__ import annotations

from .imports import ImportedName, ImportInfo, collect_imports, resolve_relative
from .resolver import ResolvedSymbol, resolve_import_name
from .symbols import (
    ModuleSymbols,
    collect_module_symbols,
    find_definition,
)

__all__ = [
    "ImportInfo",
    "ImportedName",
    "ModuleSymbols",
    "ResolvedSymbol",
    "collect_imports",
    "collect_module_symbols",
    "find_definition",
    "resolve_import_name",
    "resolve_relative",
]
