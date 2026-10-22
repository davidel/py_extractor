"""Human-readable and JSON rendering of an extraction result."""

from __future__ import annotations

import json

from .extract.closure import ClosureResult


def render_text(result: ClosureResult) -> str:
  """Render a closure result as a plain-text report."""
  lines: list[str] = []
  lines.append(f"Target: {result.target.module}:{result.target.symbol or '<module>'}")
  lines.append(f"  file: {result.target.path}")
  lines.append("")
  lines.append(f"Inlined definitions ({len(result.units)}):")
  for unit in result.units:
    label = f"{unit.module}:{unit.symbol}" if unit.symbol else unit.module
    lines.append(f"  - {label}  ({unit.repo_id}:{unit.path})")
  lines.append("")
  lines.append(f"Imports kept as-is ({len(result.kept_imports)}):")
  for imp in result.kept_imports:
    lines.append(f"  - {_render_import(imp)}")
  lines.append("")
  lines.append(f"Local bindings ({len(result.bindings)}):")
  for binding in result.bindings:
    if binding.kind == "alias":
      lines.append(f"  - {binding.local} = {binding.target}")
    else:
      path = ".".join((binding.local, *binding.parts))
      lines.append(f"  - {path} -> namespace({', '.join(binding.members)})")
  lines.append("")
  lines.append(f"Future imports ({len(result.future_imports)}):")
  for imp in result.future_imports:
    lines.append(f"  - {_render_import(imp)}")
  lines.append("")
  lines.append(f"Optional-import blocks preserved ({len(result.guarded_blocks)}):")
  for block in result.guarded_blocks:
    lines.append(f"  - {block.module}:{block.lineno}")
  lines.append("")
  if result.flags:
    lines.append(f"Warnings ({len(result.flags)}):")
    for flag in result.flags:
      location = f"  ({flag.location})" if flag.location else ""
      lines.append(f"  - [{flag.kind.value}] {flag.detail}{location}")
  else:
    lines.append("Warnings: none")
  return "\n".join(lines)


def render_json(result: ClosureResult) -> str:
  """Render a closure result as JSON."""
  payload = {
      "target": {
          "module": result.target.module,
          "symbol": result.target.symbol,
          "path": result.target.path,
      },
      "units": [{
          "module": unit.module,
          "symbol": unit.symbol,
          "path": unit.path,
          "repo": unit.repo_id,
      } for unit in result.units],
      "kept_imports": [{
          "kind": imp.kind,
          "module": imp.module,
          "name": imp.name,
          "asname": imp.asname,
      } for imp in result.kept_imports],
      "namespace_roots":
          result.namespace_roots,
      "bindings": [{
          "kind": binding.kind,
          "local": binding.local,
          "target": binding.target,
          "parts": list(binding.parts),
          "members": list(binding.members),
      } for binding in result.bindings],
      "guarded_blocks": [{
          "module": block.module,
          "lineno": block.lineno,
      } for block in result.guarded_blocks],
      "future_imports": [{
          "module": imp.module,
          "name": imp.name,
          "asname": imp.asname,
      } for imp in result.future_imports],
      "flags": [{
          "kind": flag.kind.value,
          "detail": flag.detail,
          "location": flag.location,
      } for flag in result.flags],
  }
  return json.dumps(payload, indent=2, sort_keys=True)


def _render_import(imp) -> str:
  if imp.kind == "import":
    return f"import {imp.module}" + (f" as {imp.asname}" if imp.asname else "")
  suffix = f" as {imp.asname}" if imp.asname else ""
  return f"from {imp.module} import {imp.name}{suffix}"


__all__ = ["render_json", "render_text"]
