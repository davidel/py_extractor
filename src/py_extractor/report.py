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
    lines.append(f"  - {imp.render()}")
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
    lines.append(f"  - {imp.render()}")
  lines.append("")
  lines.append(f"Optional-import blocks preserved ({len(result.guarded_blocks)}):")
  for block in result.guarded_blocks:
    lines.append(f"  - {block.module}:{block.lineno}")
  lines.append("")
  lines.append(f"Licenses ({len(result.licenses)}):")
  for license_info in result.licenses:
    label = result.license_ids.get(license_info.repo_id, "")
    spdx = f" (SPDX: {license_info.spdx})" if license_info.spdx else ""
    origin = f"  [{license_info.origin}]" if license_info.origin else ""
    lines.append(f"  - [{label}] {license_info.name}{spdx}  "
                 f"({license_info.repo_id}){origin}")
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
      "licenses": [{
          "id": result.license_ids.get(license_info.repo_id, ""),
          "repo": license_info.repo_id,
          "name": license_info.name,
          "spdx": license_info.spdx,
          "origin": license_info.origin,
          "path": license_info.path,
          "copyright": license_info.copyright,
      } for license_info in result.licenses],
  }
  return json.dumps(payload, indent=2, sort_keys=True)


__all__ = ["render_json", "render_text"]
