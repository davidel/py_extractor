"""Command-line interface for py_extractor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, extract_symbol
from .config import ExtractOptions, TrustPolicy
from .errors import PyExtractorError
from .report import render_json, render_text


def build_parser() -> argparse.ArgumentParser:
  """Construct the argument parser."""
  parser = argparse.ArgumentParser(
      prog="py-extractor",
      description="Extract a self-contained, single-file copy of a class or function "
      "from a remote repository, vendoring only the code it needs.",
  )
  parser.add_argument("repo", help="Repository spec: github:owner/repo, a URL, or a local path.")
  parser.add_argument("symbol", help="Class/function name to extract (optionally Class.method).")
  parser.add_argument("--file",
                      dest="file_hint",
                      default=None,
                      help="Repo-relative path hint containing the symbol.")
  parser.add_argument("--ref", default="main", help="Git ref (branch/tag/commit). Default: main.")
  parser.add_argument("--out",
                      default=None,
                      help="Write the generated file here (default: stdout).")

  trust = parser.add_argument_group("trust policy")
  trust.add_argument("--trust", default="", help="Comma-separated extra trusted module roots.")
  trust.add_argument("--no-trust",
                     dest="untrusted",
                     default="",
                     help="Comma-separated trusted module roots to revoke.")
  trust.add_argument("--trust-file",
                     action="append",
                     default=[],
                     dest="trust_files",
                     help="TOML registry file with extra trusted packages (repeatable).")
  trust.add_argument("--strict-trust",
                     action="store_true",
                     help="Ignore the built-in curated registry (stdlib still trusted).")

  behavior = parser.add_argument_group("behavior")
  behavior.add_argument("--no-third-party",
                        action="store_true",
                        help="Never attempt to inline third-party code.")
  behavior.add_argument("--offline", action="store_true", help="Disallow all network access.")
  behavior.add_argument("--cache-dir", default=".py_extractor_cache", help="Download cache dir.")
  behavior.add_argument("--report",
                        choices=["text", "json", "none"],
                        default="text",
                        help="Emit a resolution report to stderr. Default: text.")
  behavior.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity.")
  behavior.add_argument("--version", action="version", version=f"py_extractor {__version__}")
  return parser


def _split(value: str) -> set[str]:
  return {part.strip() for part in value.split(",") if part.strip()}


def main(argv: list[str] | None = None) -> int:
  """CLI entry point. Returns a process exit code."""
  parser = build_parser()
  args = parser.parse_args(argv)

  policy = TrustPolicy.build(
      trusted=_split(args.trust),
      untrusted=_split(args.untrusted),
      registry_files=[Path(p) for p in args.trust_files],
      strict=args.strict_trust,
  )
  options = ExtractOptions(
      trust=policy,
      allow_third_party=not args.no_third_party,
      cache_dir=Path(args.cache_dir),
      offline=args.offline,
      verbosity=args.verbose,
  )

  try:
    result = extract_symbol(
        args.repo,
        args.symbol,
        file_hint=args.file_hint,
        ref=args.ref,
        options=options,
    )
  except PyExtractorError as exc:
    print(f"error: {exc}", file=sys.stderr)
    return 1

  if args.out:
    Path(args.out).write_text(result.source, encoding="utf-8")
    if args.report == "none":
      print(f"Wrote {args.out}", file=sys.stderr)
  else:
    sys.stdout.write(result.source)

  if args.report == "text":
    print(render_text(result.closure), file=sys.stderr)
  elif args.report == "json":
    print(render_json(result.closure), file=sys.stderr)
  return 0


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
