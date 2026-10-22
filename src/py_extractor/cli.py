"""Command-line interface for py_extractor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, extract_symbol
from .config import ExtractOptions, TrustPolicy
from .errors import AmbiguousSymbolError, PyExtractorError
from .report import render_json, render_text
from .verify import compile_file, compile_source_result, import_symbol


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
                      help="Repo-relative reference file. Overrides the symbol search "
                      "(use this instead of the interactive prompt for scripts/CI).")
  parser.add_argument("--ref",
                      default="HEAD",
                      help="Git ref (branch/tag/commit). Default: HEAD, which resolves to the "
                      "repository's default branch.")
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
  behavior.add_argument("-y",
                        "--yes",
                        action="store_true",
                        help="Never prompt; fail if a symbol is ambiguous (for scripts/CI).")
  verify = parser.add_argument_group("verification")
  verify.add_argument("--verify-compile",
                      action="store_true",
                      help="Byte-compile the written --out file and report the result.")
  verify.add_argument("--verify-import",
                      action="store_true",
                      help="Import the generated file in a subprocess to confirm the symbol "
                      "is usable (executes module-level code; opt-in).")
  verify.add_argument("--verify-timeout",
                      type=float,
                      default=60.0,
                      help="Timeout in seconds for --verify-import. Default: 60.")
  behavior.add_argument("--version", action="version", version=f"py_extractor {__version__}")
  return parser


def _split(value: str) -> set[str]:
  return {part.strip() for part in value.split(",") if part.strip()}


def _make_chooser(*, interactive: bool):
  """Return a ``choose`` callback for symbol disambiguation.

  When *interactive* is ``False`` the callback declines (``None``), letting the
  library raise :class:`AmbiguousSymbolError`; the CLI then prints the candidate
  list and exits. This keeps scripts/CI from ever hanging on a prompt.
  """

  def choose(symbol: str, candidates: list[str]) -> str | None:
    if not interactive:
      return None
    print(f"Symbol {symbol!r} is defined in multiple files:", file=sys.stderr)
    for index, path in enumerate(candidates, start=1):
      marker = "  [default]" if index == 1 else ""
      print(f"  {index}) {path}{marker}", file=sys.stderr)
    while True:
      try:
        answer = input("Which file is the reference? [1]: ").strip()
      except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
        return None
      if not answer:
        return candidates[0]
      if answer.isdigit() and 1 <= int(answer) <= len(candidates):
        return candidates[int(answer) - 1]
      print("Please enter a number from the list.", file=sys.stderr)

  return choose


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

  # Prompt for disambiguation only when attached to a real terminal and not
  # silenced with -y; otherwise decline so the library raises and we report.
  interactive = (not args.yes) and sys.stdin.isatty()
  chooser = _make_chooser(interactive=interactive)

  try:
    result = extract_symbol(
        args.repo,
        args.symbol,
        file_hint=args.file_hint,
        ref=args.ref,
        options=options,
        choose=chooser,
    )
  except AmbiguousSymbolError as exc:
    print(f"error: {exc}", file=sys.stderr)
    print("Re-run with --file <path> to choose the reference file.", file=sys.stderr)
    return 1
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

  return _run_verification(args, result)


def _run_verification(args, result) -> int:
  """Run the requested verification steps, returning the process exit code."""
  exit_code = 0

  if args.verify_compile:
    target = Path(args.out) if args.out else None
    outcome = (compile_file(target) if target is not None else compile_source_result(result.source))
    _report_verification("compile", outcome)
    if not outcome.ok:
      exit_code = 1

  if args.verify_import:
    outcome = import_symbol(result.target.symbol or "",
                            path=args.out,
                            source=result.source,
                            timeout=args.verify_timeout)
    _report_verification("import", outcome)
    if not outcome.ok:
      exit_code = 1

  return exit_code


def _report_verification(kind: str, outcome) -> None:
  status = "OK" if outcome.ok else "FAILED"
  print(f"verify-{kind}: {status} - {outcome.detail}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
  raise SystemExit(main())
