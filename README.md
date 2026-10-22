# py_extractor

Extract **one self-contained Python file** containing exactly the code a class or
function needs — pulled from a GitHub/Hugging Face repository (or a local checkout).

Use it when you want to reuse a function or class from a package you don't fully
trust, *without* importing the whole package and dragging in its dependencies.

## How it works

1. **Acquire the tree once.** The whole repository is downloaded as a tarball and
   materialised into an in-memory virtual filesystem, so imports are resolved
   against the *real* layout (no path guessing).
2. **Resolve imports properly.** Relative imports (`.`/`..`), `src/` layouts,
   namespace packages and `__init__.py` re-exports are all handled.
3. **Compute the transitive closure.** Free names are found with scope-aware AST
   analysis (locals, loop/comprehension variables, parameters and builtins are
   excluded; decorators, bases, annotations and defaults are included).
4. **Classify each dependency** (first match wins):
   - **Standard library** → kept as an `import` (always, automatically).
   - **Curated trusted package** (torch, numpy, transformers, ...) → kept as an `import`.
   - **In-repo module** → recursed into and inlined.
   - **Unknown third-party** → looked up on PyPI, optionally inlined, otherwise
     **flagged** and kept as an `import` (warn, but still emit a runnable file).
5. **Emit a single file** with the kept imports, the inlined definitions in
   dependency order, and a resolution report listing provenance and warnings.

## Install

```bash
pip install -e .            # runtime dep: requests
pip install -e '.[dev]'     # adds pytest, pytest-cov, ruff, yapf
```

## CLI

```bash
py-extractor github:owner/repo SymbolName
py-extractor https://github.com/pallets/flask Flask --out flask_flask.py
python -m py_extractor github:owner/repo SymbolName --report json
```

The symbol is discovered automatically by scanning the whole repository. If the
name is defined in **more than one file**, the CLI asks which one to use::

```text
Symbol 'Upsample' is defined in multiple files:
  1) basicsr/archs/dat_arch.py  [default]
  2) basicsr/archs/arch_util.py
Which file is the reference? [1]:
```

For scripts/CI, pass `--file PATH` to select the reference file directly, or
`-y/--yes` to fail fast (with the candidate list) instead of prompting. The
prompt is only shown on an interactive terminal; when stdin is not a TTY the run
errors out rather than hanging.

Key options:

| Option | Meaning |
| --- | --- |
| `--file PATH` | Reference file; overrides the symbol search (scripts/CI) |
| `--ref REF` | Git ref (branch/tag/commit) |
| `--out FILE` | Write output to a file (default: stdout) |
| `-y, --yes` | Never prompt; fail if the symbol is ambiguous |
| `--trust a,b` | Extra trusted top-level module names |
| `--no-trust a,b` | Revoke trusted names |
| `--trust-file FILE` | Extra TOML registry (`[packages]` table) |
| `--strict-trust` | Ignore the built-in registry (stdlib still trusted) |
| `--no-third-party` | Never attempt to inline third-party code |
| `--offline` | Disallow network access |
| `--report text\|json\|none` | Resolution report on stderr |

## Library

```python
from py_extractor import extract_symbol

result = extract_symbol(
    "github:owner/repo",
    "MyClass",
    file_hint="pkg/module.py",   # optional override (skips discovery)
    trusted={"torch", "numpy"},
)
print(result.source)        # the generated file
for flag in result.closure.flags:
    print(flag.kind.value, flag.detail)
```

When the symbol is ambiguous and no `file_hint` is given, `extract_symbol`
raises `AmbiguousSymbolError` (carrying the ranked `candidates`) unless you pass
a `choose` callback:

```python
from py_extractor import extract_symbol
from py_extractor.errors import AmbiguousSymbolError

try:
    result = extract_symbol("github:owner/repo", "Thing")
except AmbiguousSymbolError as exc:
    print(exc.candidates)               # ['pkg/one.py', 'pkg/two.py']

# ...or decide programmatically:
result = extract_symbol("github:owner/repo", "Thing",
                        choose=lambda symbol, candidates: candidates[1])
```
## Trusted packages

The curated registry lives in `src/py_extractor/data/trusted_packages.toml`
(import names, not distribution names). The Python standard library is always
trusted automatically and never needs to be listed.

## Tests

```bash
pytest                      # offline suite (network tests deselected)
pytest -m network           # real GitHub integration tests
```
