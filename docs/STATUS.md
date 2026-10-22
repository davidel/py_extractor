# py_extractor — Status & Open Investigations

_Last updated: working note captured mid-investigation._

This file tracks in-progress work and open questions so the context is not lost
between sessions. Completed work is recorded in the git log (see the bottom).

---

## Current focus: trusted-package registry vs. in-repo resolution

### Request
> "tensorflow, jax, tensor2tensor, ... should all be whitelisted packages"

### What is already true
`tensorflow`, `keras`, `jax`, `jaxlib`, `flax`, `haiku` **are already** on the
allow-list (`src/py_extractor/data/trusted_packages.toml`, section
"Deep learning / ML frameworks & companions"). So the literal request is partly
already satisfied.

### What actually triggered the request
While stress-testing against real papers-with-code repos, extracting
`github:tensorflow/tensor2tensor` → `Transformer` produced a **55,686-line,
673-unit** file. The blow-up was **not** `tensorflow` (already trusted) but two
packages that are **not** trusted and therefore got inlined:

| Imported root | Trusted? | Effect |
| --- | --- | --- |
| `six` | ❌ no | 27 units inlined from `benjaminp/six` |
| `tensorflow_probability` | ❌ no | 5 units inlined from `tensorflow/probability` |
| `absl` | ❌ no | flagged (unresolved third-party) |
| `tensor2tensor` (the source repo itself) | ❌ no | 641 units inlined — correct, it's the repo we're extracting from |

### Missing from the registry (audit result)
```
MISSING: tensorflow_probability, tensorboard, tf_keras, keras_cv,
         tensor2tensor, tensorflow_datasets, tensorflow_hub, tfds, sonnet,
         absl, six, gast, opt_einsum, flatbuffers, termcolor, wrapt,
         dm_haiku, optax, chex, orbax, equinox, jaxtyping, diffrax,
         ml_collections, ml_dtypes
```

### ⚠️ The subtlety that blocks a naive fix (KEY FINDING)
`_handle_import` (`src/py_extractor/extract/closure.py:468`) checks the trust
classification **before** in-repo resolution:

```python
root = imp.root
decision = classify(root, self.policy)
if decision.route in (Route.STDLIB, Route.TRUSTED):
    self._keep_import(imp, imported)   # <-- short-circuits
    return
# ...only now: is it an in-repo module?
internal = resolve_import_name(tree, imp, imported.name, ...)
```

Consequence: **adding a package to the allow-list breaks extraction *from* that
package's own repository.** If `tensor2tensor` were whitelisted, its
`from tensor2tensor.layers import common_layers` self-imports would be *kept as
imports* instead of inlined → the output is no longer self-contained.

Measured (synthetic + real):

| Case | Trusted | Result |
| --- | --- | --- |
| extract `tensor2tensor:Transformer` | `{}` (current) | 55,686 lines, self-contained ✅ |
| extract `tensor2tensor:Transformer` | `{"tensor2tensor"}` | **1,648 lines, NOT self-contained** ❌ |
| extract `keras:Dense` | `{}` (current, keras IS trusted) | 382 lines, **NOT self-contained** ❌ (already broken) |

Note the **`keras` case is already broken today** because `keras` is on the
allow-list and keras uses absolute self-imports (`from keras.src... import ...`).
So this is a **pre-existing bug**, not just a hypothetical.

### Why `requests` looks fine but `keras` does not
- `psf/requests` uses **relative** self-imports (`from .models import ...`).
  Relative imports resolve via `resolve_import_name` using the *real* tree, and
  the dispatch keys on the imported **name** (e.g. `BaseAdapter`) — so they
  inline correctly despite `requests` being trusted.
- `keras` / `tensor2tensor` use **absolute** self-imports (`from keras.src...`,
  `from tensor2tensor.layers...`). Here `imp.root == "keras"` → `classify`
  returns `TRUSTED` → short-circuit → kept as an import → **not self-contained**.

### Proposed fix (NOT yet implemented)
Reorder the precedence in `_handle_import` so **in-repo resolution wins over
`TRUSTED`** (stdlib still wins outright):

```
1. stdlib            -> keep import
2. resolve in-repo   -> if found, inline (recurse)          <-- moved up
3. trusted           -> keep import
4. external resolver -> inline from PyPI/GitHub if found
5. else              -> keep import + flag
```

Rationale: a module that physically exists in the tree being extracted is
definitionally part of the code under extraction — it should always be inlined,
regardless of whether its top-level name also happens to be on the allow-list.
The allow-list should only govern **external** dependencies.

### ⚠️ Risk of the proposed fix (measured)
Blindly applying "in-repo first" **explodes** on frameworks whose source tree is
huge and self-referential:

| Case | Current | "repo_first" |
| --- | --- | --- |
| `keras:Dense` | 382 lines, 1 unit | **70,388 lines, 1,188 units** |
| `tensor2tensor:Transformer` | 54,709 lines, 641 units | 54,709 lines, 641 units (no change) |

So "in-repo first" fixes self-containment but can drag in an entire framework
(keras 1,188 units). The fix therefore needs a **guard**, e.g.:
- only inline in-repo when the root module equals the **source repo's own
  package root** (self-import), not any coincidental name collision; and/or
- a size/units cap (`--max-units` / `--max-lines`) that stops runaway inlining
  and emits a flag instead.

**Decision needed before implementing.** Options:
1. Reorder precedence + restrict to self-imports (root == repo package) — likely
   correct, needs care with the `src/` layout and namespace packages.
2. Reorder precedence + add a runaway cap with a clear flag.
3. Only extend the registry (add `six`, `tensorflow_probability`, `absl`, ...),
   and leave the precedence bug for a dedicated change.

### Recommendation
Split into **two independent changes**:
- **(A) Registry expansion** — low risk, immediate value: add the genuinely
  missing ecosystem packages (`six`, `absl`, `tensorflow_probability`,
  `tensorboard`, `tf_keras`, `tensorflow_datasets`, `tensorflow_hub`, `tfds`,
  `sonnet`, `optax`, `chex`, `orbax`, `dm_haiku`, `ml_collections`, `ml_dtypes`,
  `jaxtyping`, ...). This alone shrinks the tensor2tensor blow-up.
- **(B) Precedence fix** — in-repo beats `TRUSTED`, **with** a self-import
  restriction and/or runaway cap. Higher risk; needs its own tests.

Do **not** add `tensor2tensor` to the allow-list without (B), or it will break
extraction from the tensor2tensor repo itself.

---

## Reproduction commands (scratch harnesses, gitignored)
All under `.chatty/scratch/`:
- `paper_models_harness.py` — extract flagship model classes from landmark repos.
- `real_repo_harness.py` — auto-discover + extract every top-level symbol.
- `precedence_measure.py` — compares `current` vs `repo_first` ordering.
- `rel_vs_abs.py`, `collision_probe.py` — relative vs absolute self-imports.
- `registry_audit.py` — lists which ecosystem packages are trusted vs missing.

Run, e.g.:
```bash
python3 .chatty/scratch/registry_audit.py
python3 .chatty/scratch/precedence_measure.py
```

---

## Verified completed work (git log)
```
1f3fc6e  Make symbol-not-found errors actionable
410af85  Handle UTF-8 BOM and default to the repository's real default branch
22116fb  Preserve original source text (comments, blank lines, formatting) in output
7829709  Emit from __future__ imports first so they are never dropped
8284ee5  Preserve optional-import guards; stop hoisting local imports
c88cb84  Bind inlined definitions back to the local names call sites use
9d1b7e6  Support Hugging Face datasets and spaces; broaden HF test/CI coverage
33ba0e1  Add output verification: full compile check + opt-in import smoke test
9a4ca65  Add py_extractor: single-file source extraction from remote repos
```

### Gates at time of writing
- Offline suite: **311 tests pass**; network suite: **7 tests pass**.
- `ruff check` clean; `yapf --diff` clean.
- Working tree clean (this file is the only pending addition).

### Known caveats (not bugs)
- Large, dependency-heavy targets can balloon (AlphaFold 12k lines, mmdetection
  ~10k lines). Flags + `--verify-import` are the honest signals.
- The trust/registry precedence issue documented above is the main open item.
