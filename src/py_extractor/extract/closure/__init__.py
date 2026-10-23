"""Transitive dependency closure.

Consumers import the concrete submodules directly:

* :mod:`.models` -- the value objects (flags, units, bindings, results).
* :mod:`.hoisting` -- import-map construction and ``__future__`` collection.
* :mod:`.guards` -- preservation of optional-import guard blocks.
* :mod:`.builder` -- the recursive closure walk and ``compute_closure``.

Like :mod:`py_extractor.analysis`, this package keeps its public surface minimal
by re-exporting nothing.
"""
