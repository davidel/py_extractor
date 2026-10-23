"""Tests for license detection and SPDX resolution."""

from __future__ import annotations

from py_extractor.licensing import (
    detect_license,
    is_license_path,
    pypi_license,
    resolve_hint,
)
from py_extractor.source.tree import SourceTree

MIT_TEXT = """MIT License

Copyright (c) 2021 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction.

The software is provided "as is", without warranty of any kind.
"""

APACHE_TEXT = """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright 2019 Other Inc.
"""

BSD3_TEXT = """Copyright (c) 2018, Someone

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Neither the name of the copyright holder may be used to endorse or promote.
"""


def _tree(files: dict[str, str], **kwargs) -> SourceTree:
  tree = SourceTree.from_mapping(files)
  for key, value in kwargs.items():
    setattr(tree, key, value)
  return tree


def test_is_license_path_matches_common_names() -> None:
  assert is_license_path("LICENSE")
  assert is_license_path("LICENSE.txt")
  assert is_license_path("docs/LICENSE.md")
  assert is_license_path("LICENSE-MIT")
  assert is_license_path("COPYING")
  assert is_license_path("NOTICE")
  assert not is_license_path("license_utils.py")
  assert not is_license_path("README.md")


def test_detect_mit_from_file_text() -> None:
  tree = _tree({"LICENSE": MIT_TEXT, "pkg/mod.py": "x = 1\n"})
  info = detect_license(tree, "owner/repo")
  assert info.spdx == "MIT"
  assert info.name == "MIT License"
  assert info.origin == "file-text"
  assert info.path == "LICENSE"
  assert info.copyright == "Copyright (c) 2021 The Authors"


def test_detect_apache_from_file_text() -> None:
  tree = _tree({"LICENSE.txt": APACHE_TEXT})
  info = detect_license(tree, "owner/repo")
  assert info.spdx == "Apache-2.0"
  assert info.copyright == "Copyright 2019 Other Inc."


def test_detect_bsd3_from_file_text() -> None:
  tree = _tree({"LICENSE": BSD3_TEXT})
  info = detect_license(tree, "owner/repo")
  assert info.spdx == "BSD-3-Clause"


def test_detect_unknown_license_text() -> None:
  tree = _tree({"LICENSE": "Do whatever you want, but not really.\n"})
  info = detect_license(tree, "owner/repo")
  assert info.spdx is None
  assert info.name == "Unknown license"
  assert info.origin == "file-text"
  assert info.path == "LICENSE"


def test_detect_no_license_file() -> None:
  tree = _tree({"pkg/mod.py": "x = 1\n"})
  info = detect_license(tree, "owner/repo")
  assert info.spdx is None
  assert info.name == "Unknown license"
  assert info.origin == "unknown"
  assert info.path == ""


def test_authoritative_hint_wins_over_text() -> None:
  tree = _tree({"LICENSE": "some unrelated text\n"},
               license_hint="Apache-2.0",
               license_hint_origin="github-api")
  info = detect_license(tree, "owner/repo")
  assert info.spdx == "Apache-2.0"
  assert info.origin == "github-api"


def test_shallowest_license_file_preferred() -> None:
  tree = _tree({"docs/LICENSE": MIT_TEXT, "LICENSE": APACHE_TEXT})
  info = detect_license(tree, "owner/repo")
  assert info.path == "LICENSE"
  assert info.spdx == "Apache-2.0"


def test_resolve_hint_variants() -> None:
  assert resolve_hint("MIT")[0] == "MIT"
  assert resolve_hint("apache 2.0")[0] == "Apache-2.0"
  assert resolve_hint("BSD-3-Clause")[0] == "BSD-3-Clause"
  assert resolve_hint("Weird License") == (None, "Weird License")
  assert resolve_hint("")[0] is None


def test_pypi_license_prefers_classifier() -> None:
  info = {"classifiers": ["License :: OSI Approved :: MIT License"], "license": "Apache-2.0"}
  hint, origin = pypi_license(info)
  assert hint == "MIT License"
  assert origin == "classifier"


def test_pypi_license_falls_back_to_field() -> None:
  hint, origin = pypi_license({"license": "BSD-3-Clause"})
  assert hint == "BSD-3-Clause"
  assert origin == "pypi-metadata"


def test_pypi_license_ignores_unknown() -> None:
  assert pypi_license({"license": "UNKNOWN"}) == (None, "")
  assert pypi_license({"classifiers": ["License :: OSI Approved :: Unknown"]}) == (None, "")


def test_license_files_are_not_in_module_index() -> None:
  tree = _tree({"LICENSE": MIT_TEXT, "pkg/__init__.py": "", "pkg/mod.py": "x = 1\n"})
  assert "LICENSE" not in tree.paths()
  assert tree.license_files == {"LICENSE": MIT_TEXT}
