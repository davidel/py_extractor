"""Shared pytest fixtures: synthetic multi-module packages (offline)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from py_extractor.config import ExtractOptions, TrustPolicy
from py_extractor.source.tree import SourceTree

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def options(tmp_path: Path) -> ExtractOptions:
  """Extraction options isolated to a temp cache dir."""
  return ExtractOptions(cache_dir=tmp_path / "cache", trust=TrustPolicy())


@pytest.fixture
def basic_tree() -> SourceTree:
  """A small package exercising aliases, relative imports and re-exports."""
  return SourceTree.from_mapping({
      "mypkg/__init__.py": "from .core import Model\n__all__ = ['Model']\n",
      "mypkg/core.py": ("import numpy as np\n"
                        "from torch import nn\n"
                        "from .utils import SCALE\n"
                        "\n"
                        "class Model(nn.Module):\n"
                        "    def __init__(self):\n"
                        "        super().__init__()\n"
                        "        self.fc = nn.Linear(4, 4)\n"
                        "\n"
                        "    def forward(self, x):\n"
                        "        return self.fc(x) * SCALE + np.float32(1.0)\n"),
      "mypkg/utils.py": "SCALE = 2.0\n",
  })


@pytest.fixture
def relative_tree() -> SourceTree:
  """Deep relative imports with a regular package chain."""
  return SourceTree.from_mapping({
      "pkg/__init__.py": "",
      "pkg/a/__init__.py": "",
      "pkg/a/b/__init__.py": "",
      "pkg/a/b/mod.py": "from ..helpers import assist\nfrom . import sibling\n",
      "pkg/a/helpers.py": "def assist():\n    return 1\n",
      "pkg/a/b/sibling.py": "VALUE = 3\n",
  })


@pytest.fixture
def collision_tree() -> SourceTree:
  """Two distinct definitions sharing a name across files."""
  return SourceTree.from_mapping({
      "p/__init__.py": "",
      "p/one.py": "class Conv2d:\n    tag = 'one'\n",
      "p/two.py": "class Conv2d:\n    tag = 'two'\n",
      "p/user.py": "from .one import Conv2d\n\nclass Net:\n    layer = Conv2d\n",
  })


@pytest.fixture
def star_tree() -> SourceTree:
  return SourceTree.from_mapping({
      "s/__init__.py": "from .everything import *\n",
      "s/everything.py": "A = 1\nB = 2\n",
  })


def parse(source: str) -> ast.Module:
  """Parse *source* into a module AST (test helper)."""
  return ast.parse(source)
