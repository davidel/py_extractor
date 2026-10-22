"""Tests for symbol discovery and scope-aware free-name analysis."""

from __future__ import annotations

import ast

import pytest

from py_extractor.analysis.symbols import (
    collect_module_symbols,
    find_definition,
    free_names,
)
from py_extractor.errors import SymbolNotFoundError


def _first(source: str) -> ast.AST:
  return ast.parse(source).body[0]


# --- symbol discovery -------------------------------------------------------


def test_collect_defs_and_assignments() -> None:
  tree = ast.parse("X = 1\nY: int = 2\nclass C:\n    pass\ndef f():\n    pass\n")
  symbols = collect_module_symbols(tree)
  assert set(symbols.defs) == {"C", "f"}
  assert {"X", "Y"} <= set(symbols.assignments)


def test_all_exports_parsed() -> None:
  tree = ast.parse("__all__ = ['A', 'B']\n")
  assert collect_module_symbols(tree).all_exports == ["A", "B"]


def test_find_definition_top_level() -> None:
  tree = ast.parse("def f():\n    pass\n")
  assert isinstance(find_definition(tree, "f"), ast.FunctionDef)


def test_find_definition_method() -> None:
  tree = ast.parse("class C:\n    def m(self):\n        pass\n")
  assert isinstance(find_definition(tree, "C.m"), ast.FunctionDef)


def test_find_definition_missing_raises() -> None:
  with pytest.raises(SymbolNotFoundError):
    find_definition(ast.parse("x = 1\n"), "nope")


# --- free-name analysis -----------------------------------------------------


def test_locals_and_builtins_excluded() -> None:
  src = "def f(x):\n    total = 0\n    for i in range(len(x)):\n        total += x[i]\n    return total\n"
  assert free_names(_first(src)) == set()


def test_comprehension_variables_excluded() -> None:
  src = "def f(items):\n    return [transform(i) for i in items]\n"
  assert free_names(_first(src)) == {"transform"}


def test_nested_function_scope() -> None:
  src = ("def outer(a):\n"
         "    b = a + GLOBAL_CONST\n"
         "    def inner(c):\n"
         "        return c + b + np.pi\n"
         "    return inner\n")
  assert free_names(_first(src)) == {"GLOBAL_CONST", "np"}


def test_class_bases_decorators_and_body() -> None:
  src = ("class Model(Base, mixin=meta):\n"
         "    @decorator\n"
         "    def forward(self, x):\n"
         "        return torch.relu(self.conv(x)) + SCALE\n")
  assert free_names(_first(src)) == {"Base", "decorator", "meta", "torch", "SCALE"}


def test_method_parameters_not_free() -> None:
  src = "class C:\n    def m(self, x):\n        return self.f(x)\n"
  assert free_names(_first(src)) == set()


def test_lambda_parameters_and_defaults() -> None:
  src = "def g(a, b=default_b):\n    return lambda z: z + a + EXTERNAL\n"
  assert free_names(_first(src)) == {"default_b", "EXTERNAL"}


def test_annassign_value_is_dependency() -> None:
  src = "def f():\n    x: SomeType = make_default()\n    return x\n"
  assert free_names(_first(src)) == {"SomeType", "make_default"}


def test_fstring_names_are_dependencies() -> None:
  src = "def f(name):\n    return f'hello {name} {OTHER}'\n"
  assert free_names(_first(src)) == {"OTHER"}


def test_decorator_with_arguments_is_dependency() -> None:
  src = "@register(name=GLOBAL_NAME)\ndef f():\n    pass\n"
  assert free_names(_first(src)) == {"register", "GLOBAL_NAME"}


def test_global_declaration_excluded() -> None:
  src = "def f():\n    global counter\n    counter += 1\n"
  assert free_names(_first(src)) == set()
