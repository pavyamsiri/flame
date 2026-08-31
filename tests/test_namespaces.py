"""Smoke tests for public package namespaces."""

from __future__ import annotations

import importlib

import flame


def test_top_level_namespace_exports() -> None:
    for name in flame.__all__:
        assert getattr(flame, name) is not None


def test_backend_namespace_modules_import() -> None:
    for backend in (flame.numpy, flame.polars):
        for module_name in backend.__all__:
            module = importlib.import_module(f"{backend.__name__}.{module_name}")
            assert module.__all__
