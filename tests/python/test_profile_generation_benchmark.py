"""Tests for the benchmark profiling harness helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class FakeRDMol:
    def __init__(self, heavy_atom_count: int) -> None:
        self._heavy_atom_count = heavy_atom_count

    def GetNumHeavyAtoms(self) -> int:
        return self._heavy_atom_count


def _load_profile_generation_module() -> ModuleType:
    module_path = Path(__file__).parents[2] / "benchmarks" / "profile_generation.py"
    spec = importlib.util.spec_from_file_location("profile_generation_benchmark", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bucketed_molecules_reports_all_and_heavy_atom_ranges() -> None:
    module = _load_profile_generation_module()
    oe_mols: list[Any] = [object(), object(), object(), object(), object()]
    rd_mols = [
        FakeRDMol(10),
        FakeRDMol(20),
        FakeRDMol(21),
        FakeRDMol(40),
        FakeRDMol(41),
    ]

    buckets = module._bucketed_molecules(oe_mols, rd_mols)

    assert [(bucket.label, len(bucket.oe_mols), len(bucket.rd_mols)) for bucket in buckets] == [
        ("all", 5, 5),
        ("heavy_atoms<=20", 2, 2),
        ("heavy_atoms=21-40", 2, 2),
        ("heavy_atoms>40", 1, 1),
    ]
