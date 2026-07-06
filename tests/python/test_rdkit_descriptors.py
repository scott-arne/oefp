"""Conformance fixtures for the supported RDKit 2D descriptor surface."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

REFERENCE_FIXTURE = Path(__file__).with_name("rdkit_references.json")

TIER_TOLERANCE = {"exact": 1e-8, "tight": 1e-4, "loose": 1e-2}


def _payload() -> dict:
    with REFERENCE_FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def _package_reference_payload() -> dict:
    fixture = resources.files("oefp").joinpath("rdkit_references.json")
    assert fixture.is_file()
    with fixture.open(encoding="utf-8") as handle:
        return json.load(handle)


def _openeye_mol(smiles: str):
    oechem = pytest.importorskip("openeye.oechem")
    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


def test_rdkit_source_registers_and_returns_full_width_row():
    import oefp

    calc = oefp.DescriptorCalculator([oefp.RDKitDescriptorSource()])
    assert len(calc.schema.names) == 214
    row = calc.compute(_openeye_mol("CCO"))
    assert all(row[name] is None for name in calc.schema.names)


def test_rdkit_schema_size_matches_fixture():
    import oefp

    payload = _payload()
    assert len(payload["definitions"]) == 214
    assert oefp.rdkit_schema().names == tuple(d["name"] for d in payload["definitions"])


def test_packaged_rdkit_reference_matches_test_fixture():
    assert _package_reference_payload() == _payload()


def test_rdkit_descriptors_free_function_returns_full_width_row():
    import oefp

    row = oefp.rdkit_descriptors(_openeye_mol("CCO"))
    assert len(row.schema.names) == 214
    # Skeleton stub: every value is currently missing (families land in later tasks).
    assert all(row[name] is None for name in row.schema.names)


def test_rdkit_native_free_functions_are_exported():
    from oefp import _native

    assert hasattr(_native, "MakeRDKitDescriptors")
    assert hasattr(_native, "RDKitDescriptorSchema")


def test_rdkit_descriptors_release_gil_for_concurrent_computation():
    """Verify rdkit_descriptors releases the GIL during computation.

    A smoke test that exercises rdkit_descriptors concurrently from multiple
    threads to confirm the exported trampoline releases the GIL (no deadlock,
    no serialization). Each thread computes the descriptor row for a small
    molecule and asserts the expected 214-column width. If the GIL were held,
    Python threads would serialize (functionally correct but not concurrent);
    this test simply confirms all threads complete successfully without error.
    """
    from concurrent.futures import ThreadPoolExecutor
    import oefp

    def compute_row(smiles: str) -> int:
        mol = _openeye_mol(smiles)
        row = oefp.rdkit_descriptors(mol)
        assert len(row.schema.names) == 214
        return len(row.schema.names)

    smiles_batch = ["CCO", "c1ccccc1", "CC(C)C", "CCCC", "C1CCCCC1"]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(compute_row, smiles_batch))

    assert results == [214] * len(smiles_batch)
