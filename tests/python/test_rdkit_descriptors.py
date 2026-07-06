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
