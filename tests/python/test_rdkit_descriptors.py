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
    # CountsWeights is computed (Task 3); un-ported families stay missing.
    assert row["HeavyAtomCount"] == 3
    assert row["TPSA"] is None


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
    # CountsWeights is computed (Task 3); un-ported families remain missing.
    assert row["ExactMolWt"] == pytest.approx(46.0419, rel=1e-4)
    assert row["MolLogP"] is None


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


# Families enabled for conformance checking; grows as tasks land. This set
# enables the 21 dependency-free CountsWeights descriptors and the 11
# RingCounts descriptors. `SPS` and `Phi` are CountsWeights members deferred to
# later tasks (see comments below).
ENABLED_DESCRIPTOR_NAMES: set[str] = {
    "MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
    "NumRadicalElectrons", "FpDensityMorgan1", "FpDensityMorgan2",
    "FpDensityMorgan3", "FractionCSP3", "HeavyAtomCount", "NHOHCount",
    "NOCount", "NumAmideBonds", "NumAtomStereoCenters", "NumBridgeheadAtoms",
    "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
    "NumSpiroAtoms", "NumUnspecifiedAtomStereoCenters",
    # RingCounts (Task 5): 11 dependency-free SSSR ring classifications.
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
    "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles", "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles", "NumHeterocycles",
    # "SPS" deferred — RDKit SpacialScore needs RDKit-internal potential-stereo +
    # hybridization models OpenEye doesn't expose; needs a dedicated deep-dive task.
    # "Phi" added in Task 6 (needs Connectivity Kappa artifacts).
}


def _tier_by_name(payload: dict) -> dict[str, str]:
    return {d["name"]: d["tier"] for d in payload["definitions"]}


def _values_by_name(payload: dict, row: dict) -> dict:
    names = [d["name"] for d in payload["definitions"]]
    return dict(zip(names, row["values"], strict=True))


def test_enabled_rdkit_descriptors_match_reference_at_tier():
    import oefp

    payload = _payload()
    tiers = _tier_by_name(payload)
    calc = oefp.DescriptorCalculator([oefp.RDKitDescriptorSource()])

    for row in payload["reference_rows"]:
        expected = _values_by_name(payload, row)
        actual = calc.compute(_openeye_mol(row["smiles"]))
        for name in ENABLED_DESCRIPTOR_NAMES:
            ref = expected[name]
            got = actual[name]
            assert got is not None, f"{name} @ {row['smiles']}"
            if isinstance(ref, dict):
                continue  # oracle error/missing; not enabled for exact match
            tol = TIER_TOLERANCE[tiers[name]]
            if isinstance(ref, bool) or isinstance(ref, int):
                assert got == ref, f"{name} @ {row['smiles']}: {got} != {ref}"
            else:
                assert got == pytest.approx(ref, rel=tol, abs=tol), \
                    f"{name} @ {row['smiles']} tier={tiers[name]}"
