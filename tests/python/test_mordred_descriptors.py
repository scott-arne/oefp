"""Parity fixtures for the supported Mordred-compatible descriptor subset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REFERENCE_FIXTURE = Path(__file__).with_name("mordred_references.json")

SUPPORTED_COUNT_NAMES = (
    "nAromAtom",
    "nAromBond",
    "nAtom",
    "nB",
    "nBonds",
    "nBondsA",
    "nBondsD",
    "nBondsKD",
    "nBondsKS",
    "nBondsM",
    "nBondsO",
    "nBondsS",
    "nBondsT",
    "nBr",
    "nC",
    "nCl",
    "nF",
    "nH",
    "nHeavyAtom",
    "nHetero",
    "nI",
    "nN",
    "nO",
    "nP",
    "nS",
    "nX",
)


def _reference_payload() -> dict[str, Any]:
    with REFERENCE_FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def _mordred_references() -> dict[str, dict[str, int]]:
    payload = _reference_payload()
    descriptor_names = [definition["name"] for definition in payload["definitions"]]
    references = {}
    for row in payload["reference_rows"]:
        values_by_name = dict(zip(descriptor_names, row["values"], strict=True))
        references[row["smiles"]] = {
            name: int(values_by_name[name]) for name in SUPPORTED_COUNT_NAMES
        }
    return references


def _openeye_mol(smiles: str) -> Any:
    import pytest

    oechem = pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")

    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


def _descriptor_counts(descriptors: Any) -> dict[str, int]:
    return dict(zip(descriptors.string_keys, map(int, descriptors.counts), strict=True))


def _count_or_zero(counts: dict[str, int], key: str) -> int:
    return counts.get(key, 0)


def _count_tanimoto(left: dict[str, int], right: dict[str, int]) -> float:
    intersection = sum(min(_count_or_zero(left, key), _count_or_zero(right, key)) for key in SUPPORTED_COUNT_NAMES)
    union = sum(max(_count_or_zero(left, key), _count_or_zero(right, key)) for key in SUPPORTED_COUNT_NAMES)
    return intersection / union


def test_mordred_descriptors_match_copied_reference_values():
    import oefp

    for smiles, expected in _mordred_references().items():
        descriptors = oefp.mordred_descriptors(_openeye_mol(smiles))
        counts = _descriptor_counts(descriptors)

        assert descriptors.value_type == "string"
        assert descriptors.spec.value_type == "string"
        assert descriptors.spec.source_name == "Mordred-compatible"
        assert descriptors.spec.source_type == "MordredCount"
        assert descriptors.spec.source_version == "Mordred-1.2.0"
        assert descriptors.spec.parameters == "preset=atom_bond_count;zero_counts=omitted"
        for name in SUPPORTED_COUNT_NAMES:
            assert _count_or_zero(counts, name) == expected[name]


def test_mordred_descriptors_compare_with_existing_descriptor_surface():
    import oefp

    references = _mordred_references()
    query = oefp.mordred_descriptors(_openeye_mol("CCO"))
    library_smiles = ["CCO", "c1ccncc1", "CC(C)(C)Cl"]
    library = [oefp.mordred_descriptors(_openeye_mol(smiles)) for smiles in library_smiles]
    batch = oefp.DescriptorBatch.from_descriptors(library)

    expected = np.array(
        [
            _count_tanimoto(references["CCO"], references[smiles])
            for smiles in library_smiles
        ]
    )
    np.testing.assert_allclose(oefp.compare(query, batch, oefp.Metric.tanimoto()), expected)


def test_mordred_reference_fixture_contains_full_schema_and_panel():
    payload = _reference_payload()
    definitions = payload["definitions"]
    definitions_by_name = {definition["name"]: definition for definition in definitions}
    names = [definition["name"] for definition in definitions]

    assert payload["schema_id"] == "1fa9a8e86a7c8731"
    assert payload["source"] == {
        "descriptor_source": "local-mordred-1.2.0",
        "ignore_3D": False,
        "name": "Mordred",
        "version": "Mordred-1.2.0",
    }
    assert payload["source"]["version"] == "Mordred-1.2.0"
    assert len(definitions) == 1826
    assert names[0] == "ABC"
    assert names[-1] == "mZagreb2"
    assert names.index("nAtom") == 18
    assert names.index("Lipinski") == 1351
    assert names.index("GhoseFilter") == 1352
    assert len(payload["reference_rows"]) == 8
    assert definitions_by_name["ABC"]["value_kind"] == "float"
    assert definitions_by_name["nAtom"]["value_kind"] == "int"
    assert definitions_by_name["Lipinski"]["value_kind"] == "bool"
    for row in payload["reference_rows"]:
        assert len(row["values"]) == 1826
        for value in row["values"]:
            if isinstance(value, dict):
                assert set(value) == {"error_type", "state"}
                assert value["state"] in {"missing", "error", "nonfinite"}
                assert value["error_type"]
