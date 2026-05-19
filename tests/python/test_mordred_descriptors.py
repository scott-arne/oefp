"""Parity fixtures for the supported Mordred-compatible descriptor surface."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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

FIRST_BATCH_SOURCE_TYPES = {
    "AcidBase",
    "Aromatic",
    "AtomCount",
    "BondCount",
    "CarbonTypes",
    "HydrogenBond",
    "Lipinski",
    "RotatableBond",
    "TopoPSA",
    "Weight",
}


def _reference_payload() -> dict[str, Any]:
    with REFERENCE_FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def _package_reference_payload() -> dict[str, Any]:
    fixture = resources.files("oefp").joinpath("mordred_references.json")
    assert fixture.is_file()
    with fixture.open(encoding="utf-8") as handle:
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


def _definition_names(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(definition["name"] for definition in payload["definitions"])


def _first_batch_names(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        definition["name"]
        for definition in payload["definitions"]
        if definition["source_type"] in FIRST_BATCH_SOURCE_TYPES
    )


def _reference_values_by_name(payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    return dict(zip(_definition_names(payload), row["values"], strict=True))


def _openeye_mol(smiles: str) -> Any:
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

        assert descriptors.schema == oefp.mordred_schema()
        for name in SUPPORTED_COUNT_NAMES:
            assert descriptors[name] == expected[name]


def test_mordred_descriptors_compare_with_existing_descriptor_surface():
    import oefp

    query = oefp.mordred_descriptors(_openeye_mol("CCO"))
    batch = oefp.DescriptorBatch.from_descriptors(
        [oefp.mordred_descriptors(_openeye_mol("CCO"))]
    )

    assert query.schema == oefp.mordred_schema()
    assert batch.schema == oefp.mordred_schema()
    with np.testing.assert_raises(TypeError):
        oefp.compare(query, batch, oefp.Metric.tanimoto())


def test_mordred_schema_matches_committed_fixture_definitions():
    import oefp

    payload = _reference_payload()
    schema = oefp.mordred_schema()

    assert len(schema.definitions) == 1826
    assert schema.names == _definition_names(payload)


def test_packaged_mordred_reference_matches_test_fixture():
    assert _package_reference_payload() == _reference_payload()


def test_mordred_descriptors_match_first_batch_reference_values():
    import oefp

    payload = _reference_payload()
    names = _first_batch_names(payload)

    assert len(names) == 51
    for row in payload["reference_rows"]:
        descriptors = oefp.mordred_descriptors(_openeye_mol(row["smiles"]))
        expected_by_name = _reference_values_by_name(payload, row)

        assert descriptors.schema == oefp.mordred_schema()
        for name in names:
            expected = expected_by_name[name]
            actual = descriptors[name]
            if isinstance(expected, dict):
                assert expected["state"] in {"missing", "error", "nonfinite"}
                assert actual is None
            elif isinstance(expected, float):
                assert actual is not None
                assert actual == pytest.approx(expected, rel=1e-8, abs=1e-8)
            else:
                assert actual == expected


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
    assert len(payload["reference_rows"]) == 16
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
