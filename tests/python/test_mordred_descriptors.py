"""Parity fixtures for the supported Mordred-compatible descriptor surface."""

from __future__ import annotations

import json
import sys
from collections import Counter
from importlib import resources
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import compare_mordred_parity

REFERENCE_FIXTURE = Path(__file__).with_name("mordred_references.json")
DIVERGENCE_FIXTURE = Path(__file__).with_name("mordred_divergences.json")
POLICY_STATUSES = {"exact", "openeye_divergent", "deferred", "not_applicable"}
POLICY_REQUIRED_FIELDS = {
    "descriptor",
    "family",
    "status",
    "source",
    "primitive",
    "reason",
}
ROW_DIVERGENCE_REQUIRED_FIELDS = {
    "status",
    "descriptor",
    "smiles",
    "reference",
    "observed",
    "source",
    "primitive",
    "reason",
}
_OPTIONAL_REFERENCE_MODULES_AT_IMPORT = {
    name
    for name in sys.modules
    if name == "mordred"
    or name.startswith("mordred.")
    or name == "rdkit"
    or name.startswith("rdkit.")
}

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

ENABLED_SOURCE_TYPES = {
    "AcidBase",
    "Aromatic",
    "AtomCount",
    "BondCount",
    "CarbonTypes",
    "Constitutional",
    "HydrogenBond",
    "Lipinski",
    "McGowanVolume",
    "Polarizability",
    "RotatableBond",
    "SLogP",
    "TopoPSA",
    "VdwVolumeABC",
    "WalkCount",
    "Weight",
}

ENABLED_DESCRIPTOR_NAMES = {
    "AXp-0d",
    "AXp-0dv",
    "AXp-1d",
    "AXp-1dv",
    "AXp-2d",
    "AXp-2dv",
    "AXp-3d",
    "AXp-3dv",
    "AXp-4d",
    "AXp-4dv",
    "AXp-5d",
    "AXp-5dv",
    "AXp-6d",
    "AXp-6dv",
    "AXp-7d",
    "AXp-7dv",
    "Xch-3d",
    "Xch-3dv",
    "Xch-4d",
    "Xch-4dv",
    "Xch-5d",
    "Xch-5dv",
    "Xch-6d",
    "Xch-6dv",
    "Xch-7d",
    "Xch-7dv",
    "Xc-3d",
    "Xc-3dv",
    "Xc-4d",
    "Xc-4dv",
    "Xc-5d",
    "Xc-5dv",
    "Xc-6d",
    "Xc-6dv",
    "Xpc-4d",
    "Xpc-4dv",
    "Xpc-5d",
    "Xpc-5dv",
    "Xpc-6d",
    "Xpc-6dv",
    "MPC2",
    "MPC3",
    "MPC4",
    "MPC5",
    "MPC6",
    "MPC7",
    "MPC8",
    "MPC9",
    "MPC10",
    "TMPC10",
    "piPC1",
    "piPC2",
    "piPC3",
    "piPC4",
    "piPC5",
    "piPC6",
    "piPC7",
    "piPC8",
    "piPC9",
    "piPC10",
    "TpiPC10",
    "Xp-0d",
    "Xp-0dv",
    "Xp-1d",
    "Xp-1dv",
    "Xp-2d",
    "Xp-2dv",
    "Xp-3d",
    "Xp-3dv",
    "Xp-4d",
    "Xp-4dv",
    "Xp-5d",
    "Xp-5dv",
    "Xp-6d",
    "Xp-6dv",
    "Xp-7d",
    "Xp-7dv",
    "Kier1",
    "Kier2",
    "Kier3",
}


def _reference_payload() -> dict[str, Any]:
    with REFERENCE_FIXTURE.open(encoding="utf-8") as handle:
        return json.load(handle)


def _divergence_payload() -> dict[str, Any]:
    with DIVERGENCE_FIXTURE.open(encoding="utf-8") as handle:
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


def _enabled_descriptor_names(payload: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        definition["name"]
        for definition in payload["definitions"]
        if definition["source_type"] in ENABLED_SOURCE_TYPES
        or definition["name"] in ENABLED_DESCRIPTOR_NAMES
    )


def _policy_descriptors_by_status(status: str) -> tuple[str, ...]:
    return tuple(
        policy["descriptor"]
        for policy in _divergence_payload()["policies"]
        if policy["status"] == status
    )


def _row_divergence_policy() -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (row["descriptor"], row["smiles"]): row
        for row in _divergence_payload()["row_divergences"]
        if row["status"] == "openeye_divergent"
    }


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


def test_mordred_descriptors_match_enabled_reference_values():
    import oefp

    payload = _reference_payload()
    names = _enabled_descriptor_names(payload)
    row_divergences = _row_divergence_policy()

    assert len(names) == 174
    for row in payload["reference_rows"]:
        smiles = row["smiles"]
        descriptors = oefp.mordred_descriptors(_openeye_mol(row["smiles"]))
        expected_by_name = _reference_values_by_name(payload, row)

        assert descriptors.schema == oefp.mordred_schema()
        for name in names:
            expected = expected_by_name[name]
            actual = descriptors[name]
            row_divergence = row_divergences.get((name, smiles))
            if row_divergence is not None:
                assert expected == row_divergence["reference"]
                assert actual == row_divergence["observed"]
                continue
            if isinstance(expected, dict):
                assert expected["state"] in {"missing", "error", "nonfinite"}
                assert actual is None
            elif isinstance(expected, float):
                assert actual is not None
                assert actual == pytest.approx(expected, rel=1e-8, abs=1e-8)
            else:
                assert actual == expected


def test_no_mordred_filter_descriptors_remain_deferred():
    assert "Lipinski" not in _policy_descriptors_by_status("deferred")
    assert "GhoseFilter" not in _policy_descriptors_by_status("deferred")


def test_parity_harness_rejects_concrete_values_for_deferred_descriptors(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_oefp = SimpleNamespace(mordred_descriptors=lambda _mol: {"DeferredFlag": True})
    monkeypatch.setitem(sys.modules, "oefp", fake_oefp)
    monkeypatch.setattr(compare_mordred_parity, "_openeye_mol", lambda smiles: smiles)

    references = {
        "definitions": [
            {
                "name": "DeferredFlag",
                "source_type": "DeferredFamily",
            }
        ],
        "reference_rows": [
            {
                "smiles": "CCO",
                "values": [False],
            }
        ],
    }
    policy = {
        "policies": [
            {
                "descriptor": "DeferredFlag",
                "status": "deferred",
            }
        ],
        "row_divergences": [],
    }

    counts, unclassified = compare_mordred_parity._compare(
        references,
        policy,
        ("DeferredFlag",),
    )

    assert counts == {
        "exact": 0,
        "accepted_divergences": 0,
        "deferred": 0,
        "not_applicable": 0,
        "unclassified": 1,
    }
    assert unclassified == [
        "DeferredFlag CCO: deferred descriptor produced concrete value True "
        "primitive=DeferredFamily"
    ]


def test_parity_harness_rejects_duplicate_descriptor_policies():
    policy = {
        "policies": [
            {
                "descriptor": "Repeated",
                "status": "exact",
            },
            {
                "descriptor": "Repeated",
                "status": "deferred",
            },
        ]
    }

    with pytest.raises(ValueError, match="Duplicate descriptor policy entries: Repeated"):
        compare_mordred_parity._descriptor_policy(policy)


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
    assert len(payload["reference_rows"]) == 26
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


def test_mordred_divergence_policy_manifest_is_valid():
    payload = _divergence_payload()
    expected_descriptors = set(_enabled_descriptor_names(_reference_payload()))

    assert {
        "schema_version",
        "reference_sources",
        "policies",
        "row_divergences",
    } <= set(payload)
    assert payload["schema_version"] == 1

    policies = payload["policies"]
    assert isinstance(policies, list)
    assert Counter(policy["descriptor"] for policy in policies) == {
        descriptor: 1 for descriptor in expected_descriptors
    }
    for policy in policies:
        assert POLICY_REQUIRED_FIELDS <= set(policy)
        assert policy["status"] in POLICY_STATUSES


def test_mordred_divergence_rows_include_required_context():
    payload = _divergence_payload()

    for row in payload["row_divergences"]:
        assert row["status"] in POLICY_STATUSES
        if row["status"] == "openeye_divergent":
            assert ROW_DIVERGENCE_REQUIRED_FIELDS <= set(row)


def test_mordred_descriptor_tests_do_not_import_reference_toolkits():
    loaded_after_import = {
        name
        for name in sys.modules
        if name == "mordred"
        or name.startswith("mordred.")
        or name == "rdkit"
        or name.startswith("rdkit.")
    }

    assert loaded_after_import <= _OPTIONAL_REFERENCE_MODULES_AT_IMPORT
