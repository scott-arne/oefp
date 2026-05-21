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

ESTATE_COUNT_NAMES = (
    "NsLi",
    "NssBe",
    "NssssBe",
    "NssBH",
    "NsssB",
    "NssssB",
    "NsCH3",
    "NdCH2",
    "NssCH2",
    "NtCH",
    "NdsCH",
    "NaaCH",
    "NsssCH",
    "NddC",
    "NtsC",
    "NdssC",
    "NaasC",
    "NaaaC",
    "NssssC",
    "NsNH3",
    "NsNH2",
    "NssNH2",
    "NdNH",
    "NssNH",
    "NaaNH",
    "NtN",
    "NsssNH",
    "NdsN",
    "NaaN",
    "NsssN",
    "NddsN",
    "NaasN",
    "NssssN",
    "NsOH",
    "NdO",
    "NssO",
    "NaaO",
    "NsF",
    "NsSiH3",
    "NssSiH2",
    "NsssSiH",
    "NssssSi",
    "NsPH2",
    "NssPH",
    "NsssP",
    "NdsssP",
    "NsssssP",
    "NsSH",
    "NdS",
    "NssS",
    "NaaS",
    "NdssS",
    "NddssS",
    "NsCl",
    "NsGeH3",
    "NssGeH2",
    "NsssGeH",
    "NssssGe",
    "NsAsH2",
    "NssAsH",
    "NsssAs",
    "NsssdAs",
    "NsssssAs",
    "NsSeH",
    "NdSe",
    "NssSe",
    "NaaSe",
    "NdssSe",
    "NddssSe",
    "NsBr",
    "NsSnH3",
    "NssSnH2",
    "NsssSnH",
    "NssssSn",
    "NsI",
    "NsPbH3",
    "NssPbH2",
    "NsssPbH",
    "NssssPb",
)

ENABLED_SOURCE_TYPES = {
    "AcidBase",
    "Aromatic",
    "AtomCount",
    "BaryszMatrix",
    "BondCount",
    "CarbonTypes",
    "Constitutional",
    "DetourMatrix",
    "HydrogenBond",
    "InformationContent",
    "Lipinski",
    "McGowanVolume",
    "MolecularId",
    "Polarizability",
    "RotatableBond",
    "SLogP",
    "TopoPSA",
    "VdwVolumeABC",
    "WalkCount",
    "Weight",
}

ENABLED_DESCRIPTOR_NAMES = {
    "ABC",
    "ABCGG",
    "BalabanJ",
    "BertzCT",
    "SpAbs_A",
    "SpMax_A",
    "SpDiam_A",
    "SpAD_A",
    "SpMAD_A",
    "LogEE_A",
    "VE1_A",
    "VE2_A",
    "VE3_A",
    "VR1_A",
    "VR2_A",
    "VR3_A",
    "SpAbs_D",
    "SpMax_D",
    "SpDiam_D",
    "SpAD_D",
    "SpMAD_D",
    "LogEE_D",
    "VE1_D",
    "VE2_D",
    "VE3_D",
    "VR1_D",
    "VR2_D",
    "VR3_D",
    "SpAbs_Dt",
    "SpMax_Dt",
    "SpDiam_Dt",
    "SpAD_Dt",
    "SpMAD_Dt",
    "LogEE_Dt",
    "SM1_Dt",
    "VE1_Dt",
    "VE2_Dt",
    "VE3_Dt",
    "VR1_Dt",
    "VR2_Dt",
    "VR3_Dt",
    "DetourIndex",
    "SpAbs_DzZ",
    "SpMax_DzZ",
    "SpDiam_DzZ",
    "SpAD_DzZ",
    "SpMAD_DzZ",
    "LogEE_DzZ",
    "SM1_DzZ",
    "VE1_DzZ",
    "VE2_DzZ",
    "VE3_DzZ",
    "VR1_DzZ",
    "VR2_DzZ",
    "VR3_DzZ",
    "MDEC-11",
    "MDEC-12",
    "MDEC-13",
    "MDEC-14",
    "MDEC-22",
    "MDEC-23",
    "MDEC-24",
    "MDEC-33",
    "MDEC-34",
    "MDEC-44",
    "MDEO-11",
    "MDEO-12",
    "MDEO-22",
    "MDEN-11",
    "MDEN-12",
    "MDEN-13",
    "MDEN-22",
    "MDEN-23",
    "MDEN-33",
    "fMF",
    "fragCpx",
    "GGI1",
    "GGI2",
    "GGI3",
    "GGI4",
    "GGI5",
    "GGI6",
    "GGI7",
    "GGI8",
    "GGI9",
    "GGI10",
    "JGI1",
    "JGI2",
    "JGI3",
    "JGI4",
    "JGI5",
    "JGI6",
    "JGI7",
    "JGI8",
    "JGI9",
    "JGI10",
    "JGT10",
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
    "nRing",
    "n3Ring",
    "n4Ring",
    "n5Ring",
    "n6Ring",
    "n7Ring",
    "n8Ring",
    "n9Ring",
    "n10Ring",
    "n11Ring",
    "n12Ring",
    "nG12Ring",
    "nHRing",
    "n3HRing",
    "n4HRing",
    "n5HRing",
    "n6HRing",
    "n7HRing",
    "n8HRing",
    "n9HRing",
    "n10HRing",
    "n11HRing",
    "n12HRing",
    "nG12HRing",
    "naRing",
    "n3aRing",
    "n4aRing",
    "n5aRing",
    "n6aRing",
    "n7aRing",
    "n8aRing",
    "n9aRing",
    "n10aRing",
    "n11aRing",
    "n12aRing",
    "nG12aRing",
    "naHRing",
    "n3aHRing",
    "n4aHRing",
    "n5aHRing",
    "n6aHRing",
    "n7aHRing",
    "n8aHRing",
    "n9aHRing",
    "n10aHRing",
    "n11aHRing",
    "n12aHRing",
    "nG12aHRing",
    "nARing",
    "n3ARing",
    "n4ARing",
    "n5ARing",
    "n6ARing",
    "n7ARing",
    "n8ARing",
    "n9ARing",
    "n10ARing",
    "n11ARing",
    "n12ARing",
    "nG12ARing",
    "nAHRing",
    "n3AHRing",
    "n4AHRing",
    "n5AHRing",
    "n6AHRing",
    "n7AHRing",
    "n8AHRing",
    "n9AHRing",
    "n10AHRing",
    "n11AHRing",
    "n12AHRing",
    "nG12AHRing",
    "nFRing",
    "n4FRing",
    "n5FRing",
    "n6FRing",
    "n7FRing",
    "n8FRing",
    "n9FRing",
    "n10FRing",
    "n11FRing",
    "n12FRing",
    "nG12FRing",
    "nFHRing",
    "n4FHRing",
    "n5FHRing",
    "n6FHRing",
    "n7FHRing",
    "n8FHRing",
    "n9FHRing",
    "n10FHRing",
    "n11FHRing",
    "n12FHRing",
    "nG12FHRing",
    "nFaRing",
    "n4FaRing",
    "n5FaRing",
    "n6FaRing",
    "n7FaRing",
    "n8FaRing",
    "n9FaRing",
    "n10FaRing",
    "n11FaRing",
    "n12FaRing",
    "nG12FaRing",
    "nFaHRing",
    "n4FaHRing",
    "n5FaHRing",
    "n6FaHRing",
    "n7FaHRing",
    "n8FaHRing",
    "n9FaHRing",
    "n10FaHRing",
    "n11FaHRing",
    "n12FaHRing",
    "nG12FaHRing",
    "nFARing",
    "n4FARing",
    "n5FARing",
    "n6FARing",
    "n7FARing",
    "n8FARing",
    "n9FARing",
    "n10FARing",
    "n11FARing",
    "n12FARing",
    "nG12FARing",
    "nFAHRing",
    "n4FAHRing",
    "n5FAHRing",
    "n6FAHRing",
    "n7FAHRing",
    "n8FAHRing",
    "n9FAHRing",
    "n10FAHRing",
    "n11FAHRing",
    "n12FAHRing",
    "nG12FAHRing",
    "Diameter",
    "ECIndex",
    "PetitjeanIndex",
    "Radius",
    "TopoShapeIndex",
    "VAdjMat",
    "WPath",
    "WPol",
    "Zagreb1",
    "Zagreb2",
    "mZagreb1",
    "mZagreb2",
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
        or definition["name"] in ESTATE_COUNT_NAMES
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

    assert len(names) == 645
    assert set(ESTATE_COUNT_NAMES) <= set(names)
    assert {"VR1_A", "VR2_A", "VR3_A", "VR1_D", "VR2_D", "VR3_D"} <= set(names)
    assert {"SpAbs_Dt", "VR1_Dt", "VR2_Dt", "VR3_Dt", "DetourIndex"} <= set(names)
    assert {"SM1_DzZ", "VR1_DzZ", "VR2_DzZ", "VR3_DzZ"} <= set(names)
    assert {"SpAbs_Dzm", "VR3_Dzm", "SpAbs_Dzi", "VR3_Dzi"} <= set(names)
    assert {"MDEC-11", "MDEC-12", "MDEC-44"} <= set(names)
    assert {"MDEO-11", "MDEO-12", "MDEO-22"} <= set(names)
    assert {"MDEN-11", "MDEN-12", "MDEN-33"} <= set(names)
    assert {"IC0", "TIC5", "SIC5", "BIC5", "CIC5", "MIC5", "ZMIC5"} <= set(names)
    assert "fMF" in names
    assert "fragCpx" in names
    assert {"MID", "AMID", "MID_h", "AMID_h", "MID_X", "AMID_X"} <= set(names)
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
