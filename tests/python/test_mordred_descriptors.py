"""Parity fixtures for the supported Mordred-compatible descriptor subset."""

from __future__ import annotations

from typing import Any

import numpy as np

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

MORDRED_REFERENCES = {
    "CCO": {
        "nAromAtom": 0,
        "nAromBond": 0,
        "nAtom": 9,
        "nHeavyAtom": 3,
        "nHetero": 1,
        "nH": 6,
        "nB": 0,
        "nC": 2,
        "nN": 0,
        "nO": 1,
        "nS": 0,
        "nP": 0,
        "nF": 0,
        "nCl": 0,
        "nBr": 0,
        "nI": 0,
        "nX": 0,
        "nBonds": 8,
        "nBondsO": 2,
        "nBondsS": 8,
        "nBondsD": 0,
        "nBondsT": 0,
        "nBondsA": 0,
        "nBondsM": 0,
        "nBondsKS": 8,
        "nBondsKD": 0,
    },
    "c1ccncc1": {
        "nAromAtom": 6,
        "nAromBond": 6,
        "nAtom": 11,
        "nHeavyAtom": 6,
        "nHetero": 1,
        "nH": 5,
        "nB": 0,
        "nC": 5,
        "nN": 1,
        "nO": 0,
        "nS": 0,
        "nP": 0,
        "nF": 0,
        "nCl": 0,
        "nBr": 0,
        "nI": 0,
        "nX": 0,
        "nBonds": 11,
        "nBondsO": 6,
        "nBondsS": 5,
        "nBondsD": 0,
        "nBondsT": 0,
        "nBondsA": 6,
        "nBondsM": 6,
        "nBondsKS": 8,
        "nBondsKD": 3,
    },
    "CC(C)(C)Cl": {
        "nAromAtom": 0,
        "nAromBond": 0,
        "nAtom": 14,
        "nHeavyAtom": 5,
        "nHetero": 1,
        "nH": 9,
        "nB": 0,
        "nC": 4,
        "nN": 0,
        "nO": 0,
        "nS": 0,
        "nP": 0,
        "nF": 0,
        "nCl": 1,
        "nBr": 0,
        "nI": 0,
        "nX": 1,
        "nBonds": 13,
        "nBondsO": 4,
        "nBondsS": 13,
        "nBondsD": 0,
        "nBondsT": 0,
        "nBondsA": 0,
        "nBondsM": 0,
        "nBondsKS": 13,
        "nBondsKD": 0,
    },
}


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

    for smiles, expected in MORDRED_REFERENCES.items():
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

    query = oefp.mordred_descriptors(_openeye_mol("CCO"))
    library_smiles = ["CCO", "c1ccncc1", "CC(C)(C)Cl"]
    library = [oefp.mordred_descriptors(_openeye_mol(smiles)) for smiles in library_smiles]
    batch = oefp.DescriptorBatch.from_descriptors(library)

    expected = np.array(
        [
            _count_tanimoto(MORDRED_REFERENCES["CCO"], MORDRED_REFERENCES[smiles])
            for smiles in library_smiles
        ]
    )
    np.testing.assert_allclose(oefp.compare(query, batch, oefp.Metric.tanimoto()), expected)
