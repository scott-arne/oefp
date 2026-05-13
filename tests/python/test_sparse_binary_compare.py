"""Tests for sparse binary fingerprint scalar comparisons."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")


def _openeye_mol(smiles: str):
    from openeye import oechem

    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


def _sparse_bits(fp) -> set[int]:
    return {int(index) for index in fp.indices}


def _binary_stats(a: set[int], b: set[int]) -> tuple[int, int, int, int]:
    intersection = len(a & b)
    only_a = len(a - b)
    only_b = len(b - a)
    xor_count = only_a + only_b
    return intersection, only_a, only_b, xor_count


def test_sparse_binary_compare_matches_set_math():
    import oefp

    left = oefp.morgan_sparse_fingerprint(_openeye_mol("CCO"), radius=2)
    right = oefp.morgan_sparse_fingerprint(_openeye_mol("CCN"), radius=2)
    intersection, only_left, only_right, xor_count = _binary_stats(
        _sparse_bits(left),
        _sparse_bits(right),
    )

    tanimoto = intersection / (intersection + only_left + only_right)
    dice = 2 * intersection / (len(left.indices) + len(right.indices))
    cosine = intersection / math.sqrt(len(left.indices) * len(right.indices))
    tversky = intersection / (intersection + 0.25 * only_left + 0.75 * only_right)

    assert oefp.compare(left, right, oefp.Metric.tanimoto()) == pytest.approx(tanimoto)
    assert oefp.compare(left, right, oefp.Metric.jaccard()) == pytest.approx(
        1.0 - tanimoto,
    )
    assert oefp.compare(left, right, oefp.Metric.dice()) == pytest.approx(dice)
    assert oefp.compare(left, right, oefp.Metric.cosine()) == pytest.approx(cosine)
    assert oefp.compare(left, right, oefp.Metric.tversky(0.25, 0.75)) == pytest.approx(
        tversky,
    )
    assert oefp.compare(left, right, oefp.Metric.manhattan()) == pytest.approx(xor_count)


def test_sparse_binary_compare_rejects_different_specs():
    import oefp

    left = oefp.morgan_sparse_fingerprint(_openeye_mol("CCO"), radius=2)
    right = oefp.morgan_sparse_fingerprint(_openeye_mol("CCO"), radius=1)

    with pytest.raises(RuntimeError, match="specification"):
        oefp.compare(left, right, oefp.Metric.tanimoto())
