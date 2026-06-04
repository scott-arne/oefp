"""Tests for OEFP batch kernels."""

import numpy as np
import pytest


def test_batch_views_and_pdist_shape():
    import oefp

    fps = [
        oefp.OEFP.from_on_bits(64, [0, 1], algorithm="unit-test"),
        oefp.OEFP.from_on_bits(64, [0], algorithm="unit-test"),
        oefp.OEFP.from_on_bits(64, [2], algorithm="unit-test"),
    ]
    batch = oefp.OEFPBatch.from_fingerprints(fps)

    assert batch.words.shape == (3, 1)
    assert batch.popcounts.tolist() == [2, 1, 1]
    assert batch.words.flags.writeable is False
    assert batch.popcounts.flags.writeable is False

    values = oefp.pdist(batch, oefp.Metric.tanimoto())
    np.testing.assert_allclose(values, np.array([0.5, 0.0, 0.0]))


def test_cdist_shape_and_values():
    import oefp

    a = oefp.OEFPBatch.from_fingerprints(
        [
            oefp.OEFP.from_on_bits(64, [0, 1], algorithm="unit-test"),
            oefp.OEFP.from_on_bits(64, [2, 3], algorithm="unit-test"),
        ]
    )
    b = oefp.OEFPBatch.from_fingerprints(
        [
            oefp.OEFP.from_on_bits(64, [0, 1], algorithm="unit-test"),
            oefp.OEFP.from_on_bits(64, [3], algorithm="unit-test"),
        ]
    )

    values = oefp.cdist(a, b, oefp.Metric.tanimoto())
    assert values.shape == (2, 2)
    assert values[0, 0] == pytest.approx(1.0)
    assert values[0, 1] == pytest.approx(0.0)
    assert values[1, 0] == pytest.approx(0.0)
    assert values[1, 1] == pytest.approx(0.5)


def _panel_mols():
    from openeye import oechem

    mols = []
    for smiles in ("CCO", "c1ccccc1", "CC(=O)O", "CCN"):
        mol = oechem.OEGraphMol()
        assert oechem.OESmilesToMol(mol, smiles)
        mols.append(mol)
    return mols


def test_oefp_batch_from_molecules_matches_from_fingerprints():
    import oefp

    mols = _panel_mols()
    direct = oefp.OEFPBatch.from_molecules(mols, oefp.morgan_fingerprint, radius=2)
    manual = oefp.OEFPBatch.from_fingerprints(
        [oefp.morgan_fingerprint(m, radius=2) for m in mols]
    )

    assert direct.size == manual.size
    np.testing.assert_array_equal(direct.words, manual.words)
    np.testing.assert_array_equal(direct.popcounts, manual.popcounts)


def test_oefp_batch_from_molecules_forwards_options():
    import oefp

    mols = _panel_mols()
    ecfp = oefp.OEFPBatch.from_molecules(mols, oefp.morgan_fingerprint, radius=2)
    fcfp = oefp.OEFPBatch.from_molecules(
        mols, oefp.morgan_fingerprint, radius=2, use_features=True
    )

    assert not np.array_equal(ecfp.words, fcfp.words)


def test_oefp_batch_from_molecules_empty_returns_empty_batch():
    import oefp

    calls = []

    def spy(mol):
        calls.append(mol)
        return oefp.morgan_fingerprint(mol)

    batch = oefp.OEFPBatch.from_molecules([], spy)
    assert batch.size == 0
    assert calls == []


def test_oefp_batch_from_molecules_rejects_wrong_element_type():
    import oefp

    mols = _panel_mols()
    with pytest.raises(TypeError, match="OEFP"):
        oefp.OEFPBatch.from_molecules(mols, oefp.morgan_count_fingerprint)
