"""Tests for OEFP counted batch kernels."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")


def _openeye_mol(smiles: str):
    from openeye import oechem

    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


def _count_fingerprints():
    import oefp

    return [
        oefp.morgan_count_fingerprint(_openeye_mol("CCO"), num_bits=128),
        oefp.morgan_count_fingerprint(_openeye_mol("CCN"), num_bits=128),
        oefp.morgan_count_fingerprint(_openeye_mol("CCCO"), num_bits=128),
    ]


def test_count_batch_views_and_pdist_shape():
    import oefp

    fps = _count_fingerprints()
    batch = oefp.OEFPCountBatch.from_fingerprints(fps)

    assert batch.size == 3
    assert batch.num_bits == 128
    assert batch.offsets.shape == (4,)
    assert batch.indices.shape == batch.counts.shape
    assert batch.entry_count == len(batch.indices)
    assert batch.offsets[0] == 0
    assert batch.offsets[-1] == batch.entry_count
    assert batch.indices.flags.writeable is False
    assert batch.counts.flags.writeable is False
    assert batch.offsets.flags.writeable is False

    values = oefp.pdist(batch, oefp.Metric.tanimoto())
    expected = np.array([
        oefp.compare(fps[0], fps[1], oefp.Metric.tanimoto()),
        oefp.compare(fps[0], fps[2], oefp.Metric.tanimoto()),
        oefp.compare(fps[1], fps[2], oefp.Metric.tanimoto()),
    ])
    np.testing.assert_allclose(values, expected)


def test_count_query_to_batch_matches_scalar_compare():
    import oefp

    fps = _count_fingerprints()
    batch = oefp.OEFPCountBatch.from_fingerprints(fps)

    values = oefp.compare(fps[0], batch, oefp.Metric.dice())
    expected = np.array([oefp.compare(fps[0], fp, oefp.Metric.dice()) for fp in fps])

    assert values.shape == (batch.size,)
    np.testing.assert_allclose(values, expected)


def test_count_cdist_matches_scalar_compare():
    import oefp

    fps = _count_fingerprints()
    a = oefp.OEFPCountBatch.from_fingerprints(fps[:2])
    b = oefp.OEFPCountBatch.from_fingerprints(fps[1:])

    values = oefp.cdist(a, b, oefp.Metric.jaccard())
    expected = np.array([
        [
            oefp.compare(left, right, oefp.Metric.jaccard())
            for right in fps[1:]
        ]
        for left in fps[:2]
    ])

    assert values.shape == (2, 2)
    np.testing.assert_allclose(values, expected)


def _count_panel_mols():
    from openeye import oechem

    mols = []
    for smiles in ("CCO", "c1ccccc1", "CC(=O)O", "CCN"):
        mol = oechem.OEGraphMol()
        assert oechem.OESmilesToMol(mol, smiles)
        mols.append(mol)
    return mols


def test_count_batch_from_molecules_matches_from_fingerprints():
    import oefp

    mols = _count_panel_mols()
    direct = oefp.OEFPCountBatch.from_molecules(mols, oefp.morgan_count_fingerprint, radius=2)
    manual = oefp.OEFPCountBatch.from_fingerprints(
        [oefp.morgan_count_fingerprint(m, radius=2) for m in mols]
    )

    assert direct.size == manual.size
    np.testing.assert_array_equal(direct.indices, manual.indices)
    np.testing.assert_array_equal(direct.counts, manual.counts)
    np.testing.assert_array_equal(direct.offsets, manual.offsets)


def test_count_batch_from_molecules_forwards_options():
    import oefp

    mols = _count_panel_mols()
    ecfp = oefp.OEFPCountBatch.from_molecules(mols, oefp.morgan_count_fingerprint, radius=2)
    fcfp = oefp.OEFPCountBatch.from_molecules(
        mols, oefp.morgan_count_fingerprint, radius=2, use_features=True
    )

    assert not (
        np.array_equal(ecfp.indices, fcfp.indices)
        and np.array_equal(ecfp.counts, fcfp.counts)
    )


def test_count_batch_from_molecules_empty_returns_empty_batch():
    import oefp

    batch = oefp.OEFPCountBatch.from_molecules([], oefp.morgan_count_fingerprint)
    assert batch.size == 0


def test_count_batch_from_molecules_rejects_wrong_element_type():
    import oefp

    mols = _count_panel_mols()
    with pytest.raises(TypeError, match="OEFPCount"):
        oefp.OEFPCountBatch.from_molecules(mols, oefp.morgan_fingerprint)
