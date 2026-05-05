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
