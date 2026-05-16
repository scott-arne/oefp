"""Tests for the OEFP dense-binary Python API."""

import numpy as np
import pytest


def test_oefp_words_view_is_read_only():
    import oefp

    fp = oefp.OEFP.from_on_bits(128, [0, 64, 127], algorithm="unit-test")
    arr = fp.words

    assert arr.shape == (2,)
    assert arr.dtype == np.uint64
    assert fp.popcount == 3
    assert fp.num_bits == 128
    assert arr.flags.writeable is False

    with pytest.raises(ValueError):
        arr[0] = 0


def test_scalar_compare_from_python():
    import oefp

    a = oefp.OEFP.from_on_bits(8, [0, 1, 2], algorithm="unit-test")
    b = oefp.OEFP.from_on_bits(8, [1, 2, 3], algorithm="unit-test")
    metric = oefp.Metric.tanimoto()

    assert oefp.compare(a, b, metric) == pytest.approx(0.5)
