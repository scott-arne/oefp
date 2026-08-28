"""Conformance tests for numeric descriptor comparison against scipy."""

from __future__ import annotations

import numpy as np
import pytest

distance = pytest.importorskip(
    "scipy.spatial.distance",
    reason="scipy is required for numeric descriptor conformance tests",
)

COLUMNS = ("A", "B", "C")
VALUES = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 0.5, -1.0],
        [-2.0, 7.0, 0.25],
        [0.0, -3.0, 8.0],
    ],
    dtype=np.float64,
)

# VALUES with the signs removed. Bray-Curtis is an abundance measure: OEFP's
# sum|u-v| / (sum|u| + sum|v|) and scipy's sum|u-v| / sum|u+v| coincide exactly
# for non-negative input, so conformance is only meaningful on this fixture.
NONNEGATIVE_VALUES = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 0.5, 1.0],
        [2.0, 7.0, 0.25],
        [0.0, 3.0, 8.0],
    ],
    dtype=np.float64,
)


def _batch(values=VALUES):
    import oefp

    schema = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition(name, "float") for name in COLUMNS]
    )
    return oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, dict(zip(COLUMNS, row)))
            for row in values.tolist()
        ]
    )


@pytest.mark.parametrize(
    ("build_metric", "scipy_name", "kwargs"),
    [
        (lambda o: o.Metric.euclidean(), "euclidean", {}),
        (lambda o: o.Metric.manhattan(), "cityblock", {}),
        (lambda o: o.Metric.chebyshev(), "chebyshev", {}),
        (lambda o: o.Metric.canberra(), "canberra", {}),
        (lambda o: o.Metric.hamming(), "hamming", {}),
        (lambda o: o.Metric.minkowski(3.0), "minkowski", {"p": 3.0}),
        (
            lambda o: o.Metric.minkowski(3.0, [1.0, 0.5, 2.0]),
            "minkowski",
            {"p": 3.0, "w": [1.0, 0.5, 2.0]},
        ),
    ],
)
def test_numeric_pdist_matches_scipy(build_metric, scipy_name, kwargs):
    import oefp

    actual = oefp.pdist(_batch(), build_metric(oefp), columns=list(COLUMNS))
    expected = distance.pdist(VALUES, scipy_name, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_numeric_pdist_bray_curtis_matches_scipy_on_non_negative_data():
    import oefp

    actual = oefp.pdist(
        _batch(NONNEGATIVE_VALUES), oefp.Metric.bray_curtis(), columns=list(COLUMNS)
    )
    expected = distance.pdist(NONNEGATIVE_VALUES, "braycurtis")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_numeric_pdist_bray_curtis_uses_the_signed_safe_denominator():
    """OEFP normalizes by sum|u| + sum|v|, not scipy's sum|u + v|.

    The two agree for non-negative input, but only OEFP's form stays bounded in
    [0, 1] once a descriptor goes negative, and only OEFP's is defined at u = -v.
    """
    import oefp

    # Row absolute-value sums: r0=6, r1=5.5, r2=9.25, r3=11.
    #   (0,1) 8.5/11.5   (0,2) 10.75/15.25  (0,3) 11/17
    #   (1,2) 13.75/14.75 (1,3) 16.5/16.5   (2,3) 19.75/20.25
    expected = np.array(
        [
            0.7391304347826086,
            0.7049180327868853,
            0.6470588235294118,
            0.9322033898305084,
            1.0,
            0.9753086419753086,
        ]
    )
    actual = oefp.pdist(_batch(), oefp.Metric.bray_curtis(), columns=list(COLUMNS))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    # The pair at index 4 reaches the upper bound exactly; scipy's form is not
    # bounded here at all, so a switch to it would break this assertion.
    assert actual.max() <= 1.0
    scipy_values = distance.pdist(VALUES, "braycurtis")
    assert not np.allclose(actual, scipy_values, rtol=1e-6)


def test_numeric_pdist_bray_curtis_returns_zero_for_zero_mass_rows():
    """OEFP resolves the undefined quotient to 0.0; scipy returns NaN.

    Two rows with no absolute-value mass give Bray-Curtis a zero denominator.
    ``zero_safe_divide`` (src/compare_detail.h:64) returns 0.0 rather than letting NaN
    propagate into a distance matrix.
    """
    import oefp

    zeros = np.zeros((2, len(COLUMNS)), dtype=np.float64)
    actual = oefp.pdist(_batch(zeros), oefp.Metric.bray_curtis(), columns=list(COLUMNS))
    np.testing.assert_array_equal(actual, np.zeros(1))
    assert np.isnan(distance.pdist(zeros, "braycurtis")).all()


def test_numeric_pdist_matches_scipy_for_the_variance_aware_metrics():
    import oefp

    statistics = oefp.column_statistics(_batch(), list(COLUMNS))
    seuclidean = oefp.Metric.standardized_euclidean(statistics.variance)
    np.testing.assert_allclose(
        oefp.pdist(_batch(), seuclidean, columns=list(COLUMNS)),
        distance.pdist(VALUES, "seuclidean", V=statistics.variance),
        rtol=1e-12,
        atol=1e-12,
    )

    inverse = oefp.inverse_covariance_matrix(_batch(), list(COLUMNS))
    assert inverse.rank == len(COLUMNS)
    mahalanobis = oefp.Metric.mahalanobis(inverse.matrix.ravel())
    np.testing.assert_allclose(
        oefp.pdist(_batch(), mahalanobis, columns=list(COLUMNS)),
        distance.pdist(VALUES, "mahalanobis", VI=inverse.matrix),
        rtol=1e-10,
        atol=1e-12,
    )


def test_numeric_cdist_matches_scipy():
    import oefp

    actual = oefp.cdist(_batch(), _batch(), oefp.Metric.euclidean(), columns=list(COLUMNS))
    expected = distance.cdist(VALUES, VALUES, "euclidean")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_chebyshev_propagates_a_present_nan():
    """A present NaN is data, not a gap, and Chebyshev must not report a finite maximum.

    OEFP used to return 0.0 here: its running maximum came from ``std::max``, which keeps
    its first argument when the comparison is false, so a NaN difference was dropped and
    the answer depended on which row held the NaN.

    scipy is not a single reference for this one. ``distance.chebyshev`` propagates, but
    ``distance.pdist(..., "chebyshev")`` returns 0.0 -- its compiled kernel has the same
    defect. This test therefore pins OEFP against the scalar function, which agrees with
    every other metric on both sides, and asserts the vectorized divergence explicitly so
    that a future scipy release fixing it shows up here rather than silently.
    """
    import oefp

    values = np.array([[1.0, 2.0, 3.0], [np.nan, 2.0, 3.0]], dtype=np.float64)
    actual = oefp.pdist(_batch(values), oefp.Metric.chebyshev(), columns=list(COLUMNS))

    assert np.isnan(distance.chebyshev(values[0], values[1]))
    assert np.isnan(actual[0])

    # Documents scipy's own inconsistency rather than endorsing it.
    assert distance.pdist(values, "chebyshev")[0] == 0.0
