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


def _batch():
    import oefp

    schema = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition(name, "float") for name in COLUMNS]
    )
    return oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, dict(zip(COLUMNS, row)))
            for row in VALUES.tolist()
        ]
    )


@pytest.mark.parametrize(
    ("build_metric", "scipy_name", "kwargs"),
    [
        (lambda o: o.Metric.euclidean(), "euclidean", {}),
        (lambda o: o.Metric.manhattan(), "cityblock", {}),
        (lambda o: o.Metric.chebyshev(), "chebyshev", {}),
        (lambda o: o.Metric.canberra(), "canberra", {}),
        (lambda o: o.Metric.bray_curtis(), "braycurtis", {}),
        (lambda o: o.Metric.hamming(), "hamming", {}),
        (lambda o: o.Metric.minkowski(3.0), "minkowski", {"p": 3.0}),
    ],
)
def test_numeric_pdist_matches_scipy(build_metric, scipy_name, kwargs):
    import oefp

    actual = oefp.pdist(_batch(), build_metric(oefp), columns=list(COLUMNS))
    expected = distance.pdist(VALUES, scipy_name, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


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
