"""Conformance tests against scikit-learn DistanceMetric formulas."""

from __future__ import annotations

import math

import numpy as np
import pytest

sklearn_metrics = pytest.importorskip(
    "sklearn.metrics",
    reason="scikit-learn is required for DistanceMetric conformance tests",
)


def _assert_close_or_nan(actual: float, expected: float) -> None:
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == pytest.approx(expected)


def test_dense_binary_distances_match_sklearn_distance_metric():
    import oefp

    left = np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
    right = np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    fp_left = oefp.OEFP.from_on_bits(6, [0, 2, 5], algorithm="unit-test")
    fp_right = oefp.OEFP.from_on_bits(6, [0, 1], algorithm="unit-test")
    weights = np.array([1.0, 0.5, 2.0, 1.5, 3.0, 4.0])
    variances = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    inverse_covariance = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    cases = [
        (oefp.Metric.euclidean(), "euclidean", {}),
        (oefp.Metric.manhattan(), "manhattan", {}),
        (oefp.Metric.chebyshev(), "chebyshev", {}),
        (oefp.Metric.minkowski(3.0, weights), "minkowski", {"p": 3.0, "w": weights}),
        (oefp.Metric.standardized_euclidean(variances), "seuclidean", {"V": variances}),
        (
            oefp.Metric.mahalanobis(inverse_covariance.ravel()),
            "mahalanobis",
            {"VI": inverse_covariance},
        ),
        (oefp.Metric.hamming(), "hamming", {}),
        (oefp.Metric.canberra(), "canberra", {}),
        (oefp.Metric.bray_curtis(), "braycurtis", {}),
        (oefp.Metric.jaccard(), "jaccard", {}),
        (oefp.Metric.matching(), "matching", {}),
        (oefp.Metric.dice(), "dice", {}),
        (oefp.Metric.kulsinski(), "kulsinski", {}),
        (oefp.Metric.rogers_tanimoto(), "rogerstanimoto", {}),
        (oefp.Metric.russell_rao(), "russellrao", {}),
        (oefp.Metric.sokal_michener(), "sokalmichener", {}),
        (oefp.Metric.sokal_sneath(), "sokalsneath", {}),
    ]

    for oefp_metric, sklearn_name, kwargs in cases:
        expected = sklearn_metrics.DistanceMetric.get_metric(sklearn_name, **kwargs).pairwise(left, right)[0, 0]
        _assert_close_or_nan(oefp.compare(fp_left, fp_right, oefp_metric), expected)


def test_haversine_distance_matches_sklearn_distance_metric():
    import oefp

    left = np.array([[1.0, 0.0]])
    right = np.array([[0.0, 1.0]])
    fp_left = oefp.OEFP.from_on_bits(2, [0], algorithm="unit-test")
    fp_right = oefp.OEFP.from_on_bits(2, [1], algorithm="unit-test")

    expected = sklearn_metrics.DistanceMetric.get_metric("haversine").pairwise(left, right)[0, 0]

    assert oefp.compare(fp_left, fp_right, oefp.Metric.haversine()) == pytest.approx(expected)
