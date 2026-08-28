"""Metric capability predicates and the degenerate-input conventions they depend on."""

import math

import pytest

# Every metric factory that takes no arguments, with the capability answers it must report.
# Parameterized metrics are covered separately because their answers assume valid parameters.
CAPABILITIES = [
    # (factory name, has_zero_self_distance, satisfies_triangle_inequality)
    ("euclidean", True, True),
    ("manhattan", True, True),
    ("chebyshev", True, True),
    ("haversine", True, True),
    ("hamming", True, True),
    ("canberra", True, True),
    ("bray_curtis", True, False),
    ("jaccard", True, True),
    ("matching", True, True),
    ("dice", True, False),
    ("kulsinski", False, True),
    ("rogers_tanimoto", True, True),
    ("russell_rao", False, True),
    ("sokal_michener", True, True),
    ("sokal_sneath", True, True),
    ("tanimoto", False, False),
]


@pytest.mark.parametrize(("factory", "zero_self", "triangle"), CAPABILITIES)
def test_capability_predicates_report_the_documented_table(factory, zero_self, triangle):
    import oefp

    metric = getattr(oefp.Metric, factory)()

    assert metric.has_zero_self_distance is zero_self
    assert metric.satisfies_triangle_inequality is triangle
    assert metric.is_symmetric is True


def test_parameterized_metric_capabilities_assume_valid_parameters():
    import oefp

    minkowski_sub_one = oefp.Metric.minkowski(0.5)
    assert minkowski_sub_one.has_zero_self_distance is True
    assert minkowski_sub_one.satisfies_triangle_inequality is False

    for p in (1.0, 1.5, 3.0):
        assert oefp.Metric.minkowski(p).satisfies_triangle_inequality is True

    # Weights scale each term and change neither answer.
    assert oefp.Metric.minkowski(0.5, [1.0, 2.0]).satisfies_triangle_inequality is False
    assert oefp.Metric.minkowski(2.0, [1.0, 2.0]).satisfies_triangle_inequality is True

    standardized = oefp.Metric.standardized_euclidean([1.0, 2.0])
    assert standardized.has_zero_self_distance is True
    assert standardized.satisfies_triangle_inequality is True

    mahalanobis = oefp.Metric.mahalanobis([1.0, 0.0, 0.0, 1.0])
    assert mahalanobis.has_zero_self_distance is True
    assert mahalanobis.satisfies_triangle_inequality is True

    symmetric_tversky = oefp.Metric.tversky(0.5, 0.5)
    assert symmetric_tversky.is_symmetric is True
    assert symmetric_tversky.has_zero_self_distance is False
    assert symmetric_tversky.satisfies_triangle_inequality is False

    assert oefp.Metric.tversky(0.2, 0.8).is_symmetric is False


def test_validate_as_distance_metric_accepts_only_true_metrics():
    import oefp

    for factory in ("euclidean", "manhattan", "chebyshev", "hamming", "canberra",
                    "jaccard", "matching", "rogers_tanimoto", "sokal_michener",
                    "sokal_sneath"):
        getattr(oefp.Metric, factory)().validate_as_distance_metric()

    oefp.Metric.minkowski(1.0).validate_as_distance_metric()

    with pytest.raises(RuntimeError, match="Similarity metrics"):
        oefp.Metric.tanimoto().validate_as_distance_metric()
    with pytest.raises(RuntimeError, match="Similarity metrics"):
        oefp.Metric.tversky(0.5, 0.5).validate_as_distance_metric()
    with pytest.raises(RuntimeError, match="zero self-distance"):
        oefp.Metric.kulsinski().validate_as_distance_metric()
    with pytest.raises(RuntimeError, match="zero self-distance"):
        oefp.Metric.russell_rao().validate_as_distance_metric()
    with pytest.raises(RuntimeError, match="triangle inequality"):
        oefp.Metric.dice().validate_as_distance_metric()
    with pytest.raises(RuntimeError, match="triangle inequality"):
        oefp.Metric.bray_curtis().validate_as_distance_metric()
    with pytest.raises(RuntimeError, match="triangle inequality"):
        oefp.Metric.minkowski(0.5).validate_as_distance_metric()


def test_zero_self_distance_holds_for_two_empty_fingerprints():
    import oefp

    empty = oefp.OEFP.from_on_bits(8, [], algorithm="unit-test")

    # The degenerate all-zero pair makes every boolean quotient 0/0. Metrics that claim
    # zero self-distance must still return 0.0 there rather than NaN.
    for factory in ("jaccard", "matching", "dice", "rogers_tanimoto",
                    "sokal_michener", "sokal_sneath"):
        metric = getattr(oefp.Metric, factory)()
        assert metric.has_zero_self_distance is True
        assert oefp.compare(empty, empty, metric) == 0.0


def test_nested_sets_violate_the_triangle_inequality_at_every_width():
    import oefp

    # Random fingerprints do not expose this: sweeps at 64 and 2048 bits found zero Dice
    # violations. The violation needs nested sets, and it is dimension-independent.
    for width in (2, 8, 64, 2048):
        a = oefp.OEFP.from_on_bits(width, [0], algorithm="unit-test")
        b = oefp.OEFP.from_on_bits(width, [0, 1], algorithm="unit-test")
        c = oefp.OEFP.from_on_bits(width, [1], algorithm="unit-test")

        for factory, args in (("dice", ()), ("bray_curtis", ()), ("minkowski", (0.5,))):
            metric = getattr(oefp.Metric, factory)(*args)
            direct = oefp.compare(a, c, metric)
            detour = oefp.compare(a, b, metric) + oefp.compare(b, c, metric)

            assert direct > detour, f"{factory} at width {width}"
            assert metric.satisfies_triangle_inequality is False


def test_mahalanobis_rejects_a_negative_quadratic_form():
    import oefp

    a = oefp.OEFP.from_on_bits(2, [0], algorithm="unit-test")
    b = oefp.OEFP.from_on_bits(2, [1], algorithm="unit-test")

    with pytest.raises(RuntimeError, match="positive semidefinite"):
        oefp.compare(a, b, oefp.Metric.mahalanobis([-1.0, 0.0, 0.0, -1.0]))
    with pytest.raises(RuntimeError, match="positive semidefinite"):
        oefp.compare(a, b, oefp.Metric.mahalanobis([1.0, 0.0, 0.0, -4.0]))

    assert oefp.compare(a, b, oefp.Metric.mahalanobis([1.0, 0.0, 0.0, 1.0])) == pytest.approx(
        math.sqrt(2.0)
    )

    # NaN is not a negative value, so the guard leaves genuine NaN propagation alone.
    nan = float("nan")
    assert math.isnan(oefp.compare(a, b, oefp.Metric.mahalanobis([nan, 0.0, 0.0, 1.0])))
