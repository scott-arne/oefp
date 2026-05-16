"""Public API boundary tests."""

from dataclasses import FrozenInstanceError
import importlib.util

import pytest


def test_top_level_api_does_not_export_typemap_smoke_test_helper():
    import oefp

    assert "calculate_molecular_weight" not in oefp.__all__
    assert not hasattr(oefp, "calculate_molecular_weight")


def test_generated_swig_proxy_is_private():
    import oefp

    assert importlib.util.find_spec("oefp._native") is not None
    assert importlib.util.find_spec("oefp.oefp") is None
    assert "_native" not in oefp.__all__


def test_wrappers_do_not_accept_native_objects_in_public_constructors():
    import oefp

    fp = oefp.OEFP.from_on_bits(8, [0, 1], algorithm="unit-test")
    metric = oefp.Metric.tanimoto()

    with pytest.raises(TypeError, match="created by OEFP"):
        oefp.OEFP(fp._native)

    with pytest.raises(TypeError, match="created by Metric"):
        oefp.Metric(metric._native)


def test_fingerprint_spec_is_read_only_public_metadata():
    import oefp

    fp = oefp.OEFP.from_on_bits(8, [0, 1], algorithm="unit-test")

    assert fp.spec.num_bits == 8
    assert fp.spec.value_type == "binary"
    assert fp.spec.source_name == "OEFP"
    assert fp.spec.source_type == "unit-test"
    assert fp.spec.source_version == ""
    assert fp.spec.parameters == ""
    assert fp.spec.source_type_id is None

    with pytest.raises(FrozenInstanceError):
        fp.spec.source_name = "changed"


def test_tanimoto_is_similarity_and_jaccard_is_distance():
    import oefp

    a = oefp.OEFP.from_on_bits(8, [0, 1], algorithm="unit-test")
    b = oefp.OEFP.from_on_bits(8, [1, 2, 3], algorithm="unit-test")
    tanimoto = oefp.Metric.tanimoto()
    jaccard = oefp.Metric.jaccard()

    assert tanimoto.name == "tanimoto"
    assert tanimoto.type == "similarity"
    assert tanimoto.space == "boolean"
    assert jaccard.name == "jaccard"
    assert jaccard.type == "distance"
    assert jaccard.space == "boolean"

    assert oefp.compare(a, b, tanimoto) == pytest.approx(0.25)
    assert oefp.compare(a, b, jaccard) == pytest.approx(0.75)


def test_metric_factories_expose_scikit_style_distances():
    import oefp

    minkowski = oefp.Metric.minkowski(3.0, [1.0, 2.0])
    standardized = oefp.Metric.standardized_euclidean([1.0, 2.0])
    mahalanobis = oefp.Metric.mahalanobis([1.0, 0.0, 0.0, 2.0])

    assert oefp.Metric.euclidean().space == "real"
    assert oefp.Metric.hamming().space == "integer"
    assert oefp.Metric.dice().type == "distance"
    assert minkowski.name == "minkowski"
    assert minkowski.p == 3.0
    assert minkowski.weights == (1.0, 2.0)
    assert standardized.variances == (1.0, 2.0)
    assert mahalanobis.inverse_covariance == (1.0, 0.0, 0.0, 2.0)
