"""Behaviour tests for numeric descriptor comparison and statistics."""

from __future__ import annotations

import math

import numpy as np
import pytest


def _mixed_batch():
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float"),
            oefp.DescriptorDefinition("nAtom", "int"),
            oefp.DescriptorDefinition("Lipinski", "bool"),
            oefp.DescriptorDefinition("Source", "string"),
        ]
    )
    return oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, {"MW": 1.0, "nAtom": 2, "Lipinski": True, "Source": "x"}),
            oefp.DescriptorSet(schema, {"MW": 4.0, "nAtom": 6, "Lipinski": False, "Source": "y"}),
            oefp.DescriptorSet(schema, {"nAtom": 6, "Lipinski": False, "Source": "z"}),
        ]
    )


def test_to_numeric_matrix_widens_and_marks_missing():
    values, validity = _mixed_batch().to_numeric_matrix(["MW", "nAtom", "Lipinski"])

    assert values.dtype == np.float64
    assert validity.dtype == np.bool_
    assert values.flags["C_CONTIGUOUS"]
    assert values.shape == (3, 3)
    np.testing.assert_array_equal(validity[2], np.array([False, True, True]))
    assert math.isnan(values[2, 0])
    np.testing.assert_allclose(values[0], np.array([1.0, 2.0, 1.0]))
    np.testing.assert_allclose(values[1], np.array([4.0, 6.0, 0.0]))


def test_to_numeric_matrix_rejects_bad_selections():
    batch = _mixed_batch()
    with pytest.raises(TypeError):
        batch.to_numeric_matrix(["Source"])
    with pytest.raises(ValueError):
        batch.to_numeric_matrix([])
    with pytest.raises(KeyError):
        batch.to_numeric_matrix(["NotAColumn"])


def test_propagate_and_ignore_differ_on_a_missing_value():
    import oefp

    batch = _mixed_batch()
    propagate = oefp.pdist(batch, oefp.Metric.euclidean(), columns=["MW", "nAtom"])
    ignore = oefp.pdist(
        batch, oefp.Metric.euclidean(), columns=["MW", "nAtom"], missing="ignore"
    )

    # Pairs (0,2) and (1,2) both touch the missing MW of row 2.
    assert not math.isnan(propagate[0])
    assert math.isnan(propagate[1])
    assert math.isnan(propagate[2])
    assert not math.isnan(ignore[1])
    # Row 1 and row 2 share nAtom = 6, so the one usable dimension is identical.
    assert ignore[2] == pytest.approx(0.0)


def test_missing_policy_is_case_insensitive_and_validated():
    import oefp

    batch = _mixed_batch()
    lower = oefp.pdist(batch, oefp.Metric.euclidean(), columns=["nAtom"], missing="ignore")
    upper = oefp.pdist(batch, oefp.Metric.euclidean(), columns=["nAtom"], missing="IGNORE")
    np.testing.assert_allclose(lower, upper)

    with pytest.raises(ValueError):
        oefp.pdist(batch, oefp.Metric.euclidean(), columns=["nAtom"], missing="drop")


def test_columns_omitted_still_raises_the_reworded_error():
    import oefp

    with pytest.raises(TypeError, match="columns="):
        oefp.pdist(_mixed_batch(), oefp.Metric.euclidean())
    with pytest.raises(TypeError, match="columns="):
        oefp.compare(
            oefp.DescriptorSet(_mixed_batch().schema, {"MW": 1.0}),
            _mixed_batch(),
            oefp.Metric.euclidean(),
        )


def _rebranded_batch():
    """Same column names and value types as `_mixed_batch`, different schema identifier."""
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float", source_version="2.0"),
            oefp.DescriptorDefinition("nAtom", "int"),
            oefp.DescriptorDefinition("Lipinski", "bool"),
            oefp.DescriptorDefinition("Source", "string"),
        ]
    )
    return oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, {"MW": 2.0, "nAtom": 3, "Lipinski": True, "Source": "p"}),
        ]
    )


def test_cdist_rejects_batches_with_different_schema_ids():
    import oefp

    a = _mixed_batch()
    b = _rebranded_batch()
    # The mismatch is metadata only: the selection resolves to the same names and the
    # same width on both sides, so nothing downstream of the check would have caught it.
    assert a.schema.names == b.schema.names
    assert a.schema.schema_id != b.schema.schema_id
    assert a.to_numeric_matrix(["MW", "nAtom"])[0].shape[1] == 2
    assert b.to_numeric_matrix(["MW", "nAtom"])[0].shape[1] == 2

    with pytest.raises(ValueError, match="schema identifier"):
        oefp.cdist(a, b, oefp.Metric.euclidean(), columns=["MW", "nAtom"])


def test_statistics_respect_the_validity_mask():
    import oefp

    statistics = oefp.column_statistics(_mixed_batch(), ["MW", "nAtom"])
    assert statistics.names == ("MW", "nAtom")
    np.testing.assert_array_equal(statistics.present_count, np.array([2, 3], dtype=np.uint64))
    assert statistics.mean[0] == pytest.approx(2.5)
    assert statistics.minimum[0] == pytest.approx(1.0)
    assert statistics.maximum[0] == pytest.approx(4.0)

    covariance = oefp.covariance_matrix(_mixed_batch(), ["MW", "nAtom"])
    # Listwise deletion drops row 2, leaving two complete rows.
    assert covariance.row_count == 2
    assert covariance.matrix.shape == (2, 2)
    assert covariance.matrix[0, 0] == pytest.approx(4.5)


def test_inverse_covariance_reports_rank_and_rejects_a_constant_column():
    import oefp

    schema = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition(name, "float") for name in ("A", "B")]
    )
    constant = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet(schema, {"A": 1.0, "B": 1.0}) for _ in range(3)]
    )
    with pytest.raises(ValueError):
        oefp.inverse_covariance_matrix(constant, ["A", "B"])

    collinear = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet(schema, {"A": v, "B": 2.0 * v}) for v in (1.0, 2.0, 3.0)]
    )
    assert oefp.inverse_covariance_matrix(collinear, ["A", "B"]).rank == 1


def test_the_feature_request_reproduction_now_succeeds():
    """Section 4 of the source request, with columns= supplied."""
    oechem = pytest.importorskip("openeye.oechem")
    import oefp

    smiles = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCCCC"]
    mols = []
    for text in smiles:
        mol = oechem.OEGraphMol()
        oechem.OESmilesToMol(mol, text)
        mols.append(mol)

    calculator = oefp.DescriptorCalculator([oefp.OpenEyePropertyDescriptorSource()])
    batch = calculator.calculate_batch(mols)
    columns = ["MolecularWeight", "HeavyAtomCount", "TopologicalPSA", "XLogP"]

    result = oefp.pdist(batch, oefp.Metric.euclidean(), columns=columns)
    assert result.shape == (6,)
    assert np.all(np.isfinite(result))
    assert result[0] > 0.0
