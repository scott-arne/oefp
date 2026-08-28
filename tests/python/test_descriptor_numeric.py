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

    expected = np.array([[1.0, 2.0, 1.0], [4.0, 6.0, 0.0], [np.nan, 6.0, 0.0]])
    np.testing.assert_array_equal(np.isnan(values), np.isnan(expected))
    np.testing.assert_allclose(values[~np.isnan(expected)], expected[~np.isnan(expected)])


def test_to_numeric_matrix_rejects_bad_selections():
    batch = _mixed_batch()
    with pytest.raises(TypeError):
        batch.to_numeric_matrix(["Source"])
    with pytest.raises(ValueError):
        batch.to_numeric_matrix([])
    with pytest.raises(KeyError):
        batch.to_numeric_matrix(["NotAColumn"])


def test_to_numeric_matrix_int_range_boundary():
    """Int values at 2^53 boundary: accepted at boundary, rejected beyond."""
    import oefp

    schema = oefp.DescriptorSchema([oefp.DescriptorDefinition("N", "int")])

    # Exactly 2^53 is accepted and materializes exactly
    at_boundary = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet(schema, {"N": 2**53})]
    )
    values, _ = at_boundary.to_numeric_matrix(["N"])
    assert values[0, 0] == 2**53

    # Negative boundary also accepted
    at_neg_boundary = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet(schema, {"N": -(2**53)})]
    )
    values, _ = at_neg_boundary.to_numeric_matrix(["N"])
    assert values[0, 0] == -(2**53)

    # Beyond boundary is rejected
    beyond_positive = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet(schema, {"N": 2**53 + 1})]
    )
    with pytest.raises(ValueError, match="exceeds 2\\^53"):
        beyond_positive.to_numeric_matrix(["N"])

    beyond_negative = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet(schema, {"N": -(2**53 + 1)})]
    )
    with pytest.raises(ValueError, match="exceeds 2\\^53"):
        beyond_negative.to_numeric_matrix(["N"])

    # Rejection also reaches callers
    with pytest.raises(ValueError, match="exceeds 2\\^53"):
        oefp.column_statistics(beyond_positive, ["N"])


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


def test_unknown_missing_policy_names_both_valid_values():
    import oefp

    batch = _mixed_batch()
    with pytest.raises(ValueError) as error:
        oefp.pdist(batch, oefp.Metric.euclidean(), columns=["nAtom"], missing="drop")

    message = str(error.value)
    assert "propagate" in message
    assert "ignore" in message


def test_default_missing_policy_is_case_insensitive_without_columns():
    """``missing="PROPAGATE"`` names the default, so the columns= guard must accept it."""
    import oefp

    legacy = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet.from_strings(["alpha", "beta"]),
            oefp.DescriptorSet.from_strings(["beta", "gamma"]),
        ]
    )
    metric = oefp.Metric.tanimoto()

    np.testing.assert_allclose(
        oefp.pdist(legacy, metric, missing="propagate"),
        oefp.pdist(legacy, metric, missing="PROPAGATE"),
    )
    np.testing.assert_allclose(
        oefp.cdist(legacy, legacy, metric, missing="propagate"),
        oefp.cdist(legacy, legacy, metric, missing="PROPAGATE"),
    )


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

    # _mixed_batch has MW in rows 0,1 (values 1.0, 4.0) and nAtom in all three (2, 6, 6)
    statistics = oefp.column_statistics(_mixed_batch(), ["MW", "nAtom"])
    assert statistics.names == ("MW", "nAtom")
    np.testing.assert_array_equal(statistics.present_count, np.array([2, 3], dtype=np.uint64))

    # MW column: mean = (1+4)/2 = 2.5, variance = ((1-2.5)^2 + (4-2.5)^2)/1 = 4.5
    assert statistics.mean[0] == pytest.approx(2.5)
    assert statistics.variance[0] == pytest.approx(4.5)
    assert statistics.minimum[0] == pytest.approx(1.0)
    assert statistics.maximum[0] == pytest.approx(4.0)

    # nAtom column: mean = (2+6+6)/3 = 14/3, variance = ((2-14/3)^2 + (6-14/3)^2 + (6-14/3)^2)/2 = 16/3
    assert statistics.mean[1] == pytest.approx(14/3)
    assert statistics.variance[1] == pytest.approx(16/3)
    assert statistics.minimum[1] == pytest.approx(2.0)
    assert statistics.maximum[1] == pytest.approx(6.0)

    covariance = oefp.covariance_matrix(_mixed_batch(), ["MW", "nAtom"])
    # Listwise deletion drops row 2, leaving rows 0,1: (1.0, 2), (4.0, 6)
    # Covariance matrix: [[4.5, 6.0], [6.0, 8.0]]
    assert covariance.row_count == 2
    assert covariance.matrix.shape == (2, 2)
    np.testing.assert_allclose(covariance.matrix, [[4.5, 6.0], [6.0, 8.0]])


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


def test_inverse_covariance_matrix_and_rcond_forwarding():
    """Test inverse covariance matrix, row_count, and rcond parameter."""
    import oefp

    schema = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition(name, "float") for name in ("X", "Y")]
    )
    # Well-conditioned fixture: [(1.0, 2.0), (3.0, 4.0), (5.0, 7.0)]
    # Covariance: [[4.0, 5.0], [5.0, 19/3]], determinant = 1/3
    # Exact inverse: [[19.0, -15.0], [-15.0, 12.0]]
    well_conditioned = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, {"X": 1.0, "Y": 2.0}),
            oefp.DescriptorSet(schema, {"X": 3.0, "Y": 4.0}),
            oefp.DescriptorSet(schema, {"X": 5.0, "Y": 7.0}),
        ]
    )

    # Default rcond yields rank 2 and exact inverse
    result_default = oefp.inverse_covariance_matrix(well_conditioned, ["X", "Y"])
    assert result_default.row_count == 3
    assert result_default.rank == 2
    np.testing.assert_allclose(result_default.matrix, [[19.0, -15.0], [-15.0, 12.0]])

    # rcond=0.001 also yields rank 2 (eigenvalues differ by ~318, cutoff is low enough)
    result_low = oefp.inverse_covariance_matrix(well_conditioned, ["X", "Y"], rcond=0.001)
    assert result_low.rank == 2
    np.testing.assert_allclose(result_low.matrix, [[19.0, -15.0], [-15.0, 12.0]])

    # rcond=0.01 drops the small eigenvalue, yielding rank 1 and different matrix
    result_high = oefp.inverse_covariance_matrix(well_conditioned, ["X", "Y"], rcond=0.01)
    assert result_high.rank == 1
    assert not np.allclose(result_high.matrix, [[19.0, -15.0], [-15.0, 12.0]])


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


def test_cdist_with_columns_computes_asymmetric_numeric_distances():
    """cdist success path with different row counts and missing patterns."""
    import oefp

    # a has 3 rows with row 2 missing MW; b has 2 rows with row 1 missing nAtom
    a = _mixed_batch()
    schema = a.schema
    b = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, {"MW": 2.0, "nAtom": 4, "Lipinski": True, "Source": "p"}),
            oefp.DescriptorSet(schema, {"MW": 3.0, "Lipinski": False, "Source": "q"}),
        ]
    )

    result = oefp.cdist(a, b, oefp.Metric.euclidean(), columns=["MW", "nAtom"])

    # Shape is fixed by Python-side allocation; value/NaN assertions kill transposition
    assert result.shape == (3, 2)

    # Hand-computed Euclidean distances for the three finite pairs:
    # a[0]=(1.0, 2) vs b[0]=(2.0, 4): sqrt((1-2)^2 + (2-4)^2) = sqrt(1+4) = sqrt(5)
    # a[1]=(4.0, 6) vs b[0]=(2.0, 4): sqrt((4-2)^2 + (6-4)^2) = sqrt(4+4) = sqrt(8)
    # All other pairs touch a missing value (a[2] missing MW, b[1] missing nAtom)
    assert result[0, 0] == pytest.approx(math.sqrt(5))
    assert result[1, 0] == pytest.approx(math.sqrt(8))

    # NaN placement kills an a_mask/b_mask swap: with masks swapped, column 0
    # would be NaN and column 1 would have the two finite values.
    assert np.isnan(result[2, :]).all()  # a[2] has missing MW
    assert np.isnan(result[:, 1]).all()  # b[1] has missing nAtom
    assert not np.isnan(result[:2, 0]).any()  # a[0], a[1] vs b[0] are finite


def test_cdist_missing_ignore_rescales_partial_dimensions():
    """cdist with missing="ignore" rescales when dimensions are missing."""
    import oefp

    # Same asymmetric fixture: a has 3 rows, row 2 missing MW; b has 2 rows, row 1 missing nAtom
    a = _mixed_batch()
    schema = a.schema
    b = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, {"MW": 2.0, "nAtom": 4, "Lipinski": True, "Source": "p"}),
            oefp.DescriptorSet(schema, {"MW": 3.0, "Lipinski": False, "Source": "q"}),
        ]
    )

    propagate = oefp.cdist(a, b, oefp.Metric.euclidean(), columns=["MW", "nAtom"])
    ignore = oefp.cdist(a, b, oefp.Metric.euclidean(), columns=["MW", "nAtom"], missing="ignore")

    # Ignore rescales: with 1 of 2 dimensions usable, multiply by total/used = 2
    # a0=(1.0,2)  b0=(2.0,4)  both dims  -> sqrt(1+4)          = sqrt(5)
    # a0=(1.0,2)  b1=(3.0, -) MW only    -> sqrt(2 * (1-3)^2)  = sqrt(8)
    # a1=(4.0,6)  b0=(2.0,4)  both dims  -> sqrt(4+4)          = sqrt(8)
    # a1=(4.0,6)  b1=(3.0, -) MW only    -> sqrt(2 * (4-3)^2)  = sqrt(2)
    # a2=( - ,6)  b0=(2.0,4)  nAtom only -> sqrt(2 * (6-4)^2)  = sqrt(8)
    # a2=( - ,6)  b1=(3.0, -) no common dimension -> NaN
    assert ignore[0, 0] == pytest.approx(math.sqrt(5))
    assert ignore[0, 1] == pytest.approx(math.sqrt(8))
    assert ignore[1, 0] == pytest.approx(math.sqrt(8))
    assert ignore[1, 1] == pytest.approx(math.sqrt(2))
    assert ignore[2, 0] == pytest.approx(math.sqrt(8))
    assert math.isnan(ignore[2, 1])

    # Sanity: propagate differs in 3 of 6 cells
    assert not np.array_equal(propagate, ignore, equal_nan=True)


def test_cdist_rejects_invalid_missing_policy():
    """cdist with columns= validates the missing-value policy."""
    import oefp

    a = _mixed_batch()

    with pytest.raises(ValueError, match="missing-value policy"):
        oefp.cdist(a, a, oefp.Metric.euclidean(), columns=["MW"], missing="drop")


def test_columns_argument_rejects_non_schema_batches():
    """The columns= argument requires schema-backed batches."""
    import oefp

    legacy = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet.from_strings(["alpha", "beta"])]
    )

    with pytest.raises(TypeError, match="schema-backed"):
        oefp.pdist(legacy, oefp.Metric.euclidean(), columns=["alpha"])

    with pytest.raises(TypeError, match="schema-backed"):
        oefp.cdist(legacy, legacy, oefp.Metric.euclidean(), columns=["alpha"])


def test_cdist_columns_rejects_non_descriptor_batch_inputs():
    """pdist and cdist with columns= require DescriptorBatch inputs."""
    oechem = pytest.importorskip("openeye.oechem")
    import oefp

    mol = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol, "CCO")
    fingerprint_batch = oefp.OEFPBatch.from_molecules([mol], oefp.morgan_fingerprint)

    with pytest.raises(TypeError, match="only valid for a DescriptorBatch"):
        oefp.pdist(fingerprint_batch, oefp.Metric.euclidean(), columns=["MW"])

    with pytest.raises(TypeError, match="only valid for DescriptorBatch inputs"):
        oefp.cdist(fingerprint_batch, fingerprint_batch, oefp.Metric.euclidean(), columns=["MW"])


def test_to_numeric_matrix_preserves_width_for_zero_rows():
    """Zero-row batch returns the correct shape, not collapsed."""
    import oefp

    schema = _mixed_batch().schema
    # from_descriptors([]) carries no schema; raw constructor is needed for a
    # schema-backed zero-row batch.
    empty_batch = oefp.DescriptorBatch(schema=schema, rows=[], row_ids=[])

    values, validity = empty_batch.to_numeric_matrix(["MW", "nAtom"])

    # Width must survive; (0,) would be a collapse
    assert values.shape == (0, 2)
    assert validity.shape == (0, 2)
    assert values.dtype == np.float64
    assert validity.dtype == np.bool_

    # Exercise the address path with zero-length allocation
    result = oefp.pdist(empty_batch, oefp.Metric.euclidean(), columns=["MW", "nAtom"])
    assert result.shape == (0,)


def test_missing_policy_is_validated_before_matrix_work():
    """Bad missing= is reported before column errors."""
    import oefp

    batch = _mixed_batch()

    # Source is a string column, so to_numeric_matrix would raise TypeError.
    # But the missing policy is bad, so that ValueError should come first.
    with pytest.raises(ValueError, match="missing-value policy"):
        oefp.pdist(batch, oefp.Metric.euclidean(), columns=["Source"], missing="typo")


def test_pdist_rejects_silently_ignored_argument_combinations():
    """Reject descriptor_mode= with columns= and missing= without columns=."""
    import oefp

    batch = _mixed_batch()
    legacy = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet.from_strings(["alpha", "beta"])]
    )

    # descriptor_mode is ignored on the numeric path
    with pytest.raises(TypeError, match="descriptor_mode"):
        oefp.pdist(
            batch, oefp.Metric.euclidean(), columns=["MW"], descriptor_mode="exact_count"
        )

    # missing is ignored on the legacy path
    with pytest.raises(TypeError, match="missing="):
        oefp.pdist(legacy, oefp.Metric.euclidean(), missing="ignore")


def test_cdist_rejects_silently_ignored_argument_combinations():
    """Reject descriptor_mode= with columns= and missing= without columns=."""
    import oefp

    a = _mixed_batch()
    b = _mixed_batch()
    legacy = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet.from_strings(["alpha", "beta"])]
    )

    # descriptor_mode is ignored on the numeric path
    with pytest.raises(TypeError, match="descriptor_mode"):
        oefp.cdist(
            a, b, oefp.Metric.euclidean(), columns=["MW"], descriptor_mode="exact_count"
        )

    # missing is ignored on the legacy path
    with pytest.raises(TypeError, match="missing="):
        oefp.cdist(legacy, legacy, oefp.Metric.euclidean(), missing="ignore")


def test_numeric_rejection_names_the_column_rather_than_its_index():
    """The Python wrappers thread the resolved names, as the C++ batch overloads do."""
    import oefp

    batch = _mixed_batch()
    metric = oefp.Metric.standardized_euclidean([1.0, 0.0])

    with pytest.raises(RuntimeError) as pairwise:
        oefp.pdist(batch, metric, columns=["MW", "nAtom"])
    assert "'nAtom'" in str(pairwise.value)
    assert "index" not in str(pairwise.value)

    with pytest.raises(RuntimeError) as cross:
        oefp.cdist(batch, batch, metric, columns=["MW", "nAtom"])
    assert "'nAtom'" in str(cross.value)
    assert "index" not in str(cross.value)
