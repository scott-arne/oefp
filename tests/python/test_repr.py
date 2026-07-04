"""Tests for the concise ``__repr__`` and ``__len__`` dunders on public wrappers."""

from __future__ import annotations


def test_oefp_repr_reports_num_bits_popcount_and_source_type():
    import oefp

    fp = oefp.OEFP.from_on_bits(8, [0, 1, 5], algorithm="unit-test")

    assert repr(fp) == "OEFP(num_bits=8, popcount=3, source_type='unit-test')"
    # str() inherits __repr__.
    assert str(fp) == repr(fp)


def test_single_fingerprint_reprs(aspirin_mol):
    import oefp

    dense = oefp.morgan_fingerprint(aspirin_mol)
    sparse = oefp.morgan_sparse_fingerprint(aspirin_mol)
    count = oefp.morgan_count_fingerprint(aspirin_mol)

    dense_repr = repr(dense)
    assert dense_repr.startswith("OEFP(num_bits=2048, popcount=")
    assert "source_type='Morgan'" in dense_repr

    # Sparse and count fingerprints intentionally omit num_bits: it is the
    # unfolded identifier domain (often the full uint64 space), not a fold size.
    sparse_repr = repr(sparse)
    assert sparse_repr.startswith("OEFPSparse(popcount=")
    assert "source_type='Morgan'" in sparse_repr
    assert "num_bits" not in sparse_repr

    count_repr = repr(count)
    assert count_repr.startswith("OEFPCount(nonzero_count=")
    assert "total_count=" in count_repr
    assert "num_bits" not in count_repr


def test_oefp_count64_repr(aspirin_mol):
    import oefp

    count64 = oefp.topological_torsions_sparse_count_fingerprint(aspirin_mol)

    count64_repr = repr(count64)
    assert count64_repr.startswith("OEFPCount64(nonzero_count=")
    assert "total_count=" in count64_repr
    assert "source_type='TopologicalTorsions'" in count64_repr
    assert "num_bits" not in count64_repr


def test_batch_reprs_and_len(panel_mols):
    import oefp

    dense = oefp.OEFPBatch.from_molecules(panel_mols, oefp.morgan_fingerprint)
    count = oefp.OEFPCountBatch.from_molecules(panel_mols, oefp.morgan_count_fingerprint)
    sparse = oefp.OEFPSparseBatch.from_molecules(panel_mols, oefp.morgan_sparse_fingerprint)

    assert repr(dense) == "OEFPBatch(size=4, num_bits=2048, source_type='Morgan')"
    assert len(dense) == dense.size == 4

    count_repr = repr(count)
    assert count_repr.startswith("OEFPCountBatch(size=4, entry_count=")
    assert "source_type='Morgan'" in count_repr
    assert "num_bits" not in count_repr
    assert len(count) == count.size == 4

    sparse_repr = repr(sparse)
    assert sparse_repr.startswith("OEFPSparseBatch(size=4, entry_count=")
    assert "source_type='Morgan'" in sparse_repr
    assert "num_bits" not in sparse_repr
    assert len(sparse) == sparse.size == 4


def test_empty_batch_repr_and_len_do_not_raise():
    import oefp

    empty = oefp.OEFPBatch.from_fingerprints([])

    assert repr(empty) == "OEFPBatch(size=0, num_bits=0, source_type='')"
    assert len(empty) == 0


def test_descriptor_schema_repr_and_len():
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("mw", "float"),
            oefp.DescriptorDefinition("hba", "int"),
        ]
    )

    assert repr(schema) == "DescriptorSchema(columns=2)"
    assert len(schema) == 2


def test_schema_backed_descriptor_set_repr():
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("mw", "float"),
            oefp.DescriptorDefinition("hba", "int"),
        ]
    )

    with_id = oefp.DescriptorSet(schema, {"mw": 180.16, "hba": 4}, row_id="aspirin")
    without_id = oefp.DescriptorSet(schema, {"mw": 46.07, "hba": 1})

    assert repr(with_id) == "DescriptorSet(columns=2, row_id='aspirin')"
    # row_id is omitted when empty.
    assert repr(without_id) == "DescriptorSet(columns=2)"


def test_legacy_descriptor_set_repr():
    import oefp

    legacy = oefp.DescriptorSet.from_integers([2, 7, 11], counts=[4, 1, 3])

    assert repr(legacy) == "DescriptorSet(value_type='integer', size=3, total_count=8)"


def test_descriptor_batch_reprs_and_len():
    import oefp

    schema = oefp.DescriptorSchema([oefp.DescriptorDefinition("mw", "float")])
    schema_batch = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, {"mw": 180.16}),
            oefp.DescriptorSet(schema, {"mw": 46.07}),
        ]
    )

    assert repr(schema_batch) == "DescriptorBatch(size=2, columns=1)"
    assert len(schema_batch) == 2

    legacy_batch = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet.from_integers([1, 2, 3], counts=[4, 5, 6])]
    )

    assert repr(legacy_batch) == "DescriptorBatch(size=1, value_type='integer')"
    assert len(legacy_batch) == 1


def test_metric_repr_non_parametric_and_tversky():
    import oefp

    assert repr(oefp.Metric.tanimoto()) == "Metric(name='tanimoto', type='similarity')"

    tversky_repr = repr(oefp.Metric.tversky(0.3, 0.7))
    assert tversky_repr.startswith("Metric(name='tversky', type='similarity'")
    assert "alpha=0.3" in tversky_repr
    assert "beta=0.7" in tversky_repr


def test_generator_reprs_use_correct_class_name():
    import oefp

    assert repr(oefp.MorganGenerator(radius=3, num_bits=1024)) == (
        "MorganGenerator(radius=3, num_bits=1024)"
    )
    assert repr(oefp.TopologicalAtomPairGenerator()) == (
        "TopologicalAtomPairGenerator(min_distance=1, max_distance=30, num_bits=2048)"
    )
    assert repr(oefp.TopologicalTorsionsGenerator()) == (
        "TopologicalTorsionsGenerator(torsion_atom_count=4, num_bits=2048)"
    )
    # AtomPairGenerator inherits its repr but must report its own class name.
    assert repr(oefp.AtomPairGenerator()) == (
        "AtomPairGenerator(min_distance=1, max_distance=30, num_bits=2048)"
    )
