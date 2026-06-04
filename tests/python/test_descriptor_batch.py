import pytest


def test_descriptor_batch_columns_and_subset():
    import numpy as np
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float", group="mordred:constitutional"),
            oefp.DescriptorDefinition("nAtom", "int", group="mordred:atom_count"),
        ]
    )
    batch = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, {"MW": 46.069, "nAtom": 9}),
            oefp.DescriptorSet(schema, {"MW": 78.114, "nAtom": 12}),
        ]
    )

    np.testing.assert_allclose(batch.float_column("MW"), np.array([46.069, 78.114]))
    np.testing.assert_array_equal(batch.int_column("nAtom"), np.array([9, 12]))
    assert batch.subset(["nAtom"]).schema.names == ("nAtom",)


def test_descriptor_batch_missing_values_and_validity():
    import numpy as np
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float"),
            oefp.DescriptorDefinition("nAtom", "int"),
            oefp.DescriptorDefinition("Lipinski", "bool"),
            oefp.DescriptorDefinition("Source", "string"),
        ]
    )
    batch = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(
                schema,
                {"MW": 46.069, "nAtom": 9, "Lipinski": True, "Source": "mordred"},
            ),
            oefp.DescriptorSet(
                schema,
                {"MW": None, "nAtom": None, "Lipinski": None, "Source": None},
            ),
        ]
    )

    np.testing.assert_allclose(batch.float_column("MW"), np.array([46.069, np.nan]))
    np.testing.assert_array_equal(batch.int_column("nAtom"), np.array([9, None], dtype=object))
    np.testing.assert_array_equal(batch.bool_column("Lipinski"), np.array([True, None], dtype=object))
    assert batch.string_column("Source") == ("mordred", None)
    np.testing.assert_array_equal(batch.column_validity("MW"), np.array([True, False]))
    np.testing.assert_array_equal(batch.column_validity("nAtom"), np.array([True, False]))
    np.testing.assert_array_equal(batch.column_validity("Lipinski"), np.array([True, False]))
    np.testing.assert_array_equal(batch.column_validity("Source"), np.array([True, False]))


def test_descriptor_batch_preserves_and_validates_row_ids():
    import pytest
    import oefp

    schema = oefp.DescriptorSchema([oefp.DescriptorDefinition("MW", "float")])
    batch = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(schema, {"MW": 46.069}, row_id="ethanol"),
            oefp.DescriptorSet(schema, {"MW": 78.114}, row_id="benzene"),
        ]
    )

    assert batch.row_ids == ("ethanol", "benzene")
    with pytest.raises(ValueError, match="row_ids length"):
        oefp.DescriptorBatch(schema=schema, rows=[{"MW": 46.069}], row_ids=[])


def test_descriptor_batch_constructor_validates_row_values():
    import pytest
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("Count", "int"),
            oefp.DescriptorDefinition("Flag", "bool"),
            oefp.DescriptorDefinition("Source", "string"),
        ]
    )

    with pytest.raises(TypeError, match="must be an integer"):
        oefp.DescriptorBatch(
            schema=schema,
            rows=[{"Count": True, "Flag": True, "Source": "mordred"}],
        )
    with pytest.raises(TypeError, match="must be a bool"):
        oefp.DescriptorBatch(
            schema=schema,
            rows=[{"Count": 1, "Flag": "False", "Source": "mordred"}],
        )
    with pytest.raises(TypeError, match="must be a string"):
        oefp.DescriptorBatch(
            schema=schema,
            rows=[{"Count": 1, "Flag": False, "Source": 5}],
        )


def test_descriptor_batch_rejects_same_names_with_different_definitions():
    import pytest
    import oefp

    left = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition("MW", "float", group="mordred:constitutional")]
    )
    right = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition("MW", "float", group="mordred:atom_count")]
    )

    with pytest.raises(ValueError, match="schema does not match"):
        oefp.DescriptorBatch.from_descriptors(
            [
                oefp.DescriptorSet(left, {"MW": 46.069}),
                oefp.DescriptorSet(right, {"MW": 78.114}),
            ]
        )


def test_descriptor_batch_wrong_mode_access_and_compare_raise_clear_type_error():
    import pytest
    import oefp

    legacy = oefp.DescriptorBatch.from_descriptors([oefp.DescriptorSet.from_strings(["alpha"])])
    schema = oefp.DescriptorSchema([oefp.DescriptorDefinition("MW", "float")])
    row = oefp.DescriptorSet(schema, {"MW": 46.069})
    batch = oefp.DescriptorBatch.from_descriptors([row])

    with pytest.raises(TypeError, match="Legacy descriptor batches"):
        _ = legacy.schema
    with pytest.raises(TypeError, match="Schema-backed descriptor batches"):
        _ = batch.value_type
    with pytest.raises(TypeError, match="schema-backed descriptors"):
        oefp.compare(row, row, oefp.Metric.tanimoto())
    with pytest.raises(TypeError, match="schema-backed descriptors"):
        oefp.compare(row, batch, oefp.Metric.tanimoto())
    with pytest.raises(TypeError, match="schema-backed descriptors"):
        oefp.cdist(batch, batch, oefp.Metric.tanimoto())
    with pytest.raises(TypeError, match="schema-backed descriptors"):
        oefp.pdist(batch, oefp.Metric.tanimoto())


def _descriptor_panel_mols():
    from openeye import oechem

    mols = []
    for smiles in ("CCO", "c1ccccc1", "CC(=O)O", "CCN"):
        mol = oechem.OEGraphMol()
        assert oechem.OESmilesToMol(mol, smiles)
        mols.append(mol)
    return mols


def test_descriptor_batch_from_molecules_matches_from_descriptors():
    import numpy as np
    import oefp

    mols = _descriptor_panel_mols()
    direct = oefp.DescriptorBatch.from_molecules(mols, oefp.morgan_descriptors, radius=2)
    manual = oefp.DescriptorBatch.from_descriptors(
        [oefp.morgan_descriptors(m, radius=2) for m in mols]
    )

    assert direct.size == manual.size
    assert direct.value_type == manual.value_type
    assert direct.integer_keys == manual.integer_keys
    np.testing.assert_array_equal(direct.counts, manual.counts)
    np.testing.assert_array_equal(direct.offsets, manual.offsets)


def test_descriptor_batch_from_molecules_forwards_options():
    import oefp

    mols = _descriptor_panel_mols()
    r2 = oefp.DescriptorBatch.from_molecules(mols, oefp.morgan_descriptors, radius=2)
    r3 = oefp.DescriptorBatch.from_molecules(mols, oefp.morgan_descriptors, radius=3)

    # A larger radius produces more environment keys for this panel.
    assert r3.entry_count != r2.entry_count


def test_descriptor_batch_from_molecules_empty_returns_empty_batch():
    import oefp

    batch = oefp.DescriptorBatch.from_molecules([], oefp.morgan_descriptors)
    assert batch.size == 0


def test_descriptor_batch_from_molecules_rejects_wrong_element_type():
    import oefp

    mols = _descriptor_panel_mols()
    with pytest.raises(TypeError, match="DescriptorSet"):
        oefp.DescriptorBatch.from_molecules(mols, oefp.morgan_fingerprint)
