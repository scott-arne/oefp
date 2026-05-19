def test_descriptor_set_typed_access_and_subset():
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float", group="mordred:constitutional"),
            oefp.DescriptorDefinition("nAtom", "int", group="mordred:atom_count"),
            oefp.DescriptorDefinition("Lipinski", "bool", group="mordred:filter"),
        ]
    )
    row = oefp.DescriptorSet(schema, {"MW": 46.069, "nAtom": 9, "Lipinski": True})

    assert row["MW"] == 46.069
    assert row["nAtom"] == 9
    assert row["Lipinski"] is True
    subset = row.subset(["nAtom", "MW"])
    assert subset.schema.names == ("nAtom", "MW")


def test_descriptor_set_preserves_missing_values():
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float"),
            oefp.DescriptorDefinition("nAtom", "int"),
            oefp.DescriptorDefinition("Lipinski", "bool"),
            oefp.DescriptorDefinition("Source", "string"),
        ]
    )
    row = oefp.DescriptorSet(
        schema,
        {"MW": None, "nAtom": None, "Lipinski": None, "Source": None},
    )

    assert row["MW"] is None
    assert row["nAtom"] is None
    assert row["Lipinski"] is None
    assert row["Source"] is None


def test_descriptor_set_rejects_bool_coercion_and_bool_int_overlap():
    import pytest
    import oefp

    bool_schema = oefp.DescriptorSchema([oefp.DescriptorDefinition("Flag", "bool")])
    int_schema = oefp.DescriptorSchema([oefp.DescriptorDefinition("Count", "int")])

    with pytest.raises(TypeError, match="must be a bool"):
        oefp.DescriptorSet(bool_schema, {"Flag": "False"})
    with pytest.raises(TypeError, match="must be an integer"):
        oefp.DescriptorSet(int_schema, {"Count": True})


def test_descriptor_set_legacy_named_access_raises_clear_type_error():
    import pytest
    import oefp

    row = oefp.DescriptorSet.from_strings(["alpha"])

    with pytest.raises(TypeError, match="Legacy descriptor sets"):
        _ = row["alpha"]
