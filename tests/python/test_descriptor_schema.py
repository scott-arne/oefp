def test_descriptor_schema_lookup_and_group_selection():
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float", group="mordred:constitutional"),
            oefp.DescriptorDefinition("nAtom", "int", group="mordred:atom_count"),
        ]
    )

    assert schema.index("MW") == 0
    assert schema.group("mordred:constitutional") == (0,)
    assert schema.names == ("MW", "nAtom")


def test_descriptor_schema_equality_includes_definition_metadata():
    import oefp

    base = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition("MW", "float", group="mordred:constitutional")]
    )
    same = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition("MW", "float", group="mordred:constitutional")]
    )
    different_group = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition("MW", "float", group="mordred:atom_count")]
    )
    different_type = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition("MW", "int", group="mordred:constitutional")]
    )

    assert base == same
    assert base.schema_id == same.schema_id
    assert base != different_group
    assert base.schema_id != different_group.schema_id
    assert base != different_type


def test_canonical_id_changes_schema_id_and_round_trips():
    from oefp import DescriptorDefinition, DescriptorSchema

    plain = DescriptorSchema([DescriptorDefinition("MW", "float", group="g")])
    tagged = DescriptorSchema(
        [DescriptorDefinition("MW", "float", group="g", canonical_id="quantity:exact_molecular_weight")]
    )
    assert plain.schema_id != tagged.schema_id
    assert tagged.definitions[0].canonical_id == "quantity:exact_molecular_weight"
