def test_descriptor_batch_round_trips_through_pyarrow(tmp_path):
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float"),
            oefp.DescriptorDefinition("nAtom", "int"),
        ]
    )
    batch = oefp.DescriptorBatch.from_descriptors(
        [oefp.DescriptorSet(schema, {"MW": 46.069, "nAtom": 9})]
    )

    table = batch.to_arrow()
    restored = oefp.DescriptorBatch.from_arrow(table)
    assert restored.float_column("MW")[0] == 46.069

    path = tmp_path / "descriptors.parquet"
    batch.write_parquet(path)
    from_disk = oefp.DescriptorBatch.read_parquet(path)
    assert from_disk.int_column("nAtom")[0] == 9


def test_descriptor_batch_arrow_preserves_nulls_types_and_row_ids(tmp_path):
    import pyarrow as pa
    import oefp

    schema = oefp.DescriptorSchema(
        [
            oefp.DescriptorDefinition("MW", "float"),
            oefp.DescriptorDefinition("nAtom", "int"),
            oefp.DescriptorDefinition("Lipinski", "bool"),
            oefp.DescriptorDefinition("Source", "string"),
            oefp.DescriptorDefinition("AllNull", "int"),
        ]
    )
    batch = oefp.DescriptorBatch.from_descriptors(
        [
            oefp.DescriptorSet(
                schema,
                {
                    "MW": 46.069,
                    "nAtom": 9,
                    "Lipinski": True,
                    "Source": "mordred",
                    "AllNull": None,
                },
                row_id="ethanol",
            ),
            oefp.DescriptorSet(
                schema,
                {
                    "MW": None,
                    "nAtom": None,
                    "Lipinski": None,
                    "Source": None,
                    "AllNull": None,
                },
                row_id="missing",
            ),
        ]
    )

    table = batch.to_arrow()
    assert table.schema.field("MW").type == pa.float64()
    assert table.schema.field("nAtom").type == pa.int64()
    assert table.schema.field("Lipinski").type == pa.bool_()
    assert table.schema.field("Source").type == pa.string()
    assert table.schema.field("AllNull").type == pa.int64()

    restored = oefp.DescriptorBatch.from_arrow(table)
    assert restored.row_ids == ("ethanol", "missing")
    assert restored.int_column("nAtom").tolist() == [9, None]
    assert restored.bool_column("Lipinski").tolist() == [True, None]
    assert restored.string_column("Source") == ("mordred", None)
    assert restored.int_column("AllNull").tolist() == [None, None]

    path = tmp_path / "descriptors-with-nulls.parquet"
    batch.write_parquet(path)
    from_disk = oefp.DescriptorBatch.read_parquet(path)
    assert from_disk.row_ids == ("ethanol", "missing")
    assert from_disk.int_column("AllNull").tolist() == [None, None]


def test_empty_schema_batch_round_trips_through_arrow_and_parquet(tmp_path):
    import oefp

    # A zero-column batch with N>0 rows is a supported state (empty calculator).
    schema = oefp.DescriptorSchema._allow_empty([])
    batch = oefp.DescriptorBatch(
        schema=schema, rows=[{}, {}, {}], row_ids=["a", "b", "c"]
    )
    assert batch.size == 3

    table = batch.to_arrow()
    restored = oefp.DescriptorBatch.from_arrow(table)
    assert len(restored.schema) == 0
    assert restored.size == 3
    assert restored.row_ids == ("a", "b", "c")
    assert list(restored) == [{}, {}, {}]

    path = tmp_path / "empty.parquet"
    batch.write_parquet(path)
    from_disk = oefp.DescriptorBatch.read_parquet(path)
    assert len(from_disk.schema) == 0
    assert from_disk.size == 3
    assert from_disk.row_ids == ("a", "b", "c")


def test_empty_calculator_batch_round_trips_through_arrow(panel_mols):
    import oefp

    calc = oefp.DescriptorCalculator([])
    batch = calc.calculate_batch(panel_mols)
    assert len(batch.schema) == 0
    assert batch.size == len(panel_mols)

    restored = oefp.DescriptorBatch.from_arrow(batch.to_arrow())
    assert len(restored.schema) == 0
    assert restored.size == len(panel_mols)
    assert list(restored) == [{} for _ in panel_mols]


def test_canonical_id_round_trips_through_arrow():
    import oefp

    schema = oefp.DescriptorSchema(
        [oefp.DescriptorDefinition("MW", "float", canonical_id="quantity:exact_molecular_weight")]
    )
    batch = oefp.DescriptorBatch.from_descriptors([oefp.DescriptorSet(schema, {"MW": 46.069})])
    restored = oefp.DescriptorBatch.from_arrow(batch.to_arrow())
    assert restored.schema.definitions[0].canonical_id == "quantity:exact_molecular_weight"
