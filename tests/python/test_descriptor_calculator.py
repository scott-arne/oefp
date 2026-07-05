"""Tests for the public ``DescriptorCalculator`` wrapper."""


def test_calculator_dedup_first_wins(panel_mols):
    import oefp

    calc = oefp.DescriptorCalculator(
        [oefp.MordredDescriptorSource(), oefp.OpenEyePropertyDescriptorSource()]
    )
    names = calc.schema.names
    assert "MW" in names                    # Mordred first keeps MW
    assert "MolecularWeight" not in names    # OpenEye tagged duplicate dropped
    assert "HBA" not in names                # HBA is tagged -> dropped when Mordred is first
    assert "XLogP" in names                  # OpenEye-unique untagged column survives

    row = calc.compute(panel_mols[0])
    assert row.schema == calc.schema

    batch = calc.calculate_batch(panel_mols)
    assert batch.size == len(panel_mols)


def test_calculate_batch_reads_int_and_float_columns(panel_mols):
    import oefp

    calc = oefp.DescriptorCalculator([oefp.OpenEyePropertyDescriptorSource()])
    batch = calc.calculate_batch(panel_mols)
    assert batch.size == len(panel_mols)

    rows = list(batch)
    # Integer column reads back as native Python int (the previously-blocked path).
    hba = [row["HBA"] for row in rows]
    assert hba == [1, 0, 1, 1]
    assert all(type(value) is int for value in hba)

    heavy = [row["HeavyAtomCount"] for row in rows]
    assert heavy == [3, 6, 4, 3]
    assert all(type(value) is int for value in heavy)

    # Float columns read back as native Python float.
    assert all(type(row["XLogP"]) is float for row in rows)
    assert all(type(row["MolecularWeight"]) is float for row in rows)


def test_calculate_batch_marks_missing_values_as_none(panel_mols):
    import oefp

    calc = oefp.DescriptorCalculator([oefp.MordredDescriptorSource()])
    batch = calc.calculate_batch(panel_mols)
    rows = list(batch)

    missing_seen = False
    for name in batch.schema.names:
        validity = batch.column_validity(name)
        for row, present in zip(rows, validity, strict=True):
            if not present:
                assert row[name] is None
                missing_seen = True
    assert missing_seen


def test_calculator_with_names_selection(panel_mols):
    import oefp

    calc = oefp.DescriptorCalculator(
        [(oefp.OpenEyePropertyDescriptorSource(), ["XLogP", "HBA"])]
    )
    assert calc.schema.names == ("XLogP", "HBA")

    batch = calc.calculate_batch(panel_mols)
    assert set(batch.schema.names) == {"XLogP", "HBA"}
    assert [row["HBA"] for row in batch] == [1, 0, 1, 1]


def test_empty_calculator_has_empty_schema_and_batch(panel_mols):
    import oefp

    calc = oefp.DescriptorCalculator([])
    assert len(calc.schema) == 0
    batch = calc.calculate_batch(panel_mols)
    assert batch.size == len(panel_mols)
    assert list(batch) == [{} for _ in panel_mols]  # zero-column rows


def test_fully_selected_empty_calculator(panel_mols):
    import oefp

    # An empty names selection keeps zero columns from the source.
    calc = oefp.DescriptorCalculator([(oefp.OpenEyePropertyDescriptorSource(), [])])
    assert len(calc.schema) == 0
    row = calc.compute(panel_mols[0])
    assert row.schema == calc.schema
