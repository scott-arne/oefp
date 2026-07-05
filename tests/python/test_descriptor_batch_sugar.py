"""Tests for Pythonic ``DescriptorBatch`` sugar."""


def _batch(panel_mols):
    import oefp
    calc = oefp.DescriptorCalculator([oefp.OpenEyePropertyDescriptorSource()])
    return calc, calc.calculate_batch(panel_mols)


def test_iter_and_index_are_row_dicts(panel_mols):
    _, batch = _batch(panel_mols)
    rows = list(batch)
    assert len(batch) == len(panel_mols)   # __len__ already returns row count
    assert len(rows) == len(panel_mols)
    assert isinstance(rows[0], dict)
    assert set(rows[0]) == set(batch.schema.names)
    assert batch[0] == rows[0]
    assert batch[-1] == rows[-1]


def test_to_records_equals_list(panel_mols):
    _, batch = _batch(panel_mols)
    assert batch.to_records() == list(batch)


def test_to_list_equals_list(panel_mols):
    _, batch = _batch(panel_mols)
    assert batch.to_list() == list(batch)


def test_to_dict_is_column_oriented(panel_mols):
    _, batch = _batch(panel_mols)
    columns = batch.to_dict()
    assert set(columns) == set(batch.schema.names)
    assert len(columns["HBA"]) == len(panel_mols)


def test_add_is_vertical_concat(panel_mols):
    _, batch = _batch(panel_mols)
    combined = batch + batch
    assert combined.size == 2 * batch.size


def test_add_rejects_schema_mismatch(panel_mols):
    import oefp
    import pytest

    _, batch = _batch(panel_mols)
    other = oefp.DescriptorCalculator([oefp.MordredDescriptorSource()]).calculate_batch(panel_mols)
    with pytest.raises(ValueError):
        _ = batch + other


def test_slice_returns_batch(panel_mols):
    import oefp
    _, batch = _batch(panel_mols)
    head = batch[:2]
    assert isinstance(head, oefp.DescriptorBatch)
    assert head.size == 2


def test_dict_batch_raises_regardless_of_column_count(panel_mols):
    import oefp
    import pytest

    # Two columns previously slipped through dict() as a bogus {key: key} map
    # because each row dict is a two-element iterable; assert it now raises.
    two_col = oefp.DescriptorCalculator(
        [(oefp.OpenEyePropertyDescriptorSource(), ["XLogP", "HBA"])]
    ).calculate_batch(panel_mols)
    assert len(two_col.schema) == 2
    with pytest.raises(TypeError):
        dict(two_col)

    # Zero columns and three-or-more columns must raise too (column-independent).
    zero_col = oefp.DescriptorCalculator([]).calculate_batch(panel_mols)
    assert len(zero_col.schema) == 0
    with pytest.raises(TypeError):
        dict(zero_col)

    three_col = oefp.DescriptorCalculator(
        [(oefp.OpenEyePropertyDescriptorSource(), ["XLogP", "HBA", "HeavyAtomCount"])]
    ).calculate_batch(panel_mols)
    assert len(three_col.schema) == 3
    with pytest.raises(TypeError):
        dict(three_col)

    # keys() refusing the mapping protocol must not break column form.
    columns = two_col.to_dict()
    assert set(columns) == {"XLogP", "HBA"}
    assert len(columns["HBA"]) == len(panel_mols)
