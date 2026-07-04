"""Regression guard for SWIG value marshalling of integer vector elements.

SWIG's ``python/std_common.i`` registers numeric *value* traits (so that
``std::vector`` elements convert to Python ``int`` / ``float`` rather than
opaque ``SwigPyObject`` pointer proxies) only for the fundamental primitive
spellings such as ``long long`` and ``unsigned long long``. Instantiating a
vector over a typedef spelling such as ``std::int64_t`` / ``std::uint64_t``
formerly fell through to ``std_vector.i``'s pointer fallback, so element access
yielded ``<Swig Object of type 'long long *'>`` values and leaked. ``oefp.i``
pre-registers value traits for those exact typedef spellings; these tests lock
that behavior in for the vector element getters that back
``DescriptorBatch.IntColumn`` / ``BoolColumn`` / ``ColumnValidity``.
"""

import pytest

from oefp import _native


def _is_swig_proxy(value: object) -> bool:
    """Return whether a value is an un-marshalled SWIG pointer proxy."""
    return type(value).__name__ == "SwigPyObject"


def test_int64_vector_marshals_python_ints():
    vector = _native.Int64Vector()
    for value in (-9223372036854775808, -1, 0, 7, 9223372036854775807):
        vector.push_back(value)
    values = list(vector)
    assert values == [-9223372036854775808, -1, 0, 7, 9223372036854775807]
    assert all(isinstance(value, int) and not _is_swig_proxy(value) for value in values)


def test_uint64_vector_marshals_python_ints():
    vector = _native.UInt64Vector()
    for value in (0, 7, 18446744073709551615):
        vector.push_back(value)
    values = list(vector)
    assert values == [0, 7, 18446744073709551615]
    assert all(isinstance(value, int) and not _is_swig_proxy(value) for value in values)


def test_uint8_vector_marshals_python_ints():
    vector = _native.UInt8Vector()
    for value in (0, 1, 255):
        vector.push_back(value)
    values = list(vector)
    assert values == [0, 1, 255]
    assert all(isinstance(value, int) and not _is_swig_proxy(value) for value in values)


def test_double_vector_still_marshals_python_floats():
    vector = _native.DoubleVector()
    vector.push_back(7)
    values = list(vector)
    assert values == [7.0]
    assert all(isinstance(value, float) for value in values)


def test_uint32_vector_still_marshals_python_ints():
    vector = _native.UInt32Vector()
    vector.push_back(7)
    values = list(vector)
    assert values == [7]
    assert all(isinstance(value, int) and not _is_swig_proxy(value) for value in values)


@pytest.fixture(scope="module")
def int_bool_batch():
    """Build a native schema-backed batch with an int and a bool column."""
    builder = _native.DescriptorSchemaBuilder()
    int_def = _native.DescriptorDefinition()
    int_def.name = "nAtom"
    int_def.value_kind = _native.DescriptorValueKind_Int
    bool_def = _native.DescriptorDefinition()
    bool_def.name = "Lipinski"
    bool_def.value_kind = _native.DescriptorValueKind_Bool
    builder.Add(int_def)
    builder.Add(bool_def)
    schema = builder.Build()

    def row(natom, lipinski, row_id):
        row_builder = _native.DescriptorSetBuilder(schema)
        if natom is not None:
            row_builder.Set("nAtom", _native.DescriptorValue.Int(natom))
        if lipinski is not None:
            row_builder.Set("Lipinski", _native.DescriptorValue.Bool(lipinski))
        return row_builder.Build(row_id)

    sets = _native.DescriptorSetVector()
    sets.push_back(row(9, True, "r0"))
    sets.push_back(row(-12, False, "r1"))
    sets.push_back(row(None, None, "r2"))
    return _native._NativeDescriptorBatch.FromDescriptorSets(sets)


def test_descriptor_batch_int_column_yields_ints(int_bool_batch):
    values = list(int_bool_batch.IntColumn("nAtom"))
    assert values == [9, -12, 0]
    assert all(isinstance(value, int) and not _is_swig_proxy(value) for value in values)


def test_descriptor_batch_bool_column_yields_ints(int_bool_batch):
    values = list(int_bool_batch.BoolColumn("Lipinski"))
    assert values == [1, 0, 0]
    assert all(isinstance(value, int) and not _is_swig_proxy(value) for value in values)


def test_descriptor_batch_column_validity_yields_ints(int_bool_batch):
    int_validity = list(int_bool_batch.ColumnValidity("nAtom"))
    bool_validity = list(int_bool_batch.ColumnValidity("Lipinski"))
    assert int_validity == [1, 1, 0]
    assert bool_validity == [1, 1, 0]
    assert all(isinstance(value, int) and not _is_swig_proxy(value) for value in int_validity)
    assert all(isinstance(value, int) and not _is_swig_proxy(value) for value in bool_validity)
