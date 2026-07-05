"""Guards that same-canonical_id columns are numerically identical across sources.

Compares RAW per-source outputs because the DescriptorCalculator drops duplicate
columns at construction and they are not observable in merged output.
"""

from __future__ import annotations


def _canonical_values(calculator, mol):
    # Python DescriptorSet is subscriptable by name and returns None for a
    # missing (unset) value; a missing tagged value is simply not compared.
    row = calculator.compute(mol)
    out = {}
    for definition in calculator.schema.definitions:
        if definition.canonical_id:
            value = row[definition.name]
            if value is not None:
                out[definition.canonical_id] = value
    return out


def test_shared_canonical_ids_are_identical(panel_mols):
    import oefp

    # Wrap each source in a single-source calculator to access schema and compute
    calculators = [
        oefp.DescriptorCalculator([oefp.MordredDescriptorSource()]),
        oefp.DescriptorCalculator([oefp.OpenEyePropertyDescriptorSource()]),
    ]
    for mol in panel_mols:
        per_source = [_canonical_values(calc, mol) for calc in calculators]
        common = set(per_source[0]).intersection(*[set(d) for d in per_source[1:]])
        assert common, "expected at least one shared canonical_id between sources"
        for canonical_id in common:
            values = [d[canonical_id] for d in per_source]
            first = values[0]
            for other in values[1:]:
                assert other == first, f"canonical_id {canonical_id} diverged: {values}"
