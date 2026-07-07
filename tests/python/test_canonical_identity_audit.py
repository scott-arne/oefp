"""Guards that same-canonical_id columns are numerically identical across sources.

Compares RAW per-source outputs because the DescriptorCalculator drops duplicate
columns at construction and they are not observable in merged output.

Asserts the full expected set of curated shared canonical identities to ensure
a dropped or renamed tag on either source fails the audit.
"""

from __future__ import annotations

# The 7 curated identities shared between Mordred and OpenEye per the spec
EXPECTED_SHARED_CANONICAL_IDS = frozenset({
    "quantity:exact_molecular_weight",
    "quantity:average_molecular_weight",
    "quantity:heavy_atom_count",
    "quantity:total_atom_count",
    "quantity:topological_psa",
    "quantity:num_hbond_donors_lipinski",
    "quantity:num_hbond_acceptors",
})

# The curated identities the RDKit source itself contributes to the audit.
# Guarded separately from EXPECTED_SHARED_CANONICAL_IDS because the global
# completeness set can be satisfied by another source pair (e.g. Mordred+OpenEye
# both carry heavy_atom_count), which would mask an RDKit-specific tag
# regression — precisely the dedup wiring this audit exists to protect. The
# source-aware guard below requires RDKit to both declare and match the id.
EXPECTED_RDKIT_SHARED = frozenset({
    "quantity:heavy_atom_count",
})


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
    mordred = oefp.DescriptorCalculator([oefp.MordredDescriptorSource()])
    openeye = oefp.DescriptorCalculator([oefp.OpenEyePropertyDescriptorSource()])
    rdkit = oefp.DescriptorCalculator([oefp.RDKitDescriptorSource()])
    calculators = [mordred, openeye, rdkit]
    rdkit_index = calculators.index(rdkit)

    # Source-aware guard: the RDKit source must DECLARE every id it is expected to
    # contribute. This catches a schema-level regression (RDKit silently loses a
    # canonical_id tag) that the global completeness set cannot, because another
    # source pair could still cover the same id.
    rdkit_schema_ids = {d.canonical_id for d in rdkit.schema.definitions if d.canonical_id}
    missing_rdkit_schema = EXPECTED_RDKIT_SHARED - rdkit_schema_ids
    assert not missing_rdkit_schema, (
        f"RDKit source no longer declares expected canonical id(s): {missing_rdkit_schema}"
    )

    shared_seen = set()
    rdkit_shared = set()  # ids where RDKit participated in a MATCHING pair
    for mol in panel_mols:
        per_source = [_canonical_values(calc, mol) for calc in calculators]
        # PAIRWISE comparison: for every unordered pair of sources, for every
        # canonical_id both carry (and whose value is not None), assert equality.
        # This avoids the intersection-across-ALL collapse when one source carries
        # only a subset of the curated ids (e.g., RDKit now carries only 1 of 7).
        # Accumulate every id observed as shared by ANY pair into shared_seen.
        for i in range(len(calculators)):
            for j in range(i + 1, len(calculators)):
                common = set(per_source[i]).intersection(per_source[j])
                shared_seen |= common
                if rdkit_index in (i, j):
                    rdkit_shared |= common
                for canonical_id in common:
                    v1, v2 = per_source[i][canonical_id], per_source[j][canonical_id]
                    assert v1 == v2, (
                        f"canonical_id {canonical_id} diverged between "
                        f"calculators {i} and {j}: {v1} != {v2}"
                    )
    missing = EXPECTED_SHARED_CANONICAL_IDS - shared_seen
    assert not missing, (
        f"curated canonical identities never observed as shared across the panel: {missing}"
    )
    # Source-aware value guard: the RDKit source must be OBSERVED matching another
    # source on every id it contributes. If RDKit's value stops matching (so it
    # never lands in a shared RDKit pair), this fires even though the global set
    # above stays satisfied by the Mordred+OpenEye pair.
    missing_rdkit = EXPECTED_RDKIT_SHARED - rdkit_shared
    assert not missing_rdkit, (
        f"RDKit source never matched another source on expected id(s): {missing_rdkit}"
    )
