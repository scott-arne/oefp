"""Tests for the public Python descriptor API."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _openeye_mol(smiles: str) -> Any:
    oechem = pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")

    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


def _string_spec() -> Any:
    import oefp

    return oefp.DescriptorSpec(
        value_type="string",
        source_name="unit-test",
        source_type="manual",
        source_version="1",
        parameters="kind=string",
    )


def test_descriptor_set_from_strings_aggregates_counts():
    import oefp

    descriptors = oefp.DescriptorSet.from_strings(
        ["beta", "alpha", "beta", "gamma", "alpha"],
        spec=_string_spec(),
    )

    assert descriptors.value_type == "string"
    assert descriptors.size == 3
    assert descriptors.total_count == 5
    assert descriptors.string_keys == ("alpha", "beta", "gamma")
    assert descriptors.integer_keys == ()
    assert descriptors.float_keys == ()
    assert descriptors.counts.tolist() == [2, 2, 1]
    assert descriptors.spec == _string_spec()

    with pytest.raises(TypeError, match="created by DescriptorSet"):
        oefp.DescriptorSet(descriptors._native)


def test_descriptor_set_supports_integer_and_float_keys_with_counts():
    import oefp

    integer_spec = oefp.DescriptorSpec(
        value_type="integer",
        source_name="unit-test",
        source_type="manual",
        source_version="1",
        parameters="kind=integer",
    )
    integers = oefp.DescriptorSet.from_integers(
        [2, 7, 11],
        counts=[4, 1, 3],
        spec=integer_spec,
    )
    assert integers.value_type == "integer"
    assert integers.integer_keys == (2, 7, 11)
    assert integers.counts.tolist() == [4, 1, 3]

    float_spec = oefp.DescriptorSpec(
        value_type="float",
        source_name="unit-test",
        source_type="manual",
        source_version="1",
        parameters="kind=float",
    )
    floats = oefp.DescriptorSet.from_floats(
        [1.5, 2.5],
        counts=[2, 5],
        spec=float_spec,
    )
    assert floats.value_type == "float"
    assert floats.float_keys == (1.5, 2.5)
    assert floats.counts.tolist() == [2, 5]


def test_descriptor_batch_from_descriptors_exposes_flattened_rows():
    import oefp

    spec = _string_spec()
    first = oefp.DescriptorSet.from_strings(["alpha", "beta", "alpha"], spec=spec)
    second = oefp.DescriptorSet.from_strings(["beta", "gamma"], spec=spec)

    batch = oefp.DescriptorBatch.from_descriptors([first, second])

    assert batch.value_type == "string"
    assert batch.size == 2
    assert batch.entry_count == 4
    assert batch.string_keys == ("alpha", "beta", "beta", "gamma")
    assert batch.counts.tolist() == [2, 1, 1, 1]
    assert batch.offsets.tolist() == [0, 2, 4]
    assert batch.spec == spec


def test_descriptor_compare_modes_and_batches():
    import oefp

    spec = _string_spec()
    left = oefp.DescriptorSet.from_strings(["a", "a", "b"], spec=spec)
    right = oefp.DescriptorSet.from_strings(["a", "b", "b"], spec=spec)
    exact_peer = oefp.DescriptorSet.from_strings(["a", "a", "a", "b"], spec=spec)
    only_a = oefp.DescriptorSet.from_strings(["a"], spec=spec)

    assert oefp.compare(left, right, oefp.Metric.tanimoto()) == pytest.approx(0.5)
    assert oefp.compare(left, right, oefp.Metric.tanimoto(), descriptor_mode="presence") == pytest.approx(1.0)
    assert oefp.compare(left, exact_peer, oefp.Metric.tanimoto(), descriptor_mode="exact_count") == pytest.approx(1 / 3)

    batch = oefp.DescriptorBatch.from_descriptors([left, right, only_a])
    np.testing.assert_allclose(
        oefp.compare(left, batch, oefp.Metric.tanimoto(), descriptor_mode="presence"),
        np.array([1.0, 1.0, 0.5]),
    )

    np.testing.assert_allclose(
        oefp.cdist(
            oefp.DescriptorBatch.from_descriptors([left, only_a]),
            oefp.DescriptorBatch.from_descriptors([right]),
            oefp.Metric.tanimoto(),
        ),
        np.array([[0.5], [1 / 3]]),
    )

    np.testing.assert_allclose(
        oefp.pdist(batch, oefp.Metric.tanimoto(), descriptor_mode="presence"),
        np.array([1.0, 0.5, 0.5]),
    )

    with pytest.raises(ValueError, match="Unknown descriptor comparison mode"):
        oefp.compare(left, right, oefp.Metric.tanimoto(), descriptor_mode="counts")


def test_atom_pair_descriptors_expose_raw_keys_for_aromatic_heterocycle():
    import oefp

    descriptors = oefp.atom_pair_descriptors(_openeye_mol("c1ccncc1"))

    assert descriptors.value_type == "string"
    assert descriptors.size == 6
    assert descriptors.total_count == 15
    assert descriptors.string_keys == (
        "42_1_42",
        "42_1_74",
        "42_2_42",
        "42_2_74",
        "42_3_42",
        "42_3_74",
    )
    assert descriptors.counts.tolist() == [4, 2, 4, 2, 2, 1]
    assert descriptors.spec.value_type == "string"
    assert descriptors.spec.source_name == "OEFP"
    assert descriptors.spec.source_type == "TopologicalAtomPair"
    assert descriptors.spec.source_version == "TopologicalAtomPair-1.1.0"
    assert descriptors.spec.parameters == (
        "min_distance=1;max_distance=30;use_chirality=false;output=descriptors"
    )
    assert oefp.TOPOLOGICAL_ATOM_PAIR_PREREQUISITES == (
        oefp.DESCRIPTOR_PREREQUISITE_GRAPH
    )
    assert not (
        oefp.TOPOLOGICAL_ATOM_PAIR_PREREQUISITES
        & oefp.DESCRIPTOR_PREREQUISITE_COORDINATES_2D
    )
    assert not (
        oefp.TOPOLOGICAL_ATOM_PAIR_PREREQUISITES
        & oefp.DESCRIPTOR_PREREQUISITE_COORDINATES_3D
    )
    assert (
        oefp.DISTANCE_ATOM_PAIR_PREREQUISITES
        & oefp.DESCRIPTOR_PREREQUISITE_COORDINATES_3D
    )

    distance_one = oefp.atom_pair_descriptors(
        _openeye_mol("c1ccncc1"),
        max_distance=1,
    )
    assert distance_one.total_count == 6
    assert distance_one.string_keys == ("42_1_42", "42_1_74")
    assert distance_one.counts.tolist() == [4, 2]


def test_topological_atom_pair_descriptors_alias_atom_pair_descriptors():
    import oefp

    mol = _openeye_mol("c1ccncc1")

    topological = oefp.topological_atom_pair_descriptors(mol)
    compatible = oefp.atom_pair_descriptors(mol)

    assert topological.string_keys == compatible.string_keys
    assert topological.counts.tolist() == compatible.counts.tolist()
    assert topological.spec.source_type == "TopologicalAtomPair"


def test_distance_atom_pair_descriptors_require_existing_3d_coordinates():
    import pytest
    import oefp

    with pytest.raises(ValueError, match="requires existing 3D coordinates"):
        oefp.distance_atom_pair_descriptors(_openeye_mol("CCO"))


def test_morgan_descriptors_expose_raw_integer_keys_for_complex_molecules():
    import oefp

    descriptors = oefp.morgan_descriptors(_openeye_mol("c1ccncc1"), radius=1)

    assert descriptors.value_type == "integer"
    assert descriptors.size == 5
    assert descriptors.total_count == 12
    assert descriptors.integer_keys == (
        98513984,
        2041434490,
        3118255683,
        3218693969,
        3776905034,
    )
    assert descriptors.counts.tolist() == [3, 1, 1, 5, 2]
    assert descriptors.spec.value_type == "integer"
    assert descriptors.spec.source_name == "OEFP"
    assert descriptors.spec.source_type == "Morgan"
    assert descriptors.spec.source_version == "Morgan-2026.03.1"
    assert descriptors.spec.parameters == (
        "radius=1;use_chirality=false;use_bond_types=true;"
        "only_nonzero_invariants=false;include_ring_membership=true;"
        "include_redundant_environments=false;output=descriptors"
    )


def test_morgan_descriptors_compare_as_descriptor_sets():
    import oefp

    query = oefp.morgan_descriptors(_openeye_mol("CCO"), radius=1)
    library = [
        query,
        oefp.morgan_descriptors(_openeye_mol("CC=O"), radius=1),
        oefp.morgan_descriptors(_openeye_mol("CC(C)(C)Cl"), radius=1),
    ]
    batch = oefp.DescriptorBatch.from_descriptors(library)

    np.testing.assert_allclose(
        oefp.compare(query, batch, oefp.Metric.tanimoto()),
        np.array([1.0, 1 / 11, 1 / 15]),
    )

    with pytest.raises(ValueError, match="count simulation is only supported"):
        oefp.morgan_descriptors(_openeye_mol("CCO"), count_simulation=True)
