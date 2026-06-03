"""RDKit conformance tests for OEFP Morgan count fingerprints."""

from __future__ import annotations

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem", reason="RDKit is required for Morgan count conformance tests")
rdFingerprintGenerator = pytest.importorskip(
    "rdkit.Chem.rdFingerprintGenerator",
    reason="RDKit is required for Morgan count conformance tests",
)
pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")


def _openeye_mol(smiles: str):
    from openeye import oechem

    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


def _rdkit_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return mol


def _oefp_counts(fp) -> dict[int, int]:
    indices = np.asarray(fp.indices, dtype=np.uint32)
    counts = np.asarray(fp.counts, dtype=np.uint32)
    assert indices.tolist() == sorted(indices.tolist())
    assert len(indices) == len(counts)
    return {int(index): int(count) for index, count in zip(indices, counts, strict=True)}


def _rdkit_counts(
    smiles: str,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> dict[int, int]:
    atom_invariants_generator = rdFingerprintGenerator.GetMorganAtomInvGen(include_ring_membership)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        countSimulation=False,
        includeChirality=use_chirality,
        useBondTypes=use_bond_types,
        onlyNonzeroInvariants=only_nonzero_invariants,
        includeRingMembership=include_ring_membership,
        atomInvariantsGenerator=atom_invariants_generator,
        fpSize=num_bits,
        includeRedundantEnvironments=include_redundant_environments,
    )
    return {int(index): int(count) for index, count in generator.GetCountFingerprint(_rdkit_mol(smiles)).GetNonzeroElements().items()}


def _rdkit_sparse_counts(
    smiles: str,
    *,
    radius: int = 2,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> dict[int, int]:
    atom_invariants_generator = rdFingerprintGenerator.GetMorganAtomInvGen(include_ring_membership)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        countSimulation=False,
        includeChirality=use_chirality,
        useBondTypes=use_bond_types,
        onlyNonzeroInvariants=only_nonzero_invariants,
        includeRingMembership=include_ring_membership,
        atomInvariantsGenerator=atom_invariants_generator,
        includeRedundantEnvironments=include_redundant_environments,
    )
    return {int(index): int(count) for index, count in generator.GetSparseCountFingerprint(_rdkit_mol(smiles)).GetNonzeroElements().items()}


def _rdkit_count_bit_info(
    smiles: str,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    sparse: bool = False,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> dict[int, tuple[tuple[int, int], ...]]:
    atom_invariants_generator = rdFingerprintGenerator.GetMorganAtomInvGen(include_ring_membership)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        countSimulation=False,
        includeChirality=use_chirality,
        useBondTypes=use_bond_types,
        onlyNonzeroInvariants=False,
        includeRingMembership=include_ring_membership,
        atomInvariantsGenerator=atom_invariants_generator,
        fpSize=num_bits,
        includeRedundantEnvironments=include_redundant_environments,
    )
    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    if sparse:
        generator.GetSparseCountFingerprint(_rdkit_mol(smiles), additionalOutput=output)
    else:
        generator.GetCountFingerprint(_rdkit_mol(smiles), additionalOutput=output)
    return {
        int(bit_id) & 0xFFFFFFFF: tuple((int(atom_id), int(radius)) for atom_id, radius in environments)
        for bit_id, environments in output.GetBitInfoMap().items()
    }


def _rdkit_count_fingerprint(
    smiles: str,
    *,
    radius: int = 2,
    num_bits: int = 2048,
):
    atom_invariants_generator = rdFingerprintGenerator.GetMorganAtomInvGen(True)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        countSimulation=False,
        includeChirality=False,
        useBondTypes=True,
        onlyNonzeroInvariants=False,
        includeRingMembership=True,
        atomInvariantsGenerator=atom_invariants_generator,
        fpSize=num_bits,
        includeRedundantEnvironments=False,
    )
    return generator.GetCountFingerprint(_rdkit_mol(smiles))


def _descriptor_integer_counts(descriptors) -> dict[int, int]:
    return {
        int(key): int(count)
        for key, count in zip(descriptors.integer_keys, descriptors.counts, strict=True)
    }


_ALIGNED_CHIRALITY_SMILES = [
    "F[C@](Cl)(Br)I",
    "F[C@@](Cl)(Br)I",
    "C/C=C/C",
    "C/C=C\\C",
]


def _count_stats(a: dict[int, int], b: dict[int, int]) -> tuple[int, int, int, int, float, int]:
    keys = set(a) | set(b)
    l1 = sum(abs(a.get(key, 0) - b.get(key, 0)) for key in keys)
    squared_l2 = sum((a.get(key, 0) - b.get(key, 0)) ** 2 for key in keys)
    chebyshev = max((abs(a.get(key, 0) - b.get(key, 0)) for key in keys), default=0)
    unequal = sum(1 for key in keys if a.get(key, 0) != b.get(key, 0))
    canberra = sum(
        abs(a.get(key, 0) - b.get(key, 0)) / (abs(a.get(key, 0)) + abs(b.get(key, 0)))
        for key in keys
        if abs(a.get(key, 0)) + abs(b.get(key, 0)) != 0
    )
    sum_abs = sum(abs(count) for count in a.values()) + sum(abs(count) for count in b.values())
    return l1, squared_l2, chebyshev, unequal, canberra, sum_abs


@pytest.mark.parametrize(
    "smiles",
    [
        "C",
        "CC",
        "CCO",
        "c1ccccc1",
        "C1CCCCC1",
        "C[NH+](C)C",
        "[2H]O",
    ],
)
@pytest.mark.parametrize("radius", [0, 1, 2])
@pytest.mark.parametrize("num_bits", [128, 2048])
def test_morgan_counts_match_rdkit_default_options(smiles: str, radius: int, num_bits: int):
    import oefp

    fp = oefp.morgan_count_fingerprint(_openeye_mol(smiles), radius=radius, num_bits=num_bits)

    assert fp.num_bits == num_bits
    assert _oefp_counts(fp) == _rdkit_counts(smiles, radius=radius, num_bits=num_bits)


@pytest.mark.parametrize(
    "smiles",
    [
        "C",
        "CC",
        "CCO",
        "c1ccccc1",
        "C1CCCCC1",
        "C[NH+](C)C",
        "[2H]O",
    ],
)
@pytest.mark.parametrize("radius", [0, 1, 2])
def test_morgan_sparse_counts_match_rdkit_default_options(smiles: str, radius: int):
    import oefp

    fp = oefp.morgan_sparse_count_fingerprint(_openeye_mol(smiles), radius=radius)

    assert fp.num_bits == 2**64 - 1
    assert _oefp_counts(fp) == _rdkit_sparse_counts(smiles, radius=radius)


@pytest.mark.parametrize(
    ("kwargs", "smiles"),
    [
        ({"use_bond_types": False}, "c1ccccc1"),
        ({"include_ring_membership": False}, "C1CCCCC1"),
        ({"include_redundant_environments": True}, "CC"),
    ],
)
def test_morgan_count_option_toggles_match_rdkit(kwargs: dict[str, bool], smiles: str):
    import oefp

    default_counts = _rdkit_counts(smiles, radius=2, num_bits=256)
    expected_counts = _rdkit_counts(smiles, radius=2, num_bits=256, **kwargs)
    assert expected_counts != default_counts

    fp = oefp.morgan_count_fingerprint(_openeye_mol(smiles), radius=2, num_bits=256, **kwargs)

    assert _oefp_counts(fp) == expected_counts


@pytest.mark.parametrize(
    ("kwargs", "smiles"),
    [
        ({"use_bond_types": False}, "c1ccccc1"),
        ({"include_ring_membership": False}, "C1CCCCC1"),
        ({"include_redundant_environments": True}, "CC"),
    ],
)
def test_morgan_sparse_count_option_toggles_match_rdkit(kwargs: dict[str, bool], smiles: str):
    import oefp

    default_counts = _rdkit_sparse_counts(smiles, radius=2)
    expected_counts = _rdkit_sparse_counts(smiles, radius=2, **kwargs)
    assert expected_counts != default_counts

    fp = oefp.morgan_sparse_count_fingerprint(_openeye_mol(smiles), radius=2, **kwargs)

    assert _oefp_counts(fp) == expected_counts


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("num_bits", [256, 1024])
def test_morgan_counts_match_rdkit_with_chirality(smiles: str, radius: int, num_bits: int):
    import oefp

    fp = oefp.morgan_count_fingerprint(
        _openeye_mol(smiles),
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )

    assert fp.num_bits == num_bits
    assert _oefp_counts(fp) == _rdkit_counts(
        smiles,
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("num_bits", [256, 1024])
def test_morgan_count_mapping_bit_info_matches_rdkit_with_chirality(
    smiles: str,
    radius: int,
    num_bits: int,
):
    import oefp

    result = oefp.morgan_count_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )
    expected = _rdkit_count_bit_info(
        smiles,
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )

    assert _oefp_counts(result.fingerprint) == _rdkit_counts(
        smiles,
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )
    assert result.mapping.bit_info() == expected


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
def test_morgan_sparse_counts_match_rdkit_with_chirality(smiles: str, radius: int):
    import oefp

    fp = oefp.morgan_sparse_count_fingerprint(
        _openeye_mol(smiles),
        radius=radius,
        use_chirality=True,
    )

    assert fp.num_bits == 2**64 - 1
    assert _oefp_counts(fp) == _rdkit_sparse_counts(
        smiles,
        radius=radius,
        use_chirality=True,
    )


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
def test_morgan_sparse_count_mapping_bit_info_matches_rdkit_with_chirality(
    smiles: str,
    radius: int,
):
    import oefp

    result = oefp.morgan_sparse_count_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=radius,
        use_chirality=True,
    )
    expected = _rdkit_count_bit_info(
        smiles,
        radius=radius,
        sparse=True,
        use_chirality=True,
    )

    assert _oefp_counts(result.fingerprint) == _rdkit_sparse_counts(
        smiles,
        radius=radius,
        use_chirality=True,
    )
    assert result.mapping.bit_info() == expected


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
def test_morgan_descriptor_integer_counts_match_rdkit_with_chirality(
    smiles: str,
    radius: int,
):
    import oefp

    descriptors = oefp.morgan_descriptors(
        _openeye_mol(smiles),
        radius=radius,
        use_chirality=True,
    )

    assert descriptors.value_type == "integer"
    assert _descriptor_integer_counts(descriptors) == _rdkit_sparse_counts(
        smiles,
        radius=radius,
        use_chirality=True,
    )


def test_morgan_count_mapping_bit_info_matches_rdkit():
    import oefp

    smiles = "CCC(CC)CO"
    result = oefp.morgan_count_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=1,
        num_bits=2048,
    )
    expected = _rdkit_count_bit_info(smiles, radius=1, num_bits=2048)

    assert _oefp_counts(result.fingerprint) == _rdkit_counts(smiles, radius=1, num_bits=2048)
    assert result.mapping.bit_info() == expected


def test_morgan_sparse_count_mapping_bit_info_matches_rdkit():
    import oefp

    smiles = "CCC(CC)CO"
    result = oefp.morgan_sparse_count_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=1,
    )
    expected = _rdkit_count_bit_info(smiles, radius=1, sparse=True)

    assert _oefp_counts(result.fingerprint) == _rdkit_sparse_counts(smiles, radius=1)
    assert result.mapping.bit_info() == expected


def test_morgan_count_arrays_are_read_only_views():
    import oefp

    fp = oefp.morgan_count_fingerprint(_openeye_mol("CCO"), num_bits=128)

    assert fp.indices.flags.writeable is False
    assert fp.counts.flags.writeable is False
    assert fp.nonzero_count == len(fp.indices)
    assert fp.total_count == int(np.asarray(fp.counts, dtype=np.uint32).sum())


def test_morgan_count_reuses_public_option_validation():
    import oefp

    mol = _openeye_mol("CCO")

    with pytest.raises(ValueError, match="num_bits"):
        oefp.morgan_count_fingerprint(mol, num_bits=0)
    with pytest.raises(ValueError, match="radius"):
        oefp.morgan_count_fingerprint(mol, radius=-1)
    fp = oefp.morgan_count_fingerprint(mol, use_chirality=True)
    assert fp.num_bits == 2048

    with pytest.raises(ValueError, match="radius"):
        oefp.morgan_sparse_count_fingerprint(mol, radius=-1)
    sparse_fp = oefp.morgan_sparse_count_fingerprint(mol, use_chirality=True)
    assert sparse_fp.num_bits == 2**64 - 1


def test_morgan_count_compare_uses_nonzero_boolean_metrics():
    import oefp

    kwargs = {"radius": 2, "num_bits": 256}
    fp_a = oefp.morgan_count_fingerprint(_openeye_mol("CCO"), **kwargs)
    fp_b = oefp.morgan_count_fingerprint(_openeye_mol("CCN"), **kwargs)
    indices_a = set(_oefp_counts(fp_a))
    indices_b = set(_oefp_counts(fp_b))

    intersection = len(indices_a & indices_b)
    only_a = len(indices_a - indices_b)
    only_b = len(indices_b - indices_a)
    nonzero = intersection + only_a + only_b
    unequal = only_a + only_b

    assert oefp.compare(fp_a, fp_b, oefp.Metric.tanimoto()) == pytest.approx(intersection / nonzero)
    assert oefp.compare(fp_a, fp_b, oefp.Metric.jaccard()) == pytest.approx(unequal / nonzero)
    assert oefp.compare(fp_a, fp_b, oefp.Metric.dice()) == pytest.approx(unequal / (intersection + nonzero))
    assert oefp.compare(fp_a, fp_b, oefp.Metric.tversky(0.25, 0.75)) == pytest.approx(
        intersection / (intersection + 0.25 * only_a + 0.75 * only_b),
    )


def test_morgan_count_compare_supports_real_and_integer_distances():
    import oefp

    fp_a = oefp.morgan_count_fingerprint(_openeye_mol("CCO"), num_bits=128)
    fp_b = oefp.morgan_count_fingerprint(_openeye_mol("CCN"), num_bits=128)
    counts_a = _oefp_counts(fp_a)
    counts_b = _oefp_counts(fp_b)
    l1, squared_l2, chebyshev, unequal, canberra, sum_abs = _count_stats(counts_a, counts_b)

    assert oefp.compare(fp_a, fp_b, oefp.Metric.euclidean()) == pytest.approx(np.sqrt(squared_l2))
    assert oefp.compare(fp_a, fp_b, oefp.Metric.manhattan()) == pytest.approx(l1)
    assert oefp.compare(fp_a, fp_b, oefp.Metric.chebyshev()) == pytest.approx(chebyshev)
    assert oefp.compare(fp_a, fp_b, oefp.Metric.hamming()) == pytest.approx(unequal / fp_a.num_bits)
    assert oefp.compare(fp_a, fp_b, oefp.Metric.canberra()) == pytest.approx(canberra)
    assert oefp.compare(fp_a, fp_b, oefp.Metric.bray_curtis()) == pytest.approx(l1 / sum_abs)
