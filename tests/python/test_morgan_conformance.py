"""RDKit conformance tests for OEFP Morgan binary fingerprints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem", reason="RDKit is required for Morgan conformance tests")
rdFingerprintGenerator = pytest.importorskip(
    "rdkit.Chem.rdFingerprintGenerator",
    reason="RDKit is required for Morgan conformance tests",
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


def _oefp_on_bits(fp) -> set[int]:
    words = np.asarray(fp.words, dtype=np.uint64)
    bits: set[int] = set()
    for word_index, word in enumerate(words):
        value = int(word)
        while value:
            low_bit = value & -value
            bits.add(word_index * 64 + (low_bit.bit_length() - 1))
            value ^= low_bit
    padding_bits = {bit for bit in bits if bit >= fp.num_bits}
    assert not padding_bits
    return bits


def _oefp_sparse_on_bits(fp) -> set[int]:
    indices = np.asarray(fp.indices, dtype=np.uint32)
    assert indices.tolist() == sorted(indices.tolist())
    assert len(indices) == fp.popcount
    return {int(index) for index in indices}


def _rdkit_on_bits(
    smiles: str,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
    count_simulation: bool = False,
    count_bounds: Sequence[int] | None = None,
) -> set[int]:
    atom_invariants_generator = rdFingerprintGenerator.GetMorganAtomInvGen(include_ring_membership)
    rdkit_count_bounds = tuple(count_bounds) if count_bounds is not None else None
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        countSimulation=count_simulation,
        includeChirality=use_chirality,
        useBondTypes=use_bond_types,
        onlyNonzeroInvariants=only_nonzero_invariants,
        includeRingMembership=include_ring_membership,
        countBounds=rdkit_count_bounds,
        atomInvariantsGenerator=atom_invariants_generator,
        fpSize=num_bits,
        includeRedundantEnvironments=include_redundant_environments,
    )
    return set(generator.GetFingerprint(_rdkit_mol(smiles)).GetOnBits())


def _rdkit_sparse_on_bits(
    smiles: str,
    *,
    radius: int = 2,
    use_chirality: bool = False,
    use_bond_types: bool = True,
    only_nonzero_invariants: bool = False,
    include_ring_membership: bool = True,
    include_redundant_environments: bool = False,
) -> set[int]:
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
    return {int(bit) & 0xFFFFFFFF for bit in generator.GetSparseFingerprint(_rdkit_mol(smiles)).GetOnBits()}


def _rdkit_bit_info(
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
        generator.GetSparseFingerprint(_rdkit_mol(smiles), additionalOutput=output)
    else:
        generator.GetFingerprint(_rdkit_mol(smiles), additionalOutput=output)
    return {
        int(bit_id) & 0xFFFFFFFF: tuple((int(atom_id), int(radius)) for atom_id, radius in environments)
        for bit_id, environments in output.GetBitInfoMap().items()
    }


def _rdkit_count_simulation_bit_info(
    smiles: str,
    *,
    radius: int = 2,
    num_bits: int = 2048,
    count_bounds: Sequence[int] | None = None,
) -> dict[int, tuple[tuple[int, int], ...]]:
    atom_invariants_generator = rdFingerprintGenerator.GetMorganAtomInvGen(True)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        countSimulation=True,
        countBounds=tuple(count_bounds) if count_bounds is not None else None,
        includeChirality=False,
        useBondTypes=True,
        onlyNonzeroInvariants=False,
        includeRingMembership=True,
        atomInvariantsGenerator=atom_invariants_generator,
        fpSize=num_bits,
        includeRedundantEnvironments=False,
    )
    output = rdFingerprintGenerator.AdditionalOutput()
    output.AllocateBitInfoMap()
    generator.GetFingerprint(_rdkit_mol(smiles), additionalOutput=output)
    return {
        int(bit_id) & 0xFFFFFFFF: tuple((int(atom_id), int(radius)) for atom_id, radius in environments)
        for bit_id, environments in output.GetBitInfoMap().items()
    }


_ALIGNED_CHIRALITY_SMILES = [
    "F[C@](Cl)(Br)I",
    "F[C@@](Cl)(Br)I",
    "C/C=C/C",
    "C/C=C\\C",
]


@pytest.mark.parametrize(
    "smiles",
    [
        "C",
        "CC",
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "c1ccc(O)cc1",
        "C1CCCCC1",
        "CC(=O)O",
        "C[NH+](C)C",
        "O=C([O-])C",
        "[2H]O",
        "[13CH3]CO",
        "OCl(=O)(=O)=O",
        "OBr(=O)(=O)=O",
        "OI(=O)(=O)=O",
    ],
)
@pytest.mark.parametrize("radius", [0, 1, 2])
@pytest.mark.parametrize("num_bits", [127, 128, 2048])
def test_morgan_binary_matches_rdkit_default_options(smiles: str, radius: int, num_bits: int):
    import oefp

    fp = oefp.morgan_fingerprint(_openeye_mol(smiles), radius=radius, num_bits=num_bits)

    assert fp.num_bits == num_bits
    assert _oefp_on_bits(fp) == _rdkit_on_bits(smiles, radius=radius, num_bits=num_bits)


@pytest.mark.parametrize(
    "smiles",
    [
        "C",
        "CC",
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "C1CCCCC1",
        "C[NH+](C)C",
        "[2H]O",
        "OCl(=O)(=O)=O",
        "OBr(=O)(=O)=O",
        "OI(=O)(=O)=O",
    ],
)
@pytest.mark.parametrize("radius", [0, 1, 2])
def test_morgan_sparse_binary_matches_rdkit_default_options(smiles: str, radius: int):
    import oefp

    fp = oefp.morgan_sparse_fingerprint(_openeye_mol(smiles), radius=radius)

    assert fp.num_bits == 2**64 - 1
    assert _oefp_sparse_on_bits(fp) == _rdkit_sparse_on_bits(smiles, radius=radius)


@pytest.mark.parametrize(
    ("kwargs", "smiles"),
    [
        ({"use_bond_types": False}, "c1ccccc1"),
        ({"include_ring_membership": False}, "C1CCCCC1"),
        ({"include_redundant_environments": True}, "CC"),
    ],
)
def test_morgan_binary_option_toggles_match_rdkit(kwargs: dict[str, bool], smiles: str):
    import oefp

    default_bits = _rdkit_on_bits(smiles, radius=2, num_bits=256)
    expected_bits = _rdkit_on_bits(smiles, radius=2, num_bits=256, **cast(Any, kwargs))
    assert expected_bits != default_bits

    fp = oefp.morgan_fingerprint(
        _openeye_mol(smiles),
        radius=2,
        num_bits=256,
        **cast(Any, kwargs),
    )

    assert _oefp_on_bits(fp) == expected_bits


@pytest.mark.parametrize(
    ("kwargs", "smiles"),
    [
        ({"use_bond_types": False}, "c1ccccc1"),
        ({"include_ring_membership": False}, "C1CCCCC1"),
        ({"include_redundant_environments": True}, "CC"),
    ],
)
def test_morgan_sparse_binary_option_toggles_match_rdkit(kwargs: dict[str, bool], smiles: str):
    import oefp

    default_bits = _rdkit_sparse_on_bits(smiles, radius=2)
    expected_bits = _rdkit_sparse_on_bits(smiles, radius=2, **cast(Any, kwargs))
    assert expected_bits != default_bits

    fp = oefp.morgan_sparse_fingerprint(
        _openeye_mol(smiles),
        radius=2,
        **cast(Any, kwargs),
    )

    assert _oefp_sparse_on_bits(fp) == expected_bits


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("num_bits", [256, 1024])
def test_morgan_binary_matches_rdkit_with_chirality(smiles: str, radius: int, num_bits: int):
    import oefp

    fp = oefp.morgan_fingerprint(
        _openeye_mol(smiles),
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )

    assert fp.num_bits == num_bits
    assert _oefp_on_bits(fp) == _rdkit_on_bits(
        smiles,
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
def test_morgan_sparse_binary_matches_rdkit_with_chirality(smiles: str, radius: int):
    import oefp

    fp = oefp.morgan_sparse_fingerprint(
        _openeye_mol(smiles),
        radius=radius,
        use_chirality=True,
    )

    assert fp.num_bits == 2**64 - 1
    assert _oefp_sparse_on_bits(fp) == _rdkit_sparse_on_bits(
        smiles,
        radius=radius,
        use_chirality=True,
    )


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
@pytest.mark.parametrize("num_bits", [256, 1024])
def test_morgan_binary_mapping_bit_info_matches_rdkit_with_chirality(
    smiles: str,
    radius: int,
    num_bits: int,
):
    import oefp

    result = oefp.morgan_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )
    expected = _rdkit_bit_info(
        smiles,
        radius=radius,
        num_bits=num_bits,
        use_chirality=True,
    )

    assert _oefp_on_bits(result.fingerprint) == set(expected)
    assert result.mapping.bit_info() == expected


@pytest.mark.parametrize("smiles", _ALIGNED_CHIRALITY_SMILES)
@pytest.mark.parametrize("radius", [1, 2])
def test_morgan_sparse_binary_mapping_bit_info_matches_rdkit_with_chirality(
    smiles: str,
    radius: int,
):
    import oefp

    result = oefp.morgan_sparse_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=radius,
        use_chirality=True,
    )
    expected = _rdkit_bit_info(
        smiles,
        radius=radius,
        sparse=True,
        use_chirality=True,
    )

    assert _oefp_sparse_on_bits(result.fingerprint) == set(expected)
    assert result.mapping.bit_info() == expected


def test_morgan_default_nonchiral_keeps_openeye_graph_boundary():
    import oefp

    explicit_chiral = "C[C@H](F)Cl"
    implicit_achiral = "CC(F)Cl"
    chiral_fp = oefp.morgan_fingerprint(_openeye_mol(explicit_chiral), radius=2, num_bits=1024)
    achiral_fp = oefp.morgan_fingerprint(_openeye_mol(implicit_achiral), radius=2, num_bits=1024)

    assert _rdkit_on_bits(explicit_chiral, radius=2, num_bits=1024) == _rdkit_on_bits(
        implicit_achiral,
        radius=2,
        num_bits=1024,
    )
    assert _oefp_on_bits(chiral_fp) != _oefp_on_bits(achiral_fp)


@pytest.mark.parametrize("smiles", ["FC(Cl)(Br)I", "CC=CC"])
@pytest.mark.parametrize("radius", [1, 2])
def test_morgan_with_chirality_ignores_unspecified_stereo_candidates(
    smiles: str,
    radius: int,
):
    import oefp

    fp = oefp.morgan_fingerprint(
        _openeye_mol(smiles),
        radius=radius,
        num_bits=1024,
        use_chirality=True,
    )

    assert _oefp_on_bits(fp) == _rdkit_on_bits(
        smiles,
        radius=radius,
        num_bits=1024,
        use_chirality=True,
    )


def test_morgan_binary_mapping_bit_info_matches_rdkit():
    import oefp

    smiles = "CCC(CC)CO"
    result = oefp.morgan_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=1,
        num_bits=2048,
    )
    expected = _rdkit_bit_info(smiles, radius=1, num_bits=2048)

    assert _oefp_on_bits(result.fingerprint) == set(expected)
    assert result.mapping.bit_info() == expected
    assert result.mapping.environments_for_bit(80) == expected[80]


def test_morgan_sparse_binary_mapping_bit_info_matches_rdkit():
    import oefp

    smiles = "CCC(CC)CO"
    result = oefp.morgan_sparse_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=1,
    )
    expected = _rdkit_bit_info(smiles, radius=1, sparse=True)

    assert _oefp_sparse_on_bits(result.fingerprint) == set(expected)
    assert result.mapping.bit_info() == expected
    assert result.mapping.environments_for_bit(2245384272) == expected[2245384272]


@pytest.mark.parametrize("smiles", ["CC", "CCO", "CC(C)O", "c1ccccc1", "C1CCCCC1"])
@pytest.mark.parametrize("radius", [0, 1, 2])
@pytest.mark.parametrize("num_bits", [128, 2048])
def test_morgan_binary_count_simulation_matches_rdkit_default_bounds(
    smiles: str,
    radius: int,
    num_bits: int,
):
    import oefp

    fp = oefp.morgan_fingerprint(
        _openeye_mol(smiles),
        radius=radius,
        num_bits=num_bits,
        count_simulation=True,
    )

    assert fp.num_bits == num_bits
    assert _oefp_on_bits(fp) == _rdkit_on_bits(
        smiles,
        radius=radius,
        num_bits=num_bits,
        count_simulation=True,
    )


@pytest.mark.parametrize(
    ("count_bounds", "smiles"),
    [
        ([1, 3, 7], "CCO"),
        ([0, 1, 2], "CC"),
        ([1, 8, 16, 32], "c1ccccc1"),
    ],
)
def test_morgan_binary_count_simulation_custom_bounds_match_rdkit(
    count_bounds: Sequence[int],
    smiles: str,
):
    import oefp

    fp = oefp.morgan_fingerprint(
        _openeye_mol(smiles),
        radius=2,
        num_bits=256,
        count_simulation=True,
        count_bounds=count_bounds,
    )

    assert _oefp_on_bits(fp) == _rdkit_on_bits(
        smiles,
        radius=2,
        num_bits=256,
        count_simulation=True,
        count_bounds=count_bounds,
    )


@pytest.mark.parametrize(
    ("smiles", "radius", "num_bits", "count_bounds"),
    [
        ("CC", 1, 64, [1, 2, 4]),
        ("CCC(CC)CO", 1, 64, [1, 2, 4]),
        ("CCO", 2, 128, None),
        ("CCO", 2, 256, [1, 3, 7]),
        ("CC", 2, 256, [0, 1, 2]),
        ("c1ccccc1", 1, 64, [1, 2, 4]),
    ],
)
def test_morgan_binary_count_simulation_mapping_bit_info_matches_rdkit(
    smiles: str,
    radius: int,
    num_bits: int,
    count_bounds: Sequence[int] | None,
):
    import oefp

    result = oefp.morgan_fingerprint_with_mapping(
        _openeye_mol(smiles),
        radius=radius,
        num_bits=num_bits,
        count_simulation=True,
        count_bounds=count_bounds,
    )
    expected_bits = _rdkit_on_bits(
        smiles,
        radius=radius,
        num_bits=num_bits,
        count_simulation=True,
        count_bounds=count_bounds,
    )
    expected_mapping = _rdkit_count_simulation_bit_info(
        smiles,
        radius=radius,
        num_bits=num_bits,
        count_bounds=count_bounds,
    )

    assert _oefp_on_bits(result.fingerprint) == expected_bits
    assert result.mapping.bit_info() == expected_mapping


# morgan_fingerprint is a Python convenience wrapper, so user-facing option
# validation should raise ValueError before native SWIG exception translation.
def test_morgan_rejects_zero_num_bits():
    import oefp

    with pytest.raises(ValueError, match="num_bits"):
        oefp.morgan_fingerprint(_openeye_mol("CCO"), num_bits=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"radius": -1}, "radius"),
        ({"radius": 2**32}, "radius"),
        ({"num_bits": -128}, "num_bits"),
        ({"num_bits": 2**32}, "num_bits"),
    ],
)
def test_morgan_rejects_out_of_range_integer_options(kwargs: dict[str, int], match: str):
    import oefp

    with pytest.raises(ValueError, match=match):
        oefp.morgan_fingerprint(_openeye_mol("CCO"), **cast(Any, kwargs))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"radius": 1.5}, "radius"),
        ({"num_bits": 128.5}, "num_bits"),
    ],
)
def test_morgan_rejects_non_integral_options(kwargs: dict[str, float], match: str):
    import oefp

    with pytest.raises(TypeError, match=match):
        oefp.morgan_fingerprint(_openeye_mol("CCO"), **cast(Any, kwargs))


@pytest.mark.parametrize(
    ("kwargs", "exception_type", "match"),
    [
        ({"count_simulation": True, "count_bounds": []}, ValueError, "count_bounds"),
        (
            {"count_simulation": True, "num_bits": 4, "count_bounds": [1, 2, 4, 8]},
            ValueError,
            "count_bounds",
        ),
        ({"count_bounds": [-1]}, ValueError, "count_bounds"),
        ({"count_bounds": [2**32]}, ValueError, "count_bounds"),
        ({"count_bounds": [1.5]}, TypeError, "count_bounds"),
    ],
)
def test_morgan_rejects_invalid_count_simulation_options(
    kwargs: dict[str, Any],
    exception_type: type[Exception],
    match: str,
):
    import oefp

    with pytest.raises(exception_type, match=match):
        oefp.morgan_fingerprint(_openeye_mol("CCO"), **kwargs)


def test_morgan_accepts_chirality_option():
    import oefp

    mol = _openeye_mol("F[C@](Cl)(Br)I")

    fp = oefp.morgan_fingerprint(mol, use_chirality=True)
    sparse_fp = oefp.morgan_sparse_fingerprint(mol, use_chirality=True)

    assert fp.num_bits == 2048
    assert sparse_fp.num_bits == 2**64 - 1


def test_morgan_batch_compatibility_uses_full_options():
    import oefp

    fp_a = oefp.morgan_fingerprint(_openeye_mol("CCO"), radius=2, num_bits=128)
    fp_b = oefp.morgan_fingerprint(_openeye_mol("CCN"), radius=2, num_bits=128)

    batch = oefp.OEFPBatch.from_fingerprints([fp_a, fp_b])
    assert batch.size == 2

    mismatched_fps = [
        oefp.morgan_fingerprint(_openeye_mol("CCO"), radius=1, num_bits=128),
        oefp.morgan_fingerprint(_openeye_mol("CCO"), radius=2, num_bits=128, use_bond_types=False),
        # Default Morgan atom invariants are nonzero, so this option is a
        # metadata/spec compatibility check until custom invariants exist.
        oefp.morgan_fingerprint(_openeye_mol("CCO"), radius=2, num_bits=128, only_nonzero_invariants=True),
        oefp.morgan_fingerprint(_openeye_mol("CCO"), radius=2, num_bits=128, include_ring_membership=False),
        oefp.morgan_fingerprint(_openeye_mol("CCO"), radius=2, num_bits=128, include_redundant_environments=True),
        oefp.morgan_fingerprint(_openeye_mol("CCO"), radius=2, num_bits=128, count_simulation=True),
        oefp.morgan_fingerprint(_openeye_mol("CCO"), radius=2, num_bits=128, count_bounds=[1, 3, 7]),
    ]

    for mismatched_fp in mismatched_fps:
        with pytest.raises(RuntimeError, match="spec"):
            oefp.OEFPBatch.from_fingerprints([fp_a, mismatched_fp])
