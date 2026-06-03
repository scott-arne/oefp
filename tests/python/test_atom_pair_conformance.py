"""RDKit conformance tests for OEFP Atom Pair binary fingerprints."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem", reason="RDKit is required for Atom Pair conformance tests")
rdFingerprintGenerator = pytest.importorskip(
    "rdkit.Chem.rdFingerprintGenerator",
    reason="RDKit is required for Atom Pair conformance tests",
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


def _oefp_counts(fp) -> dict[int, int]:
    indices = np.asarray(fp.indices, dtype=np.uint32)
    counts = np.asarray(fp.counts, dtype=np.uint32)
    assert indices.tolist() == sorted(indices.tolist())
    assert len(indices) == len(counts)
    return {int(index): int(count) for index, count in zip(indices, counts, strict=True)}


def _oefp_sparse_bits(fp) -> set[int]:
    indices = np.asarray(fp.indices, dtype=np.uint32)
    assert indices.tolist() == sorted(indices.tolist())
    padding_bits = {int(bit) for bit in indices if int(bit) >= fp.num_bits}
    assert not padding_bits
    return {int(bit) for bit in indices}


def _rdkit_atom_pair_on_bits(
    smiles: str,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_2d: bool = True,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
) -> set[int]:
    generator = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=min_distance,
        maxDistance=max_distance,
        includeChirality=use_chirality,
        use2D=use_2d,
        countSimulation=count_simulation,
        countBounds=tuple(count_bounds) if count_bounds is not None else None,
        fpSize=num_bits,
    )
    return set(generator.GetFingerprint(_rdkit_mol(smiles)).GetOnBits())


def _rdkit_atom_pair_sparse_on_bits(
    smiles: str,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
    use_2d: bool = True,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
) -> tuple[int, set[int]]:
    generator = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=min_distance,
        maxDistance=max_distance,
        includeChirality=use_chirality,
        use2D=use_2d,
        countSimulation=count_simulation,
        countBounds=tuple(count_bounds) if count_bounds is not None else None,
    )
    fp = generator.GetSparseFingerprint(_rdkit_mol(smiles))
    return int(fp.GetNumBits()), set(fp.GetOnBits())


def _rdkit_atom_pair_counts(
    smiles: str,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    num_bits: int = 2048,
    use_chirality: bool = False,
    use_2d: bool = True,
) -> dict[int, int]:
    generator = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=min_distance,
        maxDistance=max_distance,
        includeChirality=use_chirality,
        use2D=use_2d,
        fpSize=num_bits,
    )
    return {
        int(index): int(count)
        for index, count in generator.GetCountFingerprint(_rdkit_mol(smiles)).GetNonzeroElements().items()
    }


def _rdkit_atom_pair_sparse_counts(
    smiles: str,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
    use_2d: bool = True,
) -> tuple[int, dict[int, int]]:
    generator = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=min_distance,
        maxDistance=max_distance,
        includeChirality=use_chirality,
        use2D=use_2d,
    )
    fp = generator.GetSparseCountFingerprint(_rdkit_mol(smiles))
    return (
        int(fp.GetLength()),
        {int(index): int(count) for index, count in fp.GetNonzeroElements().items()},
    )


def _atom_pair_descriptor_key(raw_id: int, *, use_chirality: bool = False) -> str:
    num_path_bits = 5
    code_size = 9 + (2 if use_chirality else 0)
    code_mask = (1 << code_size) - 1
    distance = raw_id & ((1 << num_path_bits) - 1)
    first_code = (raw_id >> num_path_bits) & code_mask
    second_code = (raw_id >> (num_path_bits + code_size)) & code_mask
    return f"{first_code}_{distance}_{second_code}"


def _rdkit_atom_pair_descriptor_counts(
    smiles: str,
    *,
    min_distance: int = 1,
    max_distance: int = 30,
    use_chirality: bool = False,
) -> dict[str, int]:
    _, sparse_counts = _rdkit_atom_pair_sparse_counts(
        smiles,
        min_distance=min_distance,
        max_distance=max_distance,
        use_chirality=use_chirality,
    )
    return {
        _atom_pair_descriptor_key(raw_id, use_chirality=use_chirality): count
        for raw_id, count in sparse_counts.items()
    }


def _descriptor_string_counts(descriptors) -> dict[str, int]:
    return {
        str(key): int(count)
        for key, count in zip(descriptors.string_keys, descriptors.counts, strict=True)
    }


_ALIGNED_TETRAHEDRAL_SMILES = [
    "F[C@](Cl)(Br)I",
    "F[C@@](Cl)(Br)I",
]


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "c1ccc(O)cc1",
        "C1CCCCC1",
        "CC(=O)O",
        "C[NH+](C)C",
        "O=C([O-])C",
        "C=C",
        "C#N",
    ],
)
@pytest.mark.parametrize("num_bits", [128, 2048])
def test_atom_pair_binary_matches_rdkit_default_options(smiles: str, num_bits: int):
    import oefp

    fp = oefp.atom_pair_fingerprint(_openeye_mol(smiles), num_bits=num_bits)

    assert fp.num_bits == num_bits
    assert _oefp_on_bits(fp) == _rdkit_atom_pair_on_bits(smiles, num_bits=num_bits)


@pytest.mark.parametrize(
    ("smiles", "min_distance", "max_distance"),
    [
        ("CCO", 1, 1),
        ("CCO", 2, 2),
        ("CCCC", 1, 2),
        ("CCCC", 2, 3),
        ("c1ccccc1", 1, 3),
    ],
)
def test_atom_pair_binary_distance_bounds_match_rdkit(
    smiles: str,
    min_distance: int,
    max_distance: int,
):
    import oefp

    fp = oefp.atom_pair_fingerprint(
        _openeye_mol(smiles),
        min_distance=min_distance,
        max_distance=max_distance,
        num_bits=256,
    )

    assert _oefp_on_bits(fp) == _rdkit_atom_pair_on_bits(
        smiles,
        min_distance=min_distance,
        max_distance=max_distance,
        num_bits=256,
    )


@pytest.mark.parametrize("smiles", ["CCO", "CC(C)O", "c1ccccc1", "C1CCCCC1"])
def test_atom_pair_binary_without_count_simulation_matches_rdkit(smiles: str):
    import oefp

    fp = oefp.atom_pair_fingerprint(
        _openeye_mol(smiles),
        num_bits=512,
        count_simulation=False,
    )

    assert _oefp_on_bits(fp) == _rdkit_atom_pair_on_bits(
        smiles,
        num_bits=512,
        count_simulation=False,
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "c1ccc(O)cc1",
        "C1CCCCC1",
        "CC(=O)O",
        "C[NH+](C)C",
        "O=C([O-])C",
        "C=C",
        "C#N",
    ],
)
@pytest.mark.parametrize("num_bits", [128, 2048])
def test_atom_pair_counts_match_rdkit_default_options(smiles: str, num_bits: int):
    import oefp

    fp = oefp.atom_pair_count_fingerprint(_openeye_mol(smiles), num_bits=num_bits)

    assert fp.num_bits == num_bits
    assert _oefp_counts(fp) == _rdkit_atom_pair_counts(smiles, num_bits=num_bits)


@pytest.mark.parametrize(
    ("smiles", "min_distance", "max_distance"),
    [
        ("CCO", 1, 1),
        ("CCO", 2, 2),
        ("CCCC", 1, 2),
        ("CCCC", 2, 3),
        ("c1ccccc1", 1, 3),
    ],
)
def test_atom_pair_count_distance_bounds_match_rdkit(
    smiles: str,
    min_distance: int,
    max_distance: int,
):
    import oefp

    fp = oefp.atom_pair_count_fingerprint(
        _openeye_mol(smiles),
        min_distance=min_distance,
        max_distance=max_distance,
        num_bits=256,
    )

    assert _oefp_counts(fp) == _rdkit_atom_pair_counts(
        smiles,
        min_distance=min_distance,
        max_distance=max_distance,
        num_bits=256,
    )


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "c1ccc(O)cc1",
        "C1CCCCC1",
        "CC(=O)O",
        "C[NH+](C)C",
        "O=C([O-])C",
        "C=C",
        "C#N",
    ],
)
def test_atom_pair_sparse_binary_matches_rdkit_default_options(smiles: str):
    import oefp

    fp = oefp.atom_pair_sparse_fingerprint(_openeye_mol(smiles))
    expected_num_bits, expected_bits = _rdkit_atom_pair_sparse_on_bits(smiles)

    assert fp.num_bits == expected_num_bits
    assert _oefp_sparse_bits(fp) == expected_bits


@pytest.mark.parametrize("smiles", ["CCO", "CC(C)O", "c1ccccc1", "C1CCCCC1"])
def test_atom_pair_sparse_binary_without_count_simulation_matches_rdkit(smiles: str):
    import oefp

    fp = oefp.atom_pair_sparse_fingerprint(
        _openeye_mol(smiles),
        count_simulation=False,
    )
    expected_num_bits, expected_bits = _rdkit_atom_pair_sparse_on_bits(
        smiles,
        count_simulation=False,
    )

    assert fp.num_bits == expected_num_bits
    assert _oefp_sparse_bits(fp) == expected_bits


@pytest.mark.parametrize(
    ("smiles", "min_distance", "max_distance"),
    [
        ("CCO", 1, 1),
        ("CCO", 2, 2),
        ("CCCC", 1, 2),
        ("CCCC", 2, 3),
        ("c1ccccc1", 1, 3),
    ],
)
def test_atom_pair_sparse_binary_distance_bounds_match_rdkit(
    smiles: str,
    min_distance: int,
    max_distance: int,
):
    import oefp

    fp = oefp.atom_pair_sparse_fingerprint(
        _openeye_mol(smiles),
        min_distance=min_distance,
        max_distance=max_distance,
    )
    expected_num_bits, expected_bits = _rdkit_atom_pair_sparse_on_bits(
        smiles,
        min_distance=min_distance,
        max_distance=max_distance,
    )

    assert fp.num_bits == expected_num_bits
    assert _oefp_sparse_bits(fp) == expected_bits


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "CCO",
        "CC(C)O",
        "c1ccccc1",
        "c1ccc(O)cc1",
        "C1CCCCC1",
        "CC(=O)O",
        "C[NH+](C)C",
        "O=C([O-])C",
        "C=C",
        "C#N",
    ],
)
def test_atom_pair_sparse_counts_match_rdkit_default_options(smiles: str):
    import oefp

    fp = oefp.atom_pair_sparse_count_fingerprint(_openeye_mol(smiles))
    expected_num_bits, expected_counts = _rdkit_atom_pair_sparse_counts(smiles)

    assert fp.num_bits == expected_num_bits
    assert _oefp_counts(fp) == expected_counts


@pytest.mark.parametrize(
    ("smiles", "min_distance", "max_distance"),
    [
        ("CCO", 1, 1),
        ("CCO", 2, 2),
        ("CCCC", 1, 2),
        ("CCCC", 2, 3),
        ("c1ccccc1", 1, 3),
    ],
)
def test_atom_pair_sparse_count_distance_bounds_match_rdkit(
    smiles: str,
    min_distance: int,
    max_distance: int,
):
    import oefp

    fp = oefp.atom_pair_sparse_count_fingerprint(
        _openeye_mol(smiles),
        min_distance=min_distance,
        max_distance=max_distance,
    )
    expected_num_bits, expected_counts = _rdkit_atom_pair_sparse_counts(
        smiles,
        min_distance=min_distance,
        max_distance=max_distance,
    )

    assert fp.num_bits == expected_num_bits
    assert _oefp_counts(fp) == expected_counts


@pytest.mark.parametrize("smiles", _ALIGNED_TETRAHEDRAL_SMILES)
@pytest.mark.parametrize("num_bits", [256, 1024])
def test_atom_pair_binary_matches_rdkit_with_chirality(smiles: str, num_bits: int):
    import oefp

    fp = oefp.atom_pair_fingerprint(
        _openeye_mol(smiles),
        num_bits=num_bits,
        use_chirality=True,
    )

    assert fp.num_bits == num_bits
    assert _oefp_on_bits(fp) == _rdkit_atom_pair_on_bits(
        smiles,
        num_bits=num_bits,
        use_chirality=True,
    )


@pytest.mark.parametrize("smiles", _ALIGNED_TETRAHEDRAL_SMILES)
def test_atom_pair_sparse_binary_matches_rdkit_with_chirality(smiles: str):
    import oefp

    fp = oefp.atom_pair_sparse_fingerprint(
        _openeye_mol(smiles),
        use_chirality=True,
    )
    expected_num_bits, expected_bits = _rdkit_atom_pair_sparse_on_bits(
        smiles,
        use_chirality=True,
    )

    assert fp.num_bits == expected_num_bits
    assert _oefp_sparse_bits(fp) == expected_bits


@pytest.mark.parametrize("smiles", _ALIGNED_TETRAHEDRAL_SMILES)
@pytest.mark.parametrize("num_bits", [256, 1024])
def test_atom_pair_counts_match_rdkit_with_chirality(smiles: str, num_bits: int):
    import oefp

    fp = oefp.atom_pair_count_fingerprint(
        _openeye_mol(smiles),
        num_bits=num_bits,
        use_chirality=True,
    )

    assert fp.num_bits == num_bits
    assert _oefp_counts(fp) == _rdkit_atom_pair_counts(
        smiles,
        num_bits=num_bits,
        use_chirality=True,
    )


@pytest.mark.parametrize("smiles", _ALIGNED_TETRAHEDRAL_SMILES)
def test_atom_pair_sparse_counts_match_rdkit_with_chirality(smiles: str):
    import oefp

    fp = oefp.atom_pair_sparse_count_fingerprint(
        _openeye_mol(smiles),
        use_chirality=True,
    )
    expected_num_bits, expected_counts = _rdkit_atom_pair_sparse_counts(
        smiles,
        use_chirality=True,
    )

    assert fp.num_bits == expected_num_bits
    assert _oefp_counts(fp) == expected_counts


@pytest.mark.parametrize("smiles", _ALIGNED_TETRAHEDRAL_SMILES)
def test_atom_pair_descriptor_string_counts_match_rdkit_with_chirality(smiles: str):
    import oefp

    descriptors = oefp.atom_pair_descriptors(
        _openeye_mol(smiles),
        use_chirality=True,
    )

    assert descriptors.value_type == "string"
    assert _descriptor_string_counts(descriptors) == _rdkit_atom_pair_descriptor_counts(
        smiles,
        use_chirality=True,
    )


def test_atom_pair_accepts_chirality_and_rejects_3d_options():
    import oefp

    mol = _openeye_mol("CCO")
    fp = oefp.atom_pair_fingerprint(mol, use_chirality=True)
    assert fp.num_bits == 2048
    with pytest.raises(ValueError, match="3D"):
        oefp.atom_pair_fingerprint(mol, use_2d=False)
    count_fp = oefp.atom_pair_count_fingerprint(mol, use_chirality=True)
    assert count_fp.num_bits == 2048
    with pytest.raises(ValueError, match="3D"):
        oefp.atom_pair_count_fingerprint(mol, use_2d=False)
    sparse_fp = oefp.atom_pair_sparse_fingerprint(mol, use_chirality=True)
    assert sparse_fp.popcount > 0
    with pytest.raises(ValueError, match="3D"):
        oefp.atom_pair_sparse_fingerprint(mol, use_2d=False)
    sparse_count_fp = oefp.atom_pair_sparse_count_fingerprint(mol, use_chirality=True)
    assert sparse_count_fp.nonzero_count > 0
    with pytest.raises(ValueError, match="3D"):
        oefp.atom_pair_sparse_count_fingerprint(mol, use_2d=False)


@pytest.mark.parametrize(
    ("kwargs", "exception_type", "match"),
    [
        ({"min_distance": -1}, ValueError, "Atom Pair min_distance"),
        ({"min_distance": 2**32}, ValueError, "Atom Pair min_distance"),
        ({"max_distance": 2**32}, ValueError, "Atom Pair max_distance"),
        ({"num_bits": 0}, ValueError, "Atom Pair num_bits"),
        ({"num_bits": 2**32}, ValueError, "Atom Pair num_bits"),
        ({"min_distance": 1.5}, TypeError, "Atom Pair min_distance"),
        ({"max_distance": 1.5}, TypeError, "Atom Pair max_distance"),
        ({"num_bits": 128.5}, TypeError, "Atom Pair num_bits"),
        ({"count_bounds": [-1]}, ValueError, "Atom Pair count_bounds"),
        ({"count_bounds": [2**32]}, ValueError, "Atom Pair count_bounds"),
        ({"count_bounds": [1.5]}, TypeError, "Atom Pair count_bounds"),
    ],
)
def test_atom_pair_rejects_invalid_python_options(
    kwargs: dict[str, Any],
    exception_type: type[Exception],
    match: str,
):
    import oefp

    with pytest.raises(exception_type, match=match):
        oefp.atom_pair_fingerprint(_openeye_mol("CCO"), **cast(Any, kwargs))
