"""RDKit conformance tests for OEFP Atom Pair binary fingerprints."""

from __future__ import annotations

from collections.abc import Sequence

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


def test_atom_pair_rejects_unsupported_options():
    import oefp

    mol = _openeye_mol("CCO")
    with pytest.raises(ValueError, match="chirality"):
        oefp.atom_pair_fingerprint(mol, use_chirality=True)
    with pytest.raises(ValueError, match="3D"):
        oefp.atom_pair_fingerprint(mol, use_2d=False)
    with pytest.raises(ValueError, match="chirality"):
        oefp.atom_pair_count_fingerprint(mol, use_chirality=True)
    with pytest.raises(ValueError, match="3D"):
        oefp.atom_pair_count_fingerprint(mol, use_2d=False)
