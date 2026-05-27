"""RDKit conformance tests for OEFP Topological Torsions fingerprints."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

Chem = pytest.importorskip("rdkit.Chem", reason="RDKit is required for Topological Torsions conformance tests")
rdFingerprintGenerator = pytest.importorskip(
    "rdkit.Chem.rdFingerprintGenerator",
    reason="RDKit is required for Topological Torsions conformance tests",
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


def _rdkit_mol_preserving_hydrogens(smiles: str):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
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


def _oefp_counts64(fp) -> dict[int, int]:
    assert fp.indices.dtype == np.uint64
    assert fp.counts.dtype == np.uint32
    indices = np.asarray(fp.indices, dtype=np.uint64)
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


def _rdkit_generator(
    *,
    torsion_atom_count: int = 4,
    num_bits: int = 2048,
    use_chirality: bool = False,
    count_simulation: bool = True,
    count_bounds: Sequence[int] | None = None,
):
    return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
        includeChirality=use_chirality,
        torsionAtomCount=torsion_atom_count,
        countSimulation=count_simulation,
        countBounds=tuple(count_bounds) if count_bounds is not None else None,
        fpSize=num_bits,
    )


def _rdkit_topological_torsions_on_bits(smiles: str, **kwargs) -> set[int]:
    generator = _rdkit_generator(**kwargs)
    return set(generator.GetFingerprint(_rdkit_mol(smiles)).GetOnBits())


def _rdkit_topological_torsions_counts(smiles: str, **kwargs) -> dict[int, int]:
    generator = _rdkit_generator(**kwargs)
    return {
        int(index): int(count)
        for index, count in generator.GetCountFingerprint(_rdkit_mol(smiles)).GetNonzeroElements().items()
    }


def _rdkit_topological_torsions_sparse_on_bits(smiles: str, **kwargs) -> tuple[int, set[int]]:
    generator = _rdkit_generator(**kwargs)
    fp = generator.GetSparseFingerprint(_rdkit_mol(smiles))
    return int(fp.GetNumBits()), set(fp.GetOnBits())


def _rdkit_topological_torsions_sparse_counts(smiles: str, **kwargs) -> dict[int, int]:
    generator = _rdkit_generator(**kwargs)
    return {
        int(index): int(count)
        for index, count in generator.GetSparseCountFingerprint(
            _rdkit_mol(smiles)
        ).GetNonzeroElements().items()
    }


@pytest.mark.parametrize(
    "smiles",
    [
        "CCC",
        "CCCC",
        "CCCCC",
        "CC(C)CCO",
        "c1ccccc1",
        "c1ccc(O)cc1",
        "C1CC1",
        "C1CCCCC1",
        "CC(=O)O",
        "C[NH+](C)CC",
    ],
)
@pytest.mark.parametrize("num_bits", [128, 2048])
def test_topological_torsions_binary_matches_rdkit_default_options(smiles: str, num_bits: int):
    import oefp

    fp = oefp.topological_torsions_fingerprint(_openeye_mol(smiles), num_bits=num_bits)

    assert fp.num_bits == num_bits
    assert _oefp_on_bits(fp) == _rdkit_topological_torsions_on_bits(smiles, num_bits=num_bits)


@pytest.mark.parametrize(
    ("smiles", "torsion_atom_count"),
    [
        ("CCO", 2),
        ("CCCC", 3),
        ("CCCCC", 4),
        ("CCCCCC", 5),
        ("c1ccccc1", 4),
    ],
)
def test_topological_torsions_torsion_atom_count_matches_rdkit(
    smiles: str,
    torsion_atom_count: int,
):
    import oefp

    fp = oefp.topological_torsions_fingerprint(
        _openeye_mol(smiles),
        torsion_atom_count=torsion_atom_count,
        num_bits=512,
    )

    assert _oefp_on_bits(fp) == _rdkit_topological_torsions_on_bits(
        smiles,
        torsion_atom_count=torsion_atom_count,
        num_bits=512,
    )


@pytest.mark.parametrize("smiles", ["CCCC", "CC(C)CCO", "c1ccccc1", "C1CCCCC1"])
def test_topological_torsions_binary_without_count_simulation_matches_rdkit(smiles: str):
    import oefp

    fp = oefp.topological_torsions_fingerprint(
        _openeye_mol(smiles),
        num_bits=512,
        count_simulation=False,
    )

    assert _oefp_on_bits(fp) == _rdkit_topological_torsions_on_bits(
        smiles,
        num_bits=512,
        count_simulation=False,
    )


@pytest.mark.parametrize("smiles", ["CCCC", "CCCCC", "CC(C)CCO", "c1ccccc1", "C1CC1"])
def test_topological_torsions_count_fingerprint_matches_rdkit(smiles: str):
    import oefp

    fp = oefp.topological_torsions_count_fingerprint(_openeye_mol(smiles), num_bits=512)

    assert fp.num_bits == 512
    assert _oefp_counts(fp) == _rdkit_topological_torsions_counts(smiles, num_bits=512)


@pytest.mark.parametrize("smiles", ["CCCC", "CCCCC", "CC(C)CCO", "c1ccccc1", "C1CC1"])
@pytest.mark.parametrize("count_simulation", [True, False])
def test_topological_torsions_sparse_fingerprint_matches_rdkit(
    smiles: str,
    count_simulation: bool,
):
    import oefp

    fp = oefp.topological_torsions_sparse_fingerprint(
        _openeye_mol(smiles),
        count_simulation=count_simulation,
    )
    expected_size, expected_bits = _rdkit_topological_torsions_sparse_on_bits(
        smiles,
        count_simulation=count_simulation,
    )

    assert fp.num_bits == expected_size
    assert _oefp_sparse_bits(fp) == expected_bits


@pytest.mark.parametrize("smiles", ["CCCC", "CCCCC", "CC(C)CCO", "c1ccccc1", "C1CC1"])
def test_topological_torsions_sparse_count_fingerprint_matches_rdkit(smiles: str):
    import oefp

    fp = oefp.topological_torsions_sparse_count_fingerprint(_openeye_mol(smiles))

    assert fp.num_bits == 1 << 36
    assert max(fp.indices.tolist(), default=0) > np.iinfo(np.uint32).max or not fp.indices.size
    assert _oefp_counts64(fp) == _rdkit_topological_torsions_sparse_counts(smiles)


def test_topological_torsions_explicit_hydrogens_match_rdkit_when_preserved():
    import oefp

    smiles = "[H]C([H])([H])C([H])([H])C([H])([H])C([H])([H])[H]"
    oe_mol = _openeye_mol(smiles)
    rd_mol = _rdkit_mol_preserving_hydrogens(smiles)
    generator = _rdkit_generator(num_bits=512)

    assert _oefp_on_bits(
        oefp.topological_torsions_fingerprint(oe_mol, num_bits=512)
    ) == set(generator.GetFingerprint(rd_mol).GetOnBits())
    assert _oefp_counts(
        oefp.topological_torsions_count_fingerprint(oe_mol, num_bits=512)
    ) == {
        int(index): int(count)
        for index, count in generator.GetCountFingerprint(rd_mol).GetNonzeroElements().items()
    }
    assert _oefp_sparse_bits(
        oefp.topological_torsions_sparse_fingerprint(oe_mol)
    ) == set(generator.GetSparseFingerprint(rd_mol).GetOnBits())
    assert _oefp_counts64(
        oefp.topological_torsions_sparse_count_fingerprint(oe_mol)
    ) == {
        int(index): int(count)
        for index, count in generator.GetSparseCountFingerprint(rd_mol).GetNonzeroElements().items()
    }
