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
    with pytest.raises(ValueError, match="chirality"):
        oefp.morgan_count_fingerprint(mol, use_chirality=True)
