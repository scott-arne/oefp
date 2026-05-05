"""RDKit conformance tests for OEFP Morgan binary fingerprints."""

from __future__ import annotations

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
        fpSize=num_bits,
        includeRedundantEnvironments=include_redundant_environments,
    )
    return set(generator.GetFingerprint(_rdkit_mol(smiles)).GetOnBits())


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
    expected_bits = _rdkit_on_bits(smiles, radius=2, num_bits=256, **kwargs)
    assert expected_bits != default_bits

    fp = oefp.morgan_fingerprint(_openeye_mol(smiles), radius=2, num_bits=256, **kwargs)

    assert _oefp_on_bits(fp) == expected_bits


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


def test_morgan_rejects_chirality_until_conformance_is_supported():
    import oefp

    with pytest.raises(ValueError, match="chirality"):
        oefp.morgan_fingerprint(_openeye_mol("C[C@H](F)Cl"), use_chirality=True)


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
    ]

    for mismatched_fp in mismatched_fps:
        with pytest.raises(RuntimeError, match="spec"):
            oefp.OEFPBatch.from_fingerprints([fp_a, mismatched_fp])
