"""Tests for the reusable Python Atom Pair generator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")


def _openeye_mol(smiles: str) -> Any:
    from openeye import oechem

    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


def _on_bits(fp: Any) -> set[int]:
    words = np.asarray(fp.words, dtype=np.uint64)
    bits: set[int] = set()
    for word_index, word in enumerate(words):
        value = int(word)
        while value:
            low_bit = value & -value
            bits.add(word_index * 64 + (low_bit.bit_length() - 1))
            value ^= low_bit
    return bits


def test_atom_pair_generator_matches_functional_api_default_options():
    import oefp

    mol = _openeye_mol("CC(=O)Oc1ccccc1C(=O)O")
    generator = oefp.AtomPairGenerator()

    generated = generator.fingerprint(mol)
    functional = oefp.atom_pair_fingerprint(mol)

    assert generated.num_bits == functional.num_bits
    assert generated.popcount == functional.popcount
    assert generated.words.tolist() == functional.words.tolist()
    assert _on_bits(generated) == _on_bits(functional)


def test_atom_pair_generator_matches_functional_api_nondefault_options():
    import oefp

    mol = _openeye_mol("c1ccc(O)cc1")
    generator = oefp.AtomPairGenerator(
        min_distance=1,
        max_distance=4,
        num_bits=512,
        count_simulation=False,
    )

    generated = generator.fingerprint(mol)
    functional = oefp.atom_pair_fingerprint(
        mol,
        min_distance=1,
        max_distance=4,
        num_bits=512,
        count_simulation=False,
    )

    assert generated.num_bits == 512
    assert generated.words.tolist() == functional.words.tolist()


def test_atom_pair_generator_count_simulation_matches_functional_api():
    import oefp

    mol = _openeye_mol("CCC(CC)CO")
    generator = oefp.AtomPairGenerator(
        num_bits=256,
        count_simulation=True,
        count_bounds=[1, 2, 4],
    )

    generated = generator.fingerprint(mol)
    functional = oefp.atom_pair_fingerprint(
        mol,
        num_bits=256,
        count_simulation=True,
        count_bounds=[1, 2, 4],
    )

    assert generated.words.tolist() == functional.words.tolist()


def test_atom_pair_generator_rejects_invalid_options():
    import oefp

    with pytest.raises(ValueError, match="num_bits must be greater than zero"):
        oefp.AtomPairGenerator(num_bits=0)

    with pytest.raises(ValueError, match="chirality conformance"):
        oefp.AtomPairGenerator(use_chirality=True)

    with pytest.raises(ValueError, match="count_bounds cannot be empty"):
        oefp.AtomPairGenerator(count_simulation=True, count_bounds=[])
