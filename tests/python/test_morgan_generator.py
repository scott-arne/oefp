"""Tests for the reusable Python Morgan generator."""

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


def test_morgan_generator_matches_functional_api_default_options():
    import oefp

    mol = _openeye_mol("CC(=O)Oc1ccccc1C(=O)O")
    generator = oefp.MorganGenerator()

    generated = generator.fingerprint(mol)
    functional = oefp.morgan_fingerprint(mol)

    assert generated.num_bits == functional.num_bits
    assert generated.popcount == functional.popcount
    assert generated.words.tolist() == functional.words.tolist()
    assert _on_bits(generated) == _on_bits(functional)


def test_morgan_generator_matches_functional_api_nondefault_options():
    import oefp

    mol = _openeye_mol("c1ccc(O)cc1")
    generator = oefp.MorganGenerator(
        radius=1,
        num_bits=512,
        use_bond_types=False,
        only_nonzero_invariants=True,
        include_ring_membership=False,
        include_redundant_environments=True,
    )

    generated = generator.fingerprint(mol)
    functional = oefp.morgan_fingerprint(
        mol,
        radius=1,
        num_bits=512,
        use_bond_types=False,
        only_nonzero_invariants=True,
        include_ring_membership=False,
        include_redundant_environments=True,
    )

    assert generated.num_bits == 512
    assert generated.words.tolist() == functional.words.tolist()


def test_morgan_generator_count_simulation_matches_functional_api():
    import oefp

    mol = _openeye_mol("CCC(CC)CO")
    generator = oefp.MorganGenerator(
        num_bits=256,
        count_simulation=True,
        count_bounds=[1, 2, 4],
    )

    generated = generator.fingerprint(mol)
    functional = oefp.morgan_fingerprint(
        mol,
        num_bits=256,
        count_simulation=True,
        count_bounds=[1, 2, 4],
    )

    assert generated.words.tolist() == functional.words.tolist()


def test_morgan_generator_rejects_invalid_options():
    import oefp

    with pytest.raises(ValueError, match="num_bits must be greater than zero"):
        oefp.MorganGenerator(num_bits=0)

    with pytest.raises(ValueError, match="chirality conformance"):
        oefp.MorganGenerator(use_chirality=True)

    with pytest.raises(ValueError, match="count_bounds cannot be empty"):
        oefp.MorganGenerator(count_simulation=True, count_bounds=[])
