"""Tests for the reusable Topological Torsions generator wrappers."""

import numpy as np
import pytest

pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")


def _openeye_mol(smiles: str):
    from openeye import oechem

    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


def _on_bits(fp) -> set[int]:
    words = np.asarray(fp.words, dtype=np.uint64)
    bits: set[int] = set()
    for word_index, word in enumerate(words):
        value = int(word)
        while value:
            low_bit = value & -value
            bits.add(word_index * 64 + (low_bit.bit_length() - 1))
            value ^= low_bit
    return bits


def test_topological_torsions_generator_matches_functional_api():
    import oefp

    mol = _openeye_mol("CC(C)CCO")
    generator = oefp.TopologicalTorsionsGenerator(num_bits=512)

    generated = generator.fingerprint(mol)
    functional = oefp.topological_torsions_fingerprint(mol, num_bits=512)

    assert generated.spec == functional.spec
    assert _on_bits(generated) == _on_bits(functional)
    assert generated.spec.source_type == "TopologicalTorsions"


def test_topological_torsions_generator_rejects_invalid_options():
    import oefp

    with pytest.raises(ValueError, match="num_bits"):
        oefp.TopologicalTorsionsGenerator(num_bits=0)

    with pytest.raises(ValueError, match="torsion_atom_count"):
        oefp.TopologicalTorsionsGenerator(torsion_atom_count=0)

    with pytest.raises(ValueError, match="chirality"):
        oefp.TopologicalTorsionsGenerator(use_chirality=True)

    with pytest.raises(ValueError, match="count_bounds"):
        oefp.TopologicalTorsionsGenerator(count_simulation=True, count_bounds=[])
