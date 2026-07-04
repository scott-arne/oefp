"""Lifetime guard for the CalculateBatch molecule-vector SWIG typemap.

``DescriptorCalculator::CalculateBatch`` releases the GIL and dereferences the
borrowed molecule pointers from worker threads, so every molecule passed in must
stay alive until the call returns. A plain ``list``/``tuple`` keeps its own
strong reference to each element, but ``PySequence_Check`` also admits lazy
sequences whose ``__getitem__`` mints a fresh molecule proxy per call and keeps
no reference to it. The typemap must retain a strong reference to every converted
element across the C++ call; otherwise the only reference is dropped and the C++
molecule is freed before ``CalculateBatch`` reads it (a use-after-free).

The use-after-free RED signal is inherently unreliable -- freed memory is not
always reused before the read, so a broken build may still pass intermittently.
These tests are therefore a permanent guard and a documented reproduction of the
lazy-sequence lifetime contract rather than a deterministic crash detector.
"""

import pytest

pytest.importorskip("openeye.oechem", reason="OpenEye Toolkits not installed")


def _build_calculator():
    """Build an OpenEye-property calculator through the native SWIG layer.

    The Python-friendly wrapper does not exist yet, so the calculator is
    assembled from ``oefp._native`` exactly as the Task 7 smoke test did. The
    OpenEye property source is used because it is fast and exposes a stable
    ``MolecularWeight`` column for value assertions.
    """
    from oefp import _native

    entries = _native.DescriptorSourceEntryVector()
    entries.push_back(_native.DescriptorSourceEntry(_native.OpenEyePropertyDescriptorSource()))
    return _native._NativeDescriptorCalculator(entries)


def _mol(smiles):
    from openeye import oechem

    mol = oechem.OEGraphMol()
    assert oechem.OESmilesToMol(mol, smiles)
    return mol


class LazyMolSequence:
    """Sequence that constructs a fresh molecule per ``__getitem__`` call.

    No reference to the produced molecule is retained, so the reference the
    typemap obtains from ``PySequence_GetItem`` is the only strong reference to
    the proxy. This is the case the borrowed-pointer lifetime fix must handle.
    """

    def __init__(self, smiles_list):
        self._smiles = list(smiles_list)

    def __len__(self):
        return len(self._smiles)

    def __getitem__(self, index):
        if index >= len(self._smiles):
            raise IndexError(index)
        return _mol(self._smiles[index])


# Ethanol and benzene exact molecular weights from the OpenEye property source.
_EXPECTED_MW = pytest.approx([46.04186481399999, 78.046950192])


def test_calculate_batch_over_lazy_sequence():
    """A lazy sequence whose items are freed unless the typemap holds them."""
    calc = _build_calculator()

    batch = calc.CalculateBatch(LazyMolSequence(["CCO", "c1ccccc1"]))

    assert batch.Size() == 2
    assert batch.Schema().Contains("MolecularWeight")
    assert list(batch.FloatColumn("MolecularWeight")) == _EXPECTED_MW


def test_calculate_batch_over_list_control():
    """Control: a plain list holds its own references and must also work."""
    calc = _build_calculator()

    batch = calc.CalculateBatch([_mol("CCO"), _mol("c1ccccc1")])

    assert batch.Size() == 2
    assert batch.Schema().Contains("MolecularWeight")
    assert list(batch.FloatColumn("MolecularWeight")) == _EXPECTED_MW


def test_calculate_batch_bad_element_still_raises():
    """A non-molecule element must raise TypeError without leaking references."""
    calc = _build_calculator()

    with pytest.raises(TypeError, match="OEMolBase"):
        calc.CalculateBatch([_mol("CCO"), object()])
