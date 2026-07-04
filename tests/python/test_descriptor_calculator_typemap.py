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

import os
import signal
import subprocess
import sys
from pathlib import Path

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


# Child program driving CalculateBatch with a sequence that reports a length no
# allocation can satisfy. Kept in a subprocess because an unguarded typemap lets
# std::vector::reserve throw std::length_error out of the SWIG wrapper, which
# aborts the interpreter (SIGABRT / exit 134) and would otherwise kill the whole
# pytest run instead of surfacing as a distinguishable failure.
_OVERSIZED_CHILD = """
import sys

from oefp import _native


class OversizedSequence:
    \"\"\"Sequence whose __len__ overflows any allocation the typemap attempts.\"\"\"

    def __len__(self):
        return sys.maxsize

    def __getitem__(self, index):
        raise IndexError(index)


entries = _native.DescriptorSourceEntryVector()
entries.push_back(_native.DescriptorSourceEntry(_native.OpenEyePropertyDescriptorSource()))
calc = _native._NativeDescriptorCalculator(entries)

try:
    calc.CalculateBatch(OversizedSequence())
except (TypeError, ValueError, MemoryError, RuntimeError, OverflowError) as exc:
    print("CAUGHT", type(exc).__name__)
    sys.exit(0)
except BaseException as exc:  # noqa: BLE001 - report any other Python-level error
    print("OTHER", type(exc).__name__)
    sys.exit(0)
print("NO_EXCEPTION")
sys.exit(1)
"""


def test_calculate_batch_oversized_sequence_does_not_abort():
    """An oversized __len__ must raise a Python error, never abort the process.

    The conversion-time reservation runs outside the GIL-release try/catch that
    guards the C++ call, so a bad length previously reached ``std::vector::reserve``
    and threw ``std::length_error`` straight out of the SWIG wrapper, aborting the
    interpreter (``libc++abi: terminating due to uncaught exception``, exit 134).
    The typemap now validates the length and guards the reservation, so the child
    must fail with a catchable Python exception rather than a process abort.
    """
    repo_python = Path(__file__).resolve().parents[2] / "python"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_python), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(repo_python)]
    )

    result = subprocess.run(
        [sys.executable, "-c", _OVERSIZED_CHILD],
        capture_output=True,
        text=True,
        env=env,
    )

    combined = result.stdout + result.stderr

    # Core property: the child did not abort via SIGABRT (exit 134 / -6) and did
    # not terminate through an uncaught C++ exception.
    assert result.returncode != -signal.SIGABRT, combined
    assert result.returncode != 134, combined
    assert "libc++abi" not in combined, combined
    assert "terminating due to uncaught exception" not in combined, combined

    # A catchable Python exception was raised (the child prints CAUGHT/OTHER and
    # exits 0) rather than silently succeeding on a nonsense length.
    assert result.returncode == 0, combined
    assert "NO_EXCEPTION" not in combined, combined
    assert ("CAUGHT" in combined) or ("OTHER" in combined), combined
