"""Opt-in sweep: OEFP SPS vs rdkit.Chem.Descriptors.SPS over SureChEMBL.

Run: OEFP_SURECHEMBL_PARQUET=/path/compounds.parquet \
     PYTHONPATH=python python -m pytest tests/python/surechembl_sps_sweep.py -m surechembl -q

Acceptance gate (see the SPS spec): zero divergences outside the excluded
degenerate-input classes (isotope-hydrogen graph model, hydrogen-only
molecules). Any other divergence fails and must be fixed or trigger the SPS
fallback (exclude SPS -> 213).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PARQUET = Path(os.environ.get("OEFP_SURECHEMBL_PARQUET", "/Users/johnss51/Downloads/compounds.parquet"))
_SAMPLE = int(os.environ.get("OEFP_SURECHEMBL_SAMPLE", "50000"))

assert _SAMPLE > 0, f"OEFP_SURECHEMBL_SAMPLE must be positive, got {_SAMPLE}"


def _is_excluded_degenerate(rdkit_mol) -> bool:
    """True for the two surface-wide excluded classes: hydrogen-only molecules
    and molecules carrying isotope hydrogens (e.g. [2H]). These diverge across
    the whole OEFP surface, not SPS specifically, and are skipped, not counted."""
    heavy = [a for a in rdkit_mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not heavy:
        return True
    for a in rdkit_mol.GetAtoms():
        if a.GetAtomicNum() == 1 and a.GetIsotope() != 0:
            return True
    return False


@pytest.mark.surechembl
@pytest.mark.skipif(not _PARQUET.exists(), reason=f"SureChEMBL parquet not found at {_PARQUET}")
def test_surechembl_sps_sweep():
    import pyarrow.parquet as pq
    from openeye import oechem
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    import oefp

    calc = oefp.DescriptorCalculator([oefp.RDKitDescriptorSource()])
    sps_name = "SPS"

    parquet = pq.ParquetFile(_PARQUET)
    seen = 0
    skipped = 0
    degenerate = 0
    divergences: list[tuple[str, float, float]] = []

    for batch in parquet.iter_batches(columns=["smiles"], batch_size=4096):
        if seen >= _SAMPLE:
            break
        for smiles in batch.column("smiles").to_pylist():
            if seen >= _SAMPLE:
                break
            if not isinstance(smiles, str) or not smiles.strip():
                skipped += 1
                continue
            rdmol = Chem.MolFromSmiles(smiles)
            oemol = oechem.OEGraphMol()
            if rdmol is None or not oechem.OESmilesToMol(oemol, smiles):
                skipped += 1
                continue
            if _is_excluded_degenerate(rdmol):
                degenerate += 1
                continue
            seen += 1
            ref = Descriptors.SPS(rdmol)
            got = calc.compute(oemol)[sps_name]
            if got is None or got != pytest.approx(ref, rel=1e-9, abs=1e-9):
                divergences.append((smiles, ref, got))

    print(
        f"SPS sweep: sample={seen} skipped(unparseable)={skipped} "
        f"skipped(degenerate)={degenerate} divergences={len(divergences)}"
    )
    for smiles, ref, got in divergences[:25]:
        print(f"  DIVERGE {smiles}: rdkit={ref} oefp={got}")

    assert seen > 0, "no valid molecules processed (false-green guard)"
    assert not divergences, (
        f"{len(divergences)} non-degenerate SPS divergences; "
        f"fix or trigger the SPS fallback (exclude SPS -> 213)"
    )
