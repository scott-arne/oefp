"""Opt-in heuristic sweep over SureChEMBL for canonical-identity errors.

Run: OEFP_SURECHEMBL_PARQUET=/path/compounds.parquet \
     PYTHONPATH=python python -m pytest tests/python/surechembl_identity_sweep.py -m surechembl -q

Heuristics (compare RAW per-source outputs):
  * missed identity     — untagged cross-source column pairs numerically identical across the whole sample.
  * misassigned identity — tagged (shared canonical_id) pairs that diverge on any molecule.
This is a heuristic, not a proof; unparseable molecules are counted and skipped.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PARQUET = Path(os.environ.get("OEFP_SURECHEMBL_PARQUET", "/Users/johnss51/Downloads/compounds.parquet"))
_SAMPLE = int(os.environ.get("OEFP_SURECHEMBL_SAMPLE", "50000"))
_MISSED = os.environ.get("OEFP_SURECHEMBL_MISSED", "0") == "1"


@pytest.mark.surechembl
@pytest.mark.skipif(not _PARQUET.exists(), reason=f"SureChEMBL parquet not found at {_PARQUET}")
def test_surechembl_identity_sweep():
    import pyarrow.parquet as pq
    from openeye import oechem

    import oefp

    # Wrap each source in a single-source calculator to access schema and compute
    calculators = [
        oefp.DescriptorCalculator([oefp.MordredDescriptorSource()]),
        oefp.DescriptorCalculator([oefp.OpenEyePropertyDescriptorSource()]),
    ]

    parquet = pq.ParquetFile(_PARQUET)
    smiles_seen = 0
    skipped = 0

    # tagged pairs: canonical_id -> per-source (source_index, column_name)
    tagged = {}
    for idx, calc in enumerate(calculators):
        for definition in calc.schema.definitions:
            if definition.canonical_id:
                tagged.setdefault(definition.canonical_id, []).append((idx, definition.name))

    divergences = []

    # optional: missed-identity scan (expensive: all cross-source untagged pairs)
    missed_candidates = {}
    if _MISSED:
        # build untagged cross-source pairs
        for i, calc_a in enumerate(calculators):
            for def_a in calc_a.schema.definitions:
                if not def_a.canonical_id:
                    for j, calc_b in enumerate(calculators):
                        if j <= i:
                            continue
                        for def_b in calc_b.schema.definitions:
                            if not def_b.canonical_id:
                                pair = ((i, def_a.name), (j, def_b.name))
                                missed_candidates[pair] = True  # "always equal so far"

    for record_batch in parquet.iter_batches(columns=["smiles"], batch_size=4096):
        for smiles in record_batch.column("smiles").to_pylist():
            if smiles_seen >= _SAMPLE:
                break
            mol = oechem.OEGraphMol()
            if not oechem.OESmilesToMol(mol, smiles):
                skipped += 1
                continue
            smiles_seen += 1
            rows = [calc.compute(mol) for calc in calculators]

            # misassigned identity check
            for canonical_id, members in tagged.items():
                if len(members) < 2:
                    continue
                values = [rows[i][name] for (i, name) in members if rows[i][name] is not None]
                if len(values) >= 2 and any(v != values[0] for v in values[1:]):
                    divergences.append((smiles, canonical_id, values))

            # missed identity scan (optional)
            if _MISSED:
                to_drop = []
                for pair, still_equal in missed_candidates.items():
                    if not still_equal:
                        continue
                    (i_a, name_a), (i_b, name_b) = pair
                    val_a = rows[i_a][name_a]
                    val_b = rows[i_b][name_b]
                    if val_a is None or val_b is None:
                        to_drop.append(pair)
                    elif val_a != val_b:
                        missed_candidates[pair] = False
                for pair in to_drop:
                    del missed_candidates[pair]

        if smiles_seen >= _SAMPLE:
            break

    print(f"sampled={smiles_seen} skipped={skipped} tagged_pairs={sum(1 for m in tagged.values() if len(m) >= 2)}")

    # report missed-identity candidates (advisory, not asserted)
    if _MISSED:
        candidates = [pair for pair, still_equal in missed_candidates.items() if still_equal]
        if candidates:
            print("Missed-identity candidates (always identical across sample):")
            for (i_a, name_a), (i_b, name_b) in candidates[:50]:
                calc_a_source = "Mordred" if i_a == 0 else "OpenEye"
                calc_b_source = "Mordred" if i_b == 0 else "OpenEye"
                print(f"  [{calc_a_source}].{name_a} == [{calc_b_source}].{name_b}")
        else:
            print("No missed-identity candidates found.")

    # assert: no tagged divergences
    assert not divergences, f"tagged canonical_id divergences (misassigned identity): {divergences[:20]}"
