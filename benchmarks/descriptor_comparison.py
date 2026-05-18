#!/usr/bin/env python3
"""Compare raw descriptor behavior against folded fingerprints."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_SMILES = (
    "CCO",
    "CC=O",
    "CC#N",
    "c1ccccc1",
    "c1ccncc1",
    "c1ccc(O)cc1",
    "CC(C)(C)Cl",
    "CC(=O)Oc1ccccc1C(=O)O",
    "C1CCCCC1",
    "Cn1cnc2n(C)c(=O)n(C)c(=O)c12",
    "CCN(CC)CC",
    "O=C(O)c1ccccc1",
    "c1ccc2ccccc2c1",
    "CC(C)Oc1ccc(cc1)C(C)C(=O)O",
)


@dataclass(frozen=True)
class TimedResult:
    """Value returned by a measured call."""

    value: Any
    seconds: float


@dataclass(frozen=True)
class ComparisonSummary:
    """One raw-vs-folded comparison row."""

    family: str
    num_bits: int
    molecules: int
    pairs: int
    raw_generation_s: float
    folded_generation_s: float
    raw_pdist_s: float
    folded_pdist_s: float
    mean_abs_delta: float
    max_abs_delta: float
    raw_storage_bytes: int
    folded_storage_bytes: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles-file", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--num-bits",
        type=int,
        nargs="+",
        default=[128, 256, 512, 2048],
        help="Folded fingerprint sizes to compare.",
    )
    parser.add_argument(
        "--family",
        choices=("atom-pair", "morgan", "both"),
        default="both",
        help="Descriptor family to benchmark.",
    )
    return parser


def _load_smiles(path: Path | None, limit: int) -> list[str]:
    if path is None:
        smiles = list(DEFAULT_SMILES)
    else:
        smiles = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                smiles.append(stripped.split()[0])
                if limit > 0 and len(smiles) >= limit:
                    break
    if limit > 0:
        smiles = smiles[:limit]
    if len(smiles) < 2:
        raise SystemExit("At least two SMILES are required for pairwise comparison.")
    return smiles


def _parse_openeye_molecules(smiles: Sequence[str]) -> list[Any]:
    from openeye import oechem  # type: ignore[import-not-found]

    mols = []
    for smi in smiles:
        mol = oechem.OEGraphMol()
        if not oechem.OESmilesToMol(mol, smi):
            raise SystemExit(f"OpenEye failed to parse SMILES: {smi}")
        mols.append(mol)
    return mols


def _time_call(fn: Callable[[], Any]) -> TimedResult:
    start = time.perf_counter()
    value = fn()
    return TimedResult(value=value, seconds=time.perf_counter() - start)


def _descriptor_storage_bytes(batch: Any) -> int:
    key_bytes = 0
    if batch.value_type == "string":
        key_bytes = sum(len(key.encode("utf-8")) for key in batch.string_keys)
    elif batch.value_type == "integer":
        key_bytes = 8 * len(batch.integer_keys)
    elif batch.value_type == "float":
        key_bytes = 8 * len(batch.float_keys)
    return int(key_bytes + batch.counts.nbytes + batch.offsets.nbytes)


def _fingerprint_storage_bytes(batch: Any) -> int:
    return int(batch.words.nbytes + batch.popcounts.nbytes)


def _paired_deltas(raw: np.ndarray, folded: np.ndarray) -> tuple[float, float]:
    if raw.shape != folded.shape:
        raise ValueError("raw and folded pdist vectors must have the same shape.")
    deltas = np.abs(raw - folded)
    return float(np.mean(deltas)), float(np.max(deltas))


def _families(selected: str) -> tuple[str, ...]:
    if selected == "both":
        return ("atom-pair", "morgan")
    return (selected,)


def _raw_descriptors(family: str, mols: Sequence[Any]) -> list[Any]:
    import oefp  # type: ignore[import-not-found]

    if family == "atom-pair":
        return [oefp.atom_pair_descriptors(mol) for mol in mols]
    if family == "morgan":
        return [oefp.morgan_descriptors(mol) for mol in mols]
    raise ValueError(f"Unknown descriptor family: {family}")


def _folded_fingerprints(family: str, mols: Sequence[Any], num_bits: int) -> list[Any]:
    import oefp  # type: ignore[import-not-found]

    if family == "atom-pair":
        generator = oefp.AtomPairGenerator(num_bits=num_bits)
    elif family == "morgan":
        generator = oefp.MorganGenerator(num_bits=num_bits)
    else:
        raise ValueError(f"Unknown descriptor family: {family}")
    return [generator.fingerprint(mol) for mol in mols]


def _compare_family(family: str, mols: Sequence[Any], num_bits: int) -> ComparisonSummary:
    import oefp  # type: ignore[import-not-found]

    raw_timed = _time_call(lambda: _raw_descriptors(family, mols))
    raw_batch = oefp.DescriptorBatch.from_descriptors(raw_timed.value)
    raw_pdist_timed = _time_call(
        lambda: oefp.pdist(raw_batch, oefp.Metric.tanimoto(), descriptor_mode="count_overlap")
    )
    raw_values = np.asarray(raw_pdist_timed.value, dtype=float)

    folded_timed = _time_call(lambda: _folded_fingerprints(family, mols, num_bits))
    folded_batch = oefp.OEFPBatch.from_fingerprints(folded_timed.value)
    folded_pdist_timed = _time_call(lambda: oefp.pdist(folded_batch, oefp.Metric.tanimoto()))
    folded_values = np.asarray(folded_pdist_timed.value, dtype=float)

    mean_abs_delta, max_abs_delta = _paired_deltas(raw_values, folded_values)
    return ComparisonSummary(
        family=family,
        num_bits=num_bits,
        molecules=len(mols),
        pairs=len(raw_values),
        raw_generation_s=raw_timed.seconds,
        folded_generation_s=folded_timed.seconds,
        raw_pdist_s=raw_pdist_timed.seconds,
        folded_pdist_s=folded_pdist_timed.seconds,
        mean_abs_delta=mean_abs_delta,
        max_abs_delta=max_abs_delta,
        raw_storage_bytes=_descriptor_storage_bytes(raw_batch),
        folded_storage_bytes=_fingerprint_storage_bytes(folded_batch),
    )


def _print_table(rows: Sequence[ComparisonSummary]) -> None:
    headers = (
        "family",
        "bits",
        "mols",
        "pairs",
        "mean|delta|",
        "max|delta|",
        "raw_gen_ms",
        "fold_gen_ms",
        "raw_pdist_ms",
        "fold_pdist_ms",
        "raw_bytes",
        "fold_bytes",
    )
    print(" ".join(f"{header:>13}" for header in headers))
    for row in rows:
        values = (
            row.family,
            str(row.num_bits),
            str(row.molecules),
            str(row.pairs),
            f"{row.mean_abs_delta:.6f}",
            f"{row.max_abs_delta:.6f}",
            f"{row.raw_generation_s * 1000.0:.3f}",
            f"{row.folded_generation_s * 1000.0:.3f}",
            f"{row.raw_pdist_s * 1000.0:.3f}",
            f"{row.folded_pdist_s * 1000.0:.3f}",
            str(row.raw_storage_bytes),
            str(row.folded_storage_bytes),
        )
        print(" ".join(f"{value:>13}" for value in values))

    for family in sorted({row.family for row in rows}):
        family_rows = [row for row in rows if row.family == family]
        best = min(family_rows, key=lambda row: row.mean_abs_delta)
        print(
            f"{family}: best mean absolute delta at {best.num_bits} bits "
            f"({best.mean_abs_delta:.6f}); median raw storage "
            f"{statistics.median(row.raw_storage_bytes for row in family_rows):.0f} bytes"
        )


def main() -> int:
    args = _parser().parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative.")
    if any(num_bits <= 0 for num_bits in args.num_bits):
        raise SystemExit("--num-bits values must be positive.")

    smiles = _load_smiles(args.smiles_file, args.limit)
    mols = _parse_openeye_molecules(smiles)
    rows = [
        _compare_family(family, mols, num_bits)
        for family in _families(args.family)
        for num_bits in args.num_bits
    ]
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
