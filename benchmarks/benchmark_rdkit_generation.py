#!/usr/bin/env python3
"""Benchmark OEFP reusable fingerprint generation against RDKit."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimingStats:
    """Summary statistics for repeated benchmark timings."""

    median_s: float
    iqr_s: float
    min_s: float
    max_s: float
    trials: int
    trials_molecule_count: int

    @property
    def per_mol_us(self) -> float:
        """Return median runtime per molecule in microseconds."""
        if self.trials_molecule_count == 0:
            return float("inf")
        return self.median_s / self.trials_molecule_count * 1_000_000.0


def _load_molecules(path: Path, max_mols: int) -> tuple[list[Any], list[Any]]:
    """Load matching OpenEye and RDKit molecules from a SMILES file."""
    from openeye import oechem  # type: ignore[import-not-found]
    from rdkit import Chem  # type: ignore[import-not-found]

    if not path.is_file():
        raise FileNotFoundError(f"SMILES file does not exist: {path}")

    oe_mols: list[Any] = []
    rd_mols: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(oe_mols) >= max_mols:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            smiles = stripped.split()[0]
            oe_mol = oechem.OEGraphMol()
            rd_mol = Chem.MolFromSmiles(smiles)
            if rd_mol is None or not oechem.OESmilesToMol(oe_mol, smiles):
                continue
            oe_mols.append(oe_mol)
            rd_mols.append(rd_mol)

    if not oe_mols:
        raise RuntimeError(f"No molecules parsed from {path}")
    return oe_mols, rd_mols


def _time_trials(
    fn: Callable[[], Any],
    *,
    trials: int,
    warmup: int,
    molecule_count: int,
) -> TimingStats:
    """Run warmup and measured trials for a benchmark callable."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)

    if len(samples) >= 2:
        quartiles = statistics.quantiles(samples, n=4, method="inclusive")
        iqr = quartiles[2] - quartiles[0]
    else:
        iqr = 0.0
    return TimingStats(
        median_s=statistics.median(samples),
        iqr_s=iqr,
        min_s=min(samples),
        max_s=max(samples),
        trials=trials,
        trials_molecule_count=molecule_count,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles",
        type=Path,
        default=Path("/Users/johnss51/Development/cpp/rdkit/Data/NCI/first_5K.smi"),
    )
    parser.add_argument("--max-mols", type=int, default=1500)
    parser.add_argument("--trials", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--pdist-size", type=int, default=400)
    parser.add_argument("--generation-max-ratio", type=float, default=1.10)
    parser.add_argument("--atom-pair-generation-max-ratio", type=float, default=1.00)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Validate parsed benchmark arguments before importing optional toolkits."""
    if args.max_mols <= 0:
        raise SystemExit("--max-mols must be greater than 0")
    if args.trials <= 0:
        raise SystemExit("--trials must be greater than 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be greater than or equal to 0")
    if args.pdist_size <= 1:
        raise SystemExit("--pdist-size must be greater than 1")
    if args.generation_max_ratio <= 0.0:
        raise SystemExit("--generation-max-ratio must be greater than 0")
    if args.atom_pair_generation_max_ratio <= 0.0:
        raise SystemExit("--atom-pair-generation-max-ratio must be greater than 0")


def _rdkit_bulk_pdist(rd_fps: list[Any], data_structs: Any) -> list[float]:
    output: list[float] = []
    for index, fp in enumerate(rd_fps):
        output.extend(data_structs.BulkTanimotoSimilarity(fp, rd_fps[index + 1 :]))
    return output


def main() -> int:
    """Run the benchmark and fail if OEFP misses the generation ratio gate."""
    args = _parser().parse_args()
    _validate_args(args)

    from rdkit import DataStructs  # type: ignore[import-not-found]
    from rdkit.Chem import rdFingerprintGenerator  # type: ignore[import-not-found]

    import oefp  # type: ignore[import-not-found]

    oe_mols, rd_mols = _load_molecules(args.smiles, args.max_mols)
    rd_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
        useBondTypes=True,
        includeRingMembership=True,
    )
    oe_morgan_generator = oefp.MorganGenerator(radius=2, num_bits=2048)
    rd_atom_pair_generator = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=1,
        maxDistance=30,
        includeChirality=False,
        use2D=True,
        countSimulation=True,
        fpSize=2048,
    )
    oe_atom_pair_generator = oefp.AtomPairGenerator(num_bits=2048)

    # Generation timing intentionally includes the Python loop plus Python
    # wrapper/object allocation over prebuilt molecules and reusable generators.
    oe_generation = _time_trials(
        lambda: [oe_morgan_generator.fingerprint(mol) for mol in oe_mols],
        trials=args.trials,
        warmup=args.warmup,
        molecule_count=len(oe_mols),
    )
    rd_morgan_generation = _time_trials(
        lambda: [rd_morgan_generator.GetFingerprint(mol) for mol in rd_mols],
        trials=args.trials,
        warmup=args.warmup,
        molecule_count=len(rd_mols),
    )
    morgan_generation_ratio = oe_generation.median_s / rd_morgan_generation.median_s
    oe_atom_pair_generation = _time_trials(
        lambda: [oe_atom_pair_generator.fingerprint(mol) for mol in oe_mols],
        trials=args.trials,
        warmup=args.warmup,
        molecule_count=len(oe_mols),
    )
    rd_atom_pair_generation = _time_trials(
        lambda: [rd_atom_pair_generator.GetFingerprint(mol) for mol in rd_mols],
        trials=args.trials,
        warmup=args.warmup,
        molecule_count=len(rd_mols),
    )
    atom_pair_generation_ratio = (
        oe_atom_pair_generation.median_s / rd_atom_pair_generation.median_s
    )

    print(f"data path:                 {args.smiles}")
    print(f"trials:                    {args.trials}")
    print(f"warmup:                    {args.warmup}")
    print(f"Morgan generation gate:    {args.generation_max_ratio:.3f}")
    print(f"Atom Pair generation gate: {args.atom_pair_generation_max_ratio:.3f}")
    print(f"molecules:                 {len(oe_mols)}")
    print(
        f"oefp Morgan median/IQR:    "
        f"{oe_generation.median_s:.6f}s / {oe_generation.iqr_s:.6f}s"
    )
    print(
        f"rdkit Morgan median/IQR:   "
        f"{rd_morgan_generation.median_s:.6f}s / {rd_morgan_generation.iqr_s:.6f}s"
    )
    print(f"oefp/rdkit Morgan generation:     {morgan_generation_ratio:.3f}")
    print(f"oefp Morgan per molecule:  {oe_generation.per_mol_us:.3f}us")
    print(f"rdkit Morgan per molecule: {rd_morgan_generation.per_mol_us:.3f}us")
    print(
        f"oefp Atom Pair median/IQR: "
        f"{oe_atom_pair_generation.median_s:.6f}s / {oe_atom_pair_generation.iqr_s:.6f}s"
    )
    print(
        f"rdkit Atom Pair median/IQR:"
        f" {rd_atom_pair_generation.median_s:.6f}s / {rd_atom_pair_generation.iqr_s:.6f}s"
    )
    print(f"oefp/rdkit Atom Pair generation:  {atom_pair_generation_ratio:.3f}")
    print(f"oefp Atom Pair per molecule:      {oe_atom_pair_generation.per_mol_us:.3f}us")
    print(f"rdkit Atom Pair per molecule:     {rd_atom_pair_generation.per_mol_us:.3f}us")

    pdist_n = min(args.pdist_size, len(oe_mols))
    if pdist_n <= 1:
        raise SystemExit(
            "pdist benchmark requires at least two loaded molecules; "
            "increase --max-mols or use a SMILES file with at least two parseable molecules"
        )
    pdist_oe_fps = [oe_morgan_generator.fingerprint(mol) for mol in oe_mols[:pdist_n]]
    pdist_rd_fps = [rd_morgan_generator.GetFingerprint(mol) for mol in rd_mols[:pdist_n]]
    batch = oefp.OEFPBatch.from_fingerprints(pdist_oe_fps)
    metric = oefp.Metric.tanimoto()
    oe_pdist_values = np.asarray(
        oefp.pdist(batch, metric, num_threads=0, chunk_size=256),
        dtype=float,
    )
    rd_pdist_values = np.asarray(_rdkit_bulk_pdist(pdist_rd_fps, DataStructs), dtype=float)
    np.testing.assert_allclose(oe_pdist_values, rd_pdist_values, rtol=1e-7, atol=1e-7)

    oe_pdist = _time_trials(
        lambda: oefp.pdist(batch, metric, num_threads=0, chunk_size=256),
        trials=args.trials,
        warmup=args.warmup,
        molecule_count=pdist_n,
    )
    rd_pdist = _time_trials(
        lambda: _rdkit_bulk_pdist(pdist_rd_fps, DataStructs),
        trials=args.trials,
        warmup=args.warmup,
        molecule_count=pdist_n,
    )
    pdist_speedup = rd_pdist.median_s / oe_pdist.median_s

    print(f"pdist molecules:            {pdist_n}")
    print(f"oefp pdist median/IQR:      {oe_pdist.median_s:.6f}s / {oe_pdist.iqr_s:.6f}s")
    print(f"rdkit bulk median/IQR:      {rd_pdist.median_s:.6f}s / {rd_pdist.iqr_s:.6f}s")
    print(f"rdkit/oefp pdist speedup:   {pdist_speedup:.3f}")

    if morgan_generation_ratio > args.generation_max_ratio:
        raise SystemExit(
            f"OEFP Morgan generation ratio {morgan_generation_ratio:.3f} exceeds "
            f"configured gate {args.generation_max_ratio:.3f}"
        )
    if atom_pair_generation_ratio > args.atom_pair_generation_max_ratio:
        raise SystemExit(
            f"OEFP Atom Pair generation ratio {atom_pair_generation_ratio:.3f} exceeds "
            f"configured gate {args.atom_pair_generation_max_ratio:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
