#!/usr/bin/env python3
"""Profile OEFP reusable generation against RDKit generator calls."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class TimingStats:
    """Repeated timing summary."""

    median_s: float
    iqr_s: float
    min_s: float
    max_s: float
    trials: int
    molecule_count: int

    @property
    def per_mol_us(self) -> float:
        """Return median microseconds per molecule."""
        return self.median_s / self.molecule_count * 1_000_000.0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles",
        type=Path,
        default=Path("/Users/johnss51/Development/cpp/rdkit/Data/NCI/first_5K.smi"),
    )
    parser.add_argument("--max-mols", type=int, default=500)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-lines", type=int, default=25)
    return parser


def _load_molecules(path: Path, max_mols: int) -> tuple[list[Any], list[Any]]:
    from openeye import oechem  # type: ignore[import-not-found]
    from rdkit import Chem  # type: ignore[import-not-found]

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
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    iqr = 0.0
    if len(samples) >= 2:
        quartiles = statistics.quantiles(samples, n=4, method="inclusive")
        iqr = quartiles[2] - quartiles[0]
    return TimingStats(
        median_s=statistics.median(samples),
        iqr_s=iqr,
        min_s=min(samples),
        max_s=max(samples),
        trials=trials,
        molecule_count=molecule_count,
    )


def _profile_call(fn: Callable[[], Any], *, lines: int) -> str:
    profiler = cProfile.Profile()
    profiler.enable()
    fn()
    profiler.disable()
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(lines)
    return stream.getvalue()


def _print_timing(label: str, stats: TimingStats, baseline: TimingStats | None = None) -> None:
    ratio = ""
    if baseline is not None:
        ratio = f" ratio={stats.median_s / baseline.median_s:.3f}"
    print(
        f"{label:<34} median={stats.median_s:.6f}s "
        f"iqr={stats.iqr_s:.6f}s per_mol={stats.per_mol_us:.3f}us{ratio}"
    )


def _rdkit_bulk_pdist(rd_fps: list[Any], data_structs: Any) -> list[float]:
    output: list[float] = []
    for index, fp in enumerate(rd_fps):
        output.extend(data_structs.BulkTanimotoSimilarity(fp, rd_fps[index + 1 :]))
    return output


def main() -> int:
    args = _parser().parse_args()
    if args.max_mols <= 0:
        raise SystemExit("--max-mols must be greater than zero")
    if args.trials <= 0:
        raise SystemExit("--trials must be greater than zero")
    if args.warmup < 0:
        raise SystemExit("--warmup must be non-negative")

    from rdkit import DataStructs  # type: ignore[import-not-found]
    from rdkit.Chem import rdFingerprintGenerator  # type: ignore[import-not-found]

    import oefp  # type: ignore[import-not-found]
    from oefp import api as oefp_api  # type: ignore[import-not-found]

    oe_mols, rd_mols = _load_molecules(args.smiles, args.max_mols)

    py_morgan = oefp.MorganGenerator(radius=2, num_bits=2048)
    native_api = cast(Any, oefp_api)
    native_module = cast(Any, oefp_api._native)  # noqa: SLF001

    native_morgan_options = native_api._morgan_options(
        2,
        2048,
        False,
        True,
        False,
        True,
        False,
        False,
        None,
    )
    native_morgan = native_module._NativeMorganGenerator(native_morgan_options)
    rd_morgan = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
        useBondTypes=True,
        includeRingMembership=True,
    )

    py_atom_pair = oefp.AtomPairGenerator(num_bits=2048)
    native_atom_pair_options = native_api._atom_pair_options(
        1,
        30,
        2048,
        False,
        True,
        True,
        None,
    )
    native_atom_pair = native_module._NativeAtomPairGenerator(native_atom_pair_options)
    rd_atom_pair = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=1,
        maxDistance=30,
        includeChirality=False,
        use2D=True,
        countSimulation=True,
        fpSize=2048,
    )

    def morgan_oefp_python() -> list[Any]:
        return [py_morgan.fingerprint(mol) for mol in oe_mols]

    def morgan_oefp_native() -> list[Any]:
        return [native_morgan.Fingerprint(mol) for mol in oe_mols]

    def morgan_oefp_cached() -> list[Any]:
        return [oefp.morgan_fingerprint(mol) for mol in oe_mols]

    def morgan_rdkit() -> list[Any]:
        return [rd_morgan.GetFingerprint(mol) for mol in rd_mols]

    def atom_pair_oefp_python() -> list[Any]:
        return [py_atom_pair.fingerprint(mol) for mol in oe_mols]

    def atom_pair_oefp_native() -> list[Any]:
        return [native_atom_pair.Fingerprint(mol) for mol in oe_mols]

    def atom_pair_oefp_cached() -> list[Any]:
        return [oefp.atom_pair_fingerprint(mol) for mol in oe_mols]

    def atom_pair_rdkit() -> list[Any]:
        return [rd_atom_pair.GetFingerprint(mol) for mol in rd_mols]

    timings: dict[str, TimingStats] = {}
    calls: list[tuple[str, Callable[[], Any], str | None]] = [
        ("Morgan OEFP Python wrapper", morgan_oefp_python, "Morgan RDKit"),
        ("Morgan OEFP direct native", morgan_oefp_native, "Morgan RDKit"),
        ("Morgan OEFP cached function", morgan_oefp_cached, "Morgan RDKit"),
        ("Morgan RDKit", morgan_rdkit, None),
        (
            "Atom Pair OEFP Python wrapper",
            atom_pair_oefp_python,
            "Atom Pair RDKit",
        ),
        (
            "Atom Pair OEFP direct native",
            atom_pair_oefp_native,
            "Atom Pair RDKit",
        ),
        (
            "Atom Pair OEFP cached function",
            atom_pair_oefp_cached,
            "Atom Pair RDKit",
        ),
        ("Atom Pair RDKit", atom_pair_rdkit, None),
    ]

    for label, fn, _ in calls:
        timings[label] = _time_trials(
            fn,
            trials=args.trials,
            warmup=args.warmup,
            molecule_count=len(oe_mols),
        )

    print(f"molecules={len(oe_mols)} trials={args.trials} warmup={args.warmup}")
    for label, _, baseline_label in calls:
        baseline = timings[baseline_label] if baseline_label is not None else None
        _print_timing(label, timings[label], baseline)

    pdist_n = min(400, len(oe_mols))
    if pdist_n < 2:
        raise SystemExit("Need at least two parsed molecules for pdist parity guardrail")
    morgan_fps = [py_morgan.fingerprint(m) for m in oe_mols[:pdist_n]]
    rd_fps = [rd_morgan.GetFingerprint(m) for m in rd_mols[:pdist_n]]
    batch = oefp.OEFPBatch.from_fingerprints(morgan_fps)
    metric = oefp.Metric.tanimoto()
    oe_pdist = np.asarray(oefp.pdist(batch, metric), dtype=float)
    rd_pdist = np.asarray(_rdkit_bulk_pdist(rd_fps, DataStructs), dtype=float)
    np.testing.assert_allclose(oe_pdist, rd_pdist, rtol=1e-7, atol=1e-7)
    print(f"pdist_guardrail_size={pdist_n} rdkit/oefp_values_match=true")

    if args.profile:
        for label, fn, _ in calls:
            print(f"\n--- cProfile: {label} ---")
            print(_profile_call(fn, lines=args.profile_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
