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


@dataclass(frozen=True)
class MoleculeBucket:
    """OpenEye/RDKit molecule subset with a shared size label."""

    label: str
    oe_mols: list[Any]
    rd_mols: list[Any]


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


def _bucketed_molecules(oe_mols: list[Any], rd_mols: list[Any]) -> list[MoleculeBucket]:
    """Return molecule buckets for whole-set and heavy-atom range profiling."""
    if len(oe_mols) != len(rd_mols):
        raise ValueError("OpenEye and RDKit molecule lists must have matching lengths.")

    paired_mols = list(zip(oe_mols, rd_mols, strict=True))
    bucket_specs = [
        ("all", lambda heavy_atoms: True),
        ("heavy_atoms<=20", lambda heavy_atoms: heavy_atoms <= 20),
        ("heavy_atoms=21-40", lambda heavy_atoms: 21 <= heavy_atoms <= 40),
        ("heavy_atoms>40", lambda heavy_atoms: heavy_atoms > 40),
    ]
    buckets: list[MoleculeBucket] = []
    for label, predicate in bucket_specs:
        oe_bucket: list[Any] = []
        rd_bucket: list[Any] = []
        for oe_mol, rd_mol in paired_mols:
            if predicate(int(rd_mol.GetNumHeavyAtoms())):
                oe_bucket.append(oe_mol)
                rd_bucket.append(rd_mol)
        buckets.append(MoleculeBucket(label, oe_bucket, rd_bucket))
    return buckets


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


def _profile_total_seconds(profile: Any) -> float:
    return float(profile.TotalSeconds())


def _print_stage_summary(
    label: str,
    profiles: list[Any],
    stage_fields: list[tuple[str, str]],
    baseline: TimingStats,
) -> None:
    if not profiles:
        return
    stage_total = sum(_profile_total_seconds(profile) for profile in profiles)
    coverage = stage_total / baseline.median_s if baseline.median_s > 0.0 else float("inf")
    event_count = sum(int(profile.event_count) for profile in profiles)
    on_bit_count = sum(int(profile.on_bit_count) for profile in profiles)
    print(
        f"{label:<34} stage_total={stage_total:.6f}s "
        f"coverage={coverage:.3f} events={event_count} on_bits={on_bit_count}"
    )
    for stage_label, field_name in stage_fields:
        seconds = sum(float(getattr(profile, field_name)) for profile in profiles)
        share = seconds / stage_total if stage_total > 0.0 else 0.0
        print(f"  {stage_label:<28} {seconds:.6f}s share={share:.3f}")


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

    native_api = cast(Any, oefp_api)
    native_module = cast(Any, oefp_api._native)  # noqa: SLF001

    native_morgan_dense_options = native_api._morgan_options(
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
    native_morgan_count_options = native_api._morgan_options(
        2,
        2048,
        False,
        True,
        False,
        True,
        False,
        True,
        None,
    )
    py_morgan_dense = oefp.MorganGenerator(radius=2, num_bits=2048)
    py_morgan_count = oefp.MorganGenerator(radius=2, num_bits=2048, count_simulation=True)
    native_morgan_dense = native_module._NativeMorganGenerator(native_morgan_dense_options)
    native_morgan_count = native_module._NativeMorganGenerator(native_morgan_count_options)
    rd_morgan_dense = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
        useBondTypes=True,
        includeRingMembership=True,
    )
    rd_morgan_count = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        countSimulation=True,
        includeChirality=False,
        useBondTypes=True,
        includeRingMembership=True,
    )

    native_atom_pair_dense_options = native_api._atom_pair_options(
        1,
        30,
        2048,
        False,
        True,
        False,
        None,
    )
    native_atom_pair_count_options = native_api._atom_pair_options(
        1,
        30,
        2048,
        False,
        True,
        True,
        None,
    )
    py_atom_pair_dense = oefp.AtomPairGenerator(num_bits=2048, count_simulation=False)
    py_atom_pair_count = oefp.AtomPairGenerator(num_bits=2048, count_simulation=True)
    native_atom_pair_dense = native_module._NativeAtomPairGenerator(
        native_atom_pair_dense_options
    )
    native_atom_pair_count = native_module._NativeAtomPairGenerator(
        native_atom_pair_count_options
    )
    rd_atom_pair_dense = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=1,
        maxDistance=30,
        includeChirality=False,
        use2D=True,
        countSimulation=False,
        fpSize=2048,
    )
    rd_atom_pair_count = rdFingerprintGenerator.GetAtomPairGenerator(
        minDistance=1,
        maxDistance=30,
        includeChirality=False,
        use2D=True,
        countSimulation=True,
        fpSize=2048,
    )

    morgan_stage_fields = [
        ("graph", "graph_seconds"),
        ("invariant/setup", "invariant_seconds"),
        ("radius zero", "radius_zero_seconds"),
        ("neighborhood expansion", "neighborhood_seconds"),
        ("duplicate suppression", "duplicate_seconds"),
        ("bit folding", "bit_folding_seconds"),
    ]
    atom_pair_stage_fields = [
        ("molecule preparation", "molecule_preparation_seconds"),
        ("graph", "graph_seconds"),
        ("atom codes", "atom_code_seconds"),
        ("distances", "distance_seconds"),
        ("pair enumeration", "pair_enumeration_seconds"),
        ("bit folding", "bit_folding_seconds"),
    ]

    buckets = _bucketed_molecules(oe_mols, rd_mols)
    all_bucket_calls: list[tuple[str, Callable[[], Any], str | None]] = []
    for bucket in buckets:
        if not bucket.oe_mols:
            continue

        def morgan_oefp_python_dense(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [py_morgan_dense.fingerprint(mol) for mol in bucket.oe_mols]

        def morgan_oefp_native_dense(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [native_morgan_dense.Fingerprint(mol) for mol in bucket.oe_mols]

        def morgan_rdkit_dense(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [rd_morgan_dense.GetFingerprint(mol) for mol in bucket.rd_mols]

        def morgan_oefp_python_count(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [py_morgan_count.fingerprint(mol) for mol in bucket.oe_mols]

        def morgan_oefp_native_count(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [native_morgan_count.Fingerprint(mol) for mol in bucket.oe_mols]

        def morgan_rdkit_count(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [rd_morgan_count.GetFingerprint(mol) for mol in bucket.rd_mols]

        def atom_pair_oefp_python_dense(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [py_atom_pair_dense.fingerprint(mol) for mol in bucket.oe_mols]

        def atom_pair_oefp_native_dense(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [native_atom_pair_dense.Fingerprint(mol) for mol in bucket.oe_mols]

        def atom_pair_rdkit_dense(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [rd_atom_pair_dense.GetFingerprint(mol) for mol in bucket.rd_mols]

        def atom_pair_oefp_python_count(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [py_atom_pair_count.fingerprint(mol) for mol in bucket.oe_mols]

        def atom_pair_oefp_native_count(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [native_atom_pair_count.Fingerprint(mol) for mol in bucket.oe_mols]

        def atom_pair_rdkit_count(bucket: MoleculeBucket = bucket) -> list[Any]:
            return [rd_atom_pair_count.GetFingerprint(mol) for mol in bucket.rd_mols]

        calls: list[tuple[str, Callable[[], Any], str | None]] = [
            ("Morgan OEFP wrapper dense", morgan_oefp_python_dense, "Morgan RDKit dense"),
            ("Morgan OEFP native dense", morgan_oefp_native_dense, "Morgan RDKit dense"),
            ("Morgan RDKit dense", morgan_rdkit_dense, None),
            (
                "Morgan OEFP wrapper count",
                morgan_oefp_python_count,
                "Morgan RDKit count",
            ),
            (
                "Morgan OEFP native count",
                morgan_oefp_native_count,
                "Morgan RDKit count",
            ),
            ("Morgan RDKit count", morgan_rdkit_count, None),
            (
                "Atom Pair OEFP wrapper dense",
                atom_pair_oefp_python_dense,
                "Atom Pair RDKit dense",
            ),
            (
                "Atom Pair OEFP native dense",
                atom_pair_oefp_native_dense,
                "Atom Pair RDKit dense",
            ),
            ("Atom Pair RDKit dense", atom_pair_rdkit_dense, None),
            (
                "Atom Pair OEFP wrapper count",
                atom_pair_oefp_python_count,
                "Atom Pair RDKit count",
            ),
            (
                "Atom Pair OEFP native count",
                atom_pair_oefp_native_count,
                "Atom Pair RDKit count",
            ),
            ("Atom Pair RDKit count", atom_pair_rdkit_count, None),
        ]

        timings: dict[str, TimingStats] = {}
        for label, fn, _ in calls:
            timings[label] = _time_trials(
                fn,
                trials=args.trials,
                warmup=args.warmup,
                molecule_count=len(bucket.oe_mols),
            )

        print(
            f"\nbucket={bucket.label} molecules={len(bucket.oe_mols)} "
            f"trials={args.trials} warmup={args.warmup}"
        )
        for label, _, baseline_label in calls:
            baseline = timings[baseline_label] if baseline_label is not None else None
            _print_timing(label, timings[label], baseline)

        morgan_dense_profiles = [
            native_module._ProfileMorganFingerprintStages(mol, native_morgan_dense_options)
            for mol in bucket.oe_mols
        ]
        morgan_count_profiles = [
            native_module._ProfileMorganFingerprintStages(mol, native_morgan_count_options)
            for mol in bucket.oe_mols
        ]
        atom_pair_dense_profiles = [
            native_module._ProfileAtomPairFingerprintStages(
                mol,
                native_atom_pair_dense_options,
            )
            for mol in bucket.oe_mols
        ]
        atom_pair_count_profiles = [
            native_module._ProfileAtomPairFingerprintStages(
                mol,
                native_atom_pair_count_options,
            )
            for mol in bucket.oe_mols
        ]
        print("native_stage_profiles:")
        _print_stage_summary(
            "Morgan native dense stages",
            morgan_dense_profiles,
            morgan_stage_fields,
            timings["Morgan OEFP native dense"],
        )
        _print_stage_summary(
            "Morgan native count stages",
            morgan_count_profiles,
            morgan_stage_fields,
            timings["Morgan OEFP native count"],
        )
        _print_stage_summary(
            "Atom Pair native dense stages",
            atom_pair_dense_profiles,
            atom_pair_stage_fields,
            timings["Atom Pair OEFP native dense"],
        )
        _print_stage_summary(
            "Atom Pair native count stages",
            atom_pair_count_profiles,
            atom_pair_stage_fields,
            timings["Atom Pair OEFP native count"],
        )

        if bucket.label == "all":
            all_bucket_calls = calls

    pdist_n = min(400, len(oe_mols))
    if pdist_n < 2:
        raise SystemExit("Need at least two parsed molecules for pdist parity guardrail")
    morgan_fps = [py_morgan_dense.fingerprint(m) for m in oe_mols[:pdist_n]]
    rd_fps = [rd_morgan_dense.GetFingerprint(m) for m in rd_mols[:pdist_n]]
    batch = oefp.OEFPBatch.from_fingerprints(morgan_fps)
    metric = oefp.Metric.tanimoto()
    oe_pdist = np.asarray(oefp.pdist(batch, metric), dtype=float)
    rd_pdist = np.asarray(_rdkit_bulk_pdist(rd_fps, DataStructs), dtype=float)
    np.testing.assert_allclose(oe_pdist, rd_pdist, rtol=1e-7, atol=1e-7)
    print(f"pdist_guardrail_size={pdist_n} rdkit/oefp_values_match=true")

    if args.profile:
        for label, fn, _ in all_bucket_calls:
            print(f"\n--- cProfile: {label} ---")
            print(_profile_call(fn, lines=args.profile_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
