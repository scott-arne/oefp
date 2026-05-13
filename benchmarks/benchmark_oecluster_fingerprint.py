#!/usr/bin/env python3
"""Compare OEFP dense fingerprint kernels against oecluster fingerprint paths."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from openeye import oechem, oegraphsim


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_OECLUSTER_SOURCE_DIR = Path(
    os.environ.get("OEFP_OECLUSTER_SOURCE_DIR", "/Users/johnss51/Development/cpp/oecluster")
)


def _prepend_source_paths(oecluster_source_dir: Path) -> None:
    """Prefer local source-tree packages when the benchmark runs from checkout."""
    sys.path.insert(0, str(_REPO_ROOT / "python"))
    oecluster_python = oecluster_source_dir / "python"
    if oecluster_python.is_dir():
        sys.path.insert(0, str(oecluster_python))


def _seed_smiles() -> list[str]:
    return [
        "c1ccccc1",
        "c1ccc(O)cc1",
        "c1ccncc1",
        "CCCCCCCC",
        "CCO",
        "CCN",
        "CC(=O)O",
        "c1ccc(Cl)cc1",
    ]


def build_molecules(count: int) -> list[Any]:
    """Build a deterministic molecule set for repeatable benchmark inputs."""
    seeds = _seed_smiles()
    mols = []
    for index in range(count):
        mol = oechem.OEGraphMol()
        if not oechem.OESmilesToMol(mol, seeds[index % len(seeds)]):
            raise RuntimeError("OESmilesToMol failed")
        mols.append(mol)
    return mols


def time_call(fn: Callable[[], Any], repeats: int) -> tuple[float, Any]:
    """Return median runtime and the final result from repeated calls."""
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return statistics.median(values), result


def _default_atom_type(atom_type_mask: int) -> int:
    if atom_type_mask != 0:
        return atom_type_mask
    return int(oegraphsim.OEFPAtomType_DefaultCircularAtom)


def _default_bond_type(bond_type_mask: int) -> int:
    if bond_type_mask != 0:
        return bond_type_mask
    return int(oegraphsim.OEFPBondType_DefaultCircularBond)


def build_oefp_batch(
    mols: Sequence[Any],
    *,
    numbits: int,
    min_distance: int,
    max_distance: int,
    atom_type_mask: int,
    bond_type_mask: int,
) -> Any:
    """Generate OEFP fingerprints with the same settings oecluster uses."""
    import oefp

    atom_type = _default_atom_type(atom_type_mask)
    bond_type = _default_bond_type(bond_type_mask)
    fps = []
    for mol in mols:
        oe_fp = oegraphsim.OEFingerPrint()
        if not oegraphsim.OEMakeCircularFP(
            oe_fp,
            mol,
            numbits,
            min_distance,
            max_distance,
            atom_type,
            bond_type,
        ):
            raise RuntimeError("OEMakeCircularFP failed")
        fps.append(oefp.from_openeye_fingerprint(oe_fp))
    return oefp.OEFPBatch.from_fingerprints(fps)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--numbits", type=int, default=2048)
    parser.add_argument("--min-distance", type=int, default=0)
    parser.add_argument("--max-distance", type=int, default=2)
    parser.add_argument("--atom-type-mask", type=int, default=0)
    parser.add_argument("--bond-type-mask", type=int, default=0)
    parser.add_argument("--max-ratio", type=float, default=1.05)
    parser.add_argument(
        "--oecluster-source-dir",
        type=Path,
        default=_DEFAULT_OECLUSTER_SOURCE_DIR,
    )
    return parser


def main() -> int:
    """Run the benchmark and fail if OEFP misses the configured ratio gate."""
    args = _parser().parse_args()
    _prepend_source_paths(args.oecluster_source_dir)

    import oecluster
    import oefp

    mols = build_molecules(args.count)
    batch = build_oefp_batch(
        mols,
        numbits=args.numbits,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        atom_type_mask=args.atom_type_mask,
        bond_type_mask=args.bond_type_mask,
    )
    metric = oefp.Metric.jaccard()

    oe_time, oe_dist = time_call(
        lambda: oecluster.pdist(
            mols,
            "fingerprint",
            similarity=False,
            num_threads=args.threads,
            chunk_size=args.chunk_size,
            fp_type="circular",
            numbits=args.numbits,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            atom_type_mask=args.atom_type_mask,
            bond_type_mask=args.bond_type_mask,
            similarity_func="tanimoto",
        ),
        args.repeats,
    )
    fp_time, fp_dist = time_call(
        lambda: oefp.pdist(
            batch,
            metric,
            num_threads=args.threads,
            chunk_size=args.chunk_size,
        ),
        args.repeats,
    )

    oe_values = np.asarray(oe_dist)
    fp_values = np.asarray(fp_dist)
    np.testing.assert_allclose(fp_values, oe_values, rtol=1e-7, atol=1e-7)

    ratio = fp_time / oe_time if oe_time > 0 else float("inf")
    print(f"oecluster pdist median: {oe_time:.6f}s")
    print(f"oefp pdist median:      {fp_time:.6f}s")
    print(f"oefp/oecluster ratio:   {ratio:.3f}")

    if ratio > args.max_ratio:
        raise SystemExit(
            f"OEFP pdist ratio {ratio:.3f} exceeds configured gate {args.max_ratio:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
