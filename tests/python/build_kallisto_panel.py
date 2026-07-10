"""Build the kallisto conformance SDF panel from kallisto's test geometries.

This script reads kallisto's own xyz geometries, perceives bonds/charges via
OpenEye, and emits SDF files with authoritative graph+charge (the future C++ side
reads the same SDFs via OpenEye and must see identical bonds/charge).

Molecule selection:
- toluene, pyridine, Me, Et, alanine-glycine: from kallisto's test geometries.
- acetate: charged species (formal charge sum != 0), built from SMILES + Omega.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Expected formal charge sums for the panel molecules (correctness requirement)
EXPECTED_CHARGES = {
    "toluene": 0,
    "pyridine": 0,
    "methane": 0,
    "ethane": 0,
    "alanine-glycine": 0,
    "acetate": -1,
    "methanol": 0,
    "methanethiol": 0,
}


def _apply_proxy() -> None:
    for key, value in {
        "HTTP_PROXY": "http://proxy-server.bms.com:8080",
        "HTTPS_PROXY": "http://proxy-server.bms.com:8080",
    }.items():
        os.environ.setdefault(key, value)


def _xyz_to_sdf(xyz_path: Path, output_path: Path, title: str) -> None:
    """Read an xyz file, perceive bonds+charges via OpenEye, write SDF.

    DEPRECATED: xyz-based builds can produce incorrect charges due to aromatic
    perception failures. Prefer _smiles_to_sdf for aromatics.
    """
    from openeye import oechem

    # Read the xyz file (element + coords)
    lines = xyz_path.read_text(encoding="utf-8").strip().split("\n")
    atom_count = int(lines[0].strip())
    mol = oechem.OEGraphMol()

    # Parse atoms and coordinates
    coords = []
    for i, line in enumerate(lines[2 : 2 + atom_count], start=0):
        parts = line.split()
        symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        coords.append((x, y, z))

        atom = mol.NewAtom(oechem.OEGetAtomicNum(symbol))
        if atom is None:
            raise ValueError(f"Could not create atom for symbol {symbol}")

    # Set coordinates
    mol.SetDimension(3)
    for atom, (x, y, z) in zip(mol.GetAtoms(), coords):
        mol.SetCoords(atom, oechem.OEFloatArray([x, y, z]))

    # Perceive bonds and charges
    oechem.OEDetermineConnectivity(mol)
    oechem.OEPerceiveBondOrders(mol)
    oechem.OEAssignFormalCharges(mol)

    mol.SetTitle(title)

    # Verify formal charge sum
    actual_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    expected_charge = EXPECTED_CHARGES.get(title)
    if expected_charge is not None and actual_charge != expected_charge:
        raise ValueError(
            f"Formal charge mismatch for {title}: expected {expected_charge}, "
            f"got {actual_charge}. Rebuild from SMILES."
        )

    # Write SDF
    ofs = oechem.oemolostream(str(output_path))
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def _smiles_to_sdf(smiles: str, output_path: Path, title: str) -> None:
    """Build a 3D conformer from SMILES via Omega, write SDF."""
    from openeye import oechem, oeomega

    mol = oechem.OEMol()
    if not oechem.OESmilesToMol(mol, smiles):
        raise ValueError(f"Could not parse SMILES: {smiles}")

    # Generate a single 3D conformer
    omega = oeomega.OEOmega()
    omega.SetMaxConfs(1)
    omega.SetIncludeInput(False)
    omega.SetStrictStereo(False)
    if not omega(mol):
        raise RuntimeError(f"Omega failed to generate conformer for {smiles}")

    mol.SetTitle(title)

    # Verify formal charge sum
    actual_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    expected_charge = EXPECTED_CHARGES.get(title)
    if expected_charge is not None and actual_charge != expected_charge:
        raise ValueError(
            f"Formal charge mismatch for {title}: expected {expected_charge}, "
            f"got {actual_charge}"
        )

    # Write SDF
    ofs = oechem.oemolostream(str(output_path))
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build kallisto SDF panel from kallisto test geometries.")
    parser.add_argument(
        "--kallisto-structures",
        type=Path,
        default=None,
        help="Path to kallisto/tests/structures directory (or set KALLISTO_STRUCTURES env var)"
    )
    args = parser.parse_args()

    # Resolve kallisto structures path: CLI arg > env var > fail
    kallisto_structures = args.kallisto_structures or os.environ.get("KALLISTO_STRUCTURES")
    if not kallisto_structures:
        print("Error: kallisto structures directory not specified.", file=sys.stderr)
        print("Provide via --kallisto-structures or set KALLISTO_STRUCTURES environment variable.", file=sys.stderr)
        sys.exit(1)

    kallisto_structures = Path(kallisto_structures)
    if not kallisto_structures.is_dir():
        print(f"Error: kallisto structures directory does not exist: {kallisto_structures}", file=sys.stderr)
        sys.exit(1)

    # Validate required xyz files exist before proceeding
    required_files = ["Me.xyz", "Et.xyz", "alanine-glycine.xyz"]
    missing_files = [f for f in required_files if not (kallisto_structures / f).exists()]
    if missing_files:
        print(f"Error: missing required xyz files in {kallisto_structures}:", file=sys.stderr)
        for f in missing_files:
            print(f"  - {f}", file=sys.stderr)
        sys.exit(1)

    _apply_proxy()

    # Repo-relative panel dir
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    panel_dir = repo_root / "tests/data/kallisto_panel"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Aromatics: SMILES builds for correct neutral charge + 3D conformer
    _smiles_to_sdf("Cc1ccccc1", panel_dir / "toluene.sdf", "toluene")
    _smiles_to_sdf("c1ccncc1", panel_dir / "pyridine.sdf", "pyridine")

    # Non-aromatics from kallisto xyz (charge-safe)
    _xyz_to_sdf(kallisto_structures / "Me.xyz", panel_dir / "methane.sdf", "methane")
    _xyz_to_sdf(kallisto_structures / "Et.xyz", panel_dir / "ethane.sdf", "ethane")
    _xyz_to_sdf(kallisto_structures / "alanine-glycine.xyz", panel_dir / "alanine-glycine.sdf", "alanine-glycine")

    # Charged species (formal charge != 0): acetate anion
    _smiles_to_sdf("CC(=O)[O-]", panel_dir / "acetate.sdf", "acetate")

    # Additional element diversity: methanol (O), methanethiol (S)
    _smiles_to_sdf("CO", panel_dir / "methanol.sdf", "methanol")
    _smiles_to_sdf("CS", panel_dir / "methanethiol.sdf", "methanethiol")

    print(f"Panel SDFs written to {panel_dir}")


if __name__ == "__main__":
    main()
