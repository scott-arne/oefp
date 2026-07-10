"""Build the kallisto conformance SDF panel from kallisto's test geometries.

This script reads kallisto's own xyz geometries, perceives bonds/charges via
OpenEye, and emits SDF files with authoritative graph+charge (the future C++ side
reads the same SDFs via OpenEye and must see identical bonds/charge).

Molecule selection:
- toluene, pyridine, Me, Et, alanine-glycine: from kallisto's test geometries.
- acetate: charged species (formal charge sum != 0), built from SMILES + Omega.
"""

from __future__ import annotations

import os
from pathlib import Path

KALLISTO_STRUCTURES = Path("/Users/johnss51/Development/python/kallisto/tests/structures")


def _apply_proxy() -> None:
    for key, value in {
        "HTTP_PROXY": "http://proxy-server.bms.com:8080",
        "HTTPS_PROXY": "http://proxy-server.bms.com:8080",
    }.items():
        os.environ.setdefault(key, value)


def _xyz_to_sdf(xyz_path: Path, output_path: Path, title: str) -> None:
    """Read an xyz file, perceive bonds+charges via OpenEye, write SDF."""
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

    # Write SDF
    ofs = oechem.oemolostream(str(output_path))
    oechem.OEWriteMolecule(ofs, mol)
    ofs.close()


def main() -> None:
    _apply_proxy()

    panel_dir = Path("/Users/johnss51/Development/cpp/oefp/tests/data/kallisto_panel")
    panel_dir.mkdir(parents=True, exist_ok=True)

    # From kallisto's test geometries (all neutral, varied elements)
    _xyz_to_sdf(KALLISTO_STRUCTURES / "toluene.xyz", panel_dir / "toluene.sdf", "toluene")
    _xyz_to_sdf(KALLISTO_STRUCTURES / "pyridine.xyz", panel_dir / "pyridine.sdf", "pyridine")
    _xyz_to_sdf(KALLISTO_STRUCTURES / "Me.xyz", panel_dir / "methane.sdf", "methane")
    _xyz_to_sdf(KALLISTO_STRUCTURES / "Et.xyz", panel_dir / "ethane.sdf", "ethane")
    _xyz_to_sdf(KALLISTO_STRUCTURES / "alanine-glycine.xyz", panel_dir / "alanine-glycine.sdf", "alanine-glycine")

    # Charged species (formal charge != 0): acetate anion
    _smiles_to_sdf("CC(=O)[O-]", panel_dir / "acetate.sdf", "acetate")

    # Additional element diversity: methanol (O), methylamine (N), methanethiol (S)
    _xyz_to_sdf_with_fallback = lambda xyz, sdf, title: (
        _xyz_to_sdf(KALLISTO_STRUCTURES / xyz, panel_dir / sdf, title)
        if (KALLISTO_STRUCTURES / xyz).exists()
        else _smiles_to_sdf(_smiles_fallback[title], panel_dir / sdf, title)
    )

    # Fallback SMILES if kallisto doesn't have these geometries
    _smiles_fallback = {
        "methanol": "CO",
        "methanethiol": "CS",
    }

    # Use SMILES for these (kallisto doesn't have small O/S examples in structures/)
    _smiles_to_sdf("CO", panel_dir / "methanol.sdf", "methanol")
    _smiles_to_sdf("CS", panel_dir / "methanethiol.sdf", "methanethiol")

    print(f"Panel SDFs written to {panel_dir}")


if __name__ == "__main__":
    main()
