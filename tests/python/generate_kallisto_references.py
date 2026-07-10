"""Generate the kallisto atom/bond descriptor conformance fixture.

kallisto is a test-time conformance oracle only. This script reads the fixed-
coordinate SDF panel (built from kallisto's test geometries), uses OpenEye as
the graph/charge/bond authority, and computes kallisto descriptors on the same
coordinates (converted Å→Bohr) to emit reference values for C++ conformance.

Acyclic bond rule (reproduced exactly in C++):
  1. Call OEFindRingAtomsAndBonds(mol) to mark ring bonds.
  2. Enumerate all bonds where not bond.IsInRing().
  3. For each acyclic bond, emit BOTH directions: (origin, partner) pairs
     where origin=bond.GetBgn().GetIdx(), partner=bond.GetEnd().GetIdx()
     AND (partner, origin) as a second row.

Charge rule:
  Total charge = sum of atom.GetFormalCharge() over all atoms (the exact rule
  OEFP uses). This is the `charge` parameter passed to kallisto.get_eeq,
  get_alp, and get_vdw.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

EXPECTED_KALLISTO_VERSION = "1.0.10"

# Provisional tolerance tiers (subject to C++ deviation measurement).
# cn_erf/cn_cov/cn_exp/prox are tight; eeq/alp/vdw_rahm/vdw_truhlar are loose;
# Sterimol L/B1/B5 are loose (geometry-dependent, solvent model approximations).
KALLISTO_TOLERANCE_TIERS: dict[str, str] = {
    # Atom descriptors
    "cn_erf": "tight",
    "cn_cov": "tight",
    "cn_exp": "tight",
    "prox": "tight",
    "eeq": "tight",
    "alp": "tight",
    "vdw_rahm": "tight",
    "vdw_truhlar": "tight",
    # Bond descriptors (Sterimol L, B1, B5)
    "sterimol_L": "loose",
    "sterimol_B1": "loose",
    "sterimol_B5": "loose",
}


def _apply_proxy() -> None:
    for key, value in {
        "HTTP_PROXY": "http://proxy-server.bms.com:8080",
        "HTTPS_PROXY": "http://proxy-server.bms.com:8080",
    }.items():
        os.environ.setdefault(key, value)


def _kallisto_version() -> str:
    import kallisto
    return str(kallisto.__version__)


def _json_value(value: Any) -> Any:
    """Convert a value to JSON-serializable form (nonfinite → error dict)."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    number = float(value)
    if not math.isfinite(number):
        return {"error_type": str(number), "state": "nonfinite"}
    return number


def _reference_payload(panel_dir: Path) -> dict[str, Any]:
    from kallisto.molecule import Molecule
    from kallisto.sterics import getClassicalSterimol
    from kallisto.units import Bohr
    from openeye import oechem

    version = _kallisto_version()
    if version != EXPECTED_KALLISTO_VERSION:
        raise RuntimeError(
            f"Expected kallisto {EXPECTED_KALLISTO_VERSION}, imported {version!r}."
        )

    sdf_files = sorted(panel_dir.glob("*.sdf"))
    if not sdf_files:
        raise RuntimeError(
            f"No SDF files found in {panel_dir}. Run build_kallisto_panel.py first."
        )

    # Atom schema (8 descriptors: 3 CN types, prox, eeq, alp, 2 vdW types)
    atom_schema = [
        {"name": "cn_erf", "units": "", "description": "Coordination number (erf)"},
        {"name": "cn_cov", "units": "", "description": "Coordination number (cov)"},
        {"name": "cn_exp", "units": "", "description": "Coordination number (exp)"},
        {"name": "prox", "units": "", "description": "Proximity shell (2,3)"},
        {"name": "eeq", "units": "e", "description": "EEQ partial charge"},
        {"name": "alp", "units": "Bohr^3", "description": "Atomic polarizability"},
        {"name": "vdw_rahm", "units": "Bohr", "description": "vdW radius (Rahm)"},
        {"name": "vdw_truhlar", "units": "Bohr", "description": "vdW radius (Truhlar)"},
    ]

    # Bond schema (3 Sterimol descriptors)
    bond_schema = [
        {"name": "sterimol_L", "units": "Bohr", "description": "Sterimol L (length)"},
        {"name": "sterimol_B1", "units": "Bohr", "description": "Sterimol B1 (min width)"},
        {"name": "sterimol_B5", "units": "Bohr", "description": "Sterimol B5 (max width)"},
    ]

    molecules = []

    for sdf_path in sdf_files:
        # Read the SDF via OpenEye (graph/charge/bond authority)
        ifs = oechem.oemolistream(str(sdf_path))
        mol = oechem.OEGraphMol()
        if not oechem.OEReadMolecule(ifs, mol):
            raise RuntimeError(f"Could not read {sdf_path}")
        ifs.close()

        mol_id = mol.GetTitle()

        # Extract atomic numbers and coordinates (Å) via OpenEye
        atomic_numbers = []
        coords_angstrom = []
        coord_dict = mol.GetCoords()  # {idx: (x, y, z)}

        for atom in mol.GetAtoms():
            atomic_numbers.append(atom.GetAtomicNum())
            idx = atom.GetIdx()
            x, y, z = coord_dict[idx]
            coords_angstrom.append([float(x), float(y), float(z)])

        # Total charge (sum of formal charges)
        total_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

        # Convert coordinates to Bohr for kallisto
        coords_bohr = [[x / Bohr, y / Bohr, z / Bohr] for x, y, z in coords_angstrom]

        # Build kallisto Molecule directly from Z + coords-in-Bohr
        kallisto_mol = Molecule(numbers=atomic_numbers, positions=coords_bohr)

        # Compute atom descriptors
        cn_erf = kallisto_mol.get_cns("erf", threshold=800.0)
        cn_cov = kallisto_mol.get_cns("cov", threshold=800.0)
        cn_exp = kallisto_mol.get_cns("exp", threshold=800.0)
        prox = kallisto_mol.get_prox((2, 3), threshold=800.0)
        eeq = kallisto_mol.get_eeq(total_charge)
        alp = kallisto_mol.get_alp(total_charge)
        vdw_rahm = kallisto_mol.get_vdw(total_charge, "rahm", 1)
        vdw_truhlar = kallisto_mol.get_vdw(total_charge, "truhlar", 1)

        # Pack atom values (per-column lists)
        atom_values = {
            "cn_erf": [_json_value(v) for v in cn_erf],
            "cn_cov": [_json_value(v) for v in cn_cov],
            "cn_exp": [_json_value(v) for v in cn_exp],
            "prox": [_json_value(v) for v in prox],
            "eeq": [_json_value(v) for v in eeq],
            "alp": [_json_value(v) for v in alp],
            "vdw_rahm": [_json_value(v) for v in vdw_rahm],
            "vdw_truhlar": [_json_value(v) for v in vdw_truhlar],
        }

        # Identify acyclic bonds (both directions)
        oechem.OEFindRingAtomsAndBonds(mol)
        bond_rows = []
        for bond in mol.GetBonds():
            if not bond.IsInRing():
                # Emit both directions
                origin_idx = bond.GetBgn().GetIdx()
                partner_idx = bond.GetEnd().GetIdx()

                # Direction 1: origin -> partner
                L, B1, B5 = getClassicalSterimol(kallisto_mol, origin_idx, partner_idx)
                bond_rows.append({
                    "origin": origin_idx,
                    "partner": partner_idx,
                    "sterimol_L": _json_value(L),
                    "sterimol_B1": _json_value(B1),
                    "sterimol_B5": _json_value(B5),
                })

                # Direction 2: partner -> origin
                L2, B1_2, B5_2 = getClassicalSterimol(kallisto_mol, partner_idx, origin_idx)
                bond_rows.append({
                    "origin": partner_idx,
                    "partner": origin_idx,
                    "sterimol_L": _json_value(L2),
                    "sterimol_B1": _json_value(B1_2),
                    "sterimol_B5": _json_value(B5_2),
                })

        molecules.append({
            "id": mol_id,
            "charge": total_charge,
            "coords_angstrom": coords_angstrom,
            "atomic_numbers": atomic_numbers,
            "atom_values": atom_values,
            "bond_rows": bond_rows,
        })

    return {
        "kallisto_version": version,
        "atom_schema": atom_schema,
        "bond_schema": bond_schema,
        "tiers": KALLISTO_TOLERANCE_TIERS,
        "molecules": molecules,
    }


def main() -> None:
    _apply_proxy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON file path (tests/python/kallisto_references.json)")
    parser.add_argument("--panel-dir", type=Path,
                        help="Panel SDF directory (default: repo-relative tests/data/kallisto_panel)")
    args = parser.parse_args()

    # Repo-relative panel dir if not specified
    if args.panel_dir:
        panel_dir = args.panel_dir
    else:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent.parent
        panel_dir = repo_root / "tests/data/kallisto_panel"

    payload = _reference_payload(panel_dir)

    # Write the fixture
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"kallisto reference fixture written to {args.output}")

    # Copy to python/oefp/ (byte-identical)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    package_copy = repo_root / "python/oefp" / args.output.name
    package_copy.write_bytes(args.output.read_bytes())
    print(f"Copied to {package_copy} (byte-identical)")

    # Report stats
    print(f"Molecules: {len(payload['molecules'])}")
    print(f"Atom descriptors: {len(payload['atom_schema'])}")
    print(f"Bond descriptors: {len(payload['bond_schema'])}")
    total_atoms = sum(len(m["atomic_numbers"]) for m in payload["molecules"])
    total_bonds = sum(len(m["bond_rows"]) for m in payload["molecules"])
    print(f"Total atoms: {total_atoms}, total bond rows: {total_bonds}")


if __name__ == "__main__":
    main()
