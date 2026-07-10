"""Conformance tests for kallisto atom/bond descriptors.

Verifies fixture structure and compares C++ kallisto atom descriptor results
against the kallisto 1.0.10 oracle for the panel molecules.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

# OEFP binding must be importable (fail-closed, not skip-on-missing)
from oefp import kallisto_atom_descriptors, kallisto_atom_schema

# OpenEye is optional (skip tests if missing)
try:
    from openeye import oechem
    HAS_OPENEYE = True
except ImportError:
    HAS_OPENEYE = False

# Expected formal charge sums for the panel molecules
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


def test_kallisto_fixture_structure() -> None:
    """Verify the kallisto fixture loads and has the expected structure."""
    fixture_path = Path("tests/python/kallisto_references.json")
    packaged_path = Path("python/oefp/kallisto_references.json")

    # Both copies must exist
    assert fixture_path.exists(), f"Fixture not found: {fixture_path}"
    assert packaged_path.exists(), f"Packaged fixture not found: {packaged_path}"

    # Both copies must be byte-identical
    assert fixture_path.read_bytes() == packaged_path.read_bytes(), \
        "Fixture and packaged copy are not byte-identical"

    # Load the fixture
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Required top-level keys
    assert "kallisto_version" in fixture
    assert "atom_schema" in fixture
    assert "bond_schema" in fixture
    assert "tiers" in fixture
    assert "molecules" in fixture

    # Check version
    assert fixture["kallisto_version"] == "1.0.10"

    # Atom schema: 8 descriptors (3 CN types, prox, eeq, alp, 2 vdW types)
    atom_schema = fixture["atom_schema"]
    assert len(atom_schema) == 8
    expected_atom_cols = {"cn_erf", "cn_cov", "cn_exp", "prox", "eeq", "alp", "vdw_rahm", "vdw_truhlar"}
    assert {d["name"] for d in atom_schema} == expected_atom_cols

    # Bond schema: 3 Sterimol descriptors
    bond_schema = fixture["bond_schema"]
    assert len(bond_schema) == 3
    expected_bond_cols = {"sterimol_L", "sterimol_B1", "sterimol_B5"}
    assert {d["name"] for d in bond_schema} == expected_bond_cols

    # Tiers: every atom+bond column must have a tier
    tiers = fixture["tiers"]
    for col in expected_atom_cols:
        assert col in tiers, f"Missing tier for atom column {col}"
        assert tiers[col] in {"exact", "tight", "loose"}
    for col in expected_bond_cols:
        assert col in tiers, f"Missing tier for bond column {col}"
        assert tiers[col] in {"exact", "tight", "loose"}

    # Molecules
    molecules = fixture["molecules"]
    assert len(molecules) > 0, "Fixture has no molecules"

    for mol in molecules:
        # Required molecule keys
        assert "id" in mol
        assert "charge" in mol
        assert "coords_angstrom" in mol
        assert "atomic_numbers" in mol
        assert "atom_values" in mol
        assert "bond_rows" in mol

        # Verify formal charge sum
        mol_id = mol["id"]
        actual_charge = mol["charge"]
        expected_charge = EXPECTED_CHARGES.get(mol_id)
        assert expected_charge is not None, f"Unknown molecule {mol_id}"
        assert actual_charge == expected_charge, \
            f"Charge mismatch for {mol_id}: expected {expected_charge}, got {actual_charge}"

        atom_count = len(mol["atomic_numbers"])
        assert atom_count > 0

        # Coordinates: one per atom
        assert len(mol["coords_angstrom"]) == atom_count
        for coord in mol["coords_angstrom"]:
            assert len(coord) == 3  # [x, y, z]

        # Atom values: every column must have atom_count entries
        atom_values = mol["atom_values"]
        for col in expected_atom_cols:
            assert col in atom_values, f"Missing atom column {col} in molecule {mol['id']}"
            assert len(atom_values[col]) == atom_count, \
                f"Column {col} has {len(atom_values[col])} values, expected {atom_count}"

        # Bond rows: each row has origin, partner, and 3 Sterimol values
        for i, bond_row in enumerate(mol["bond_rows"]):
            assert "origin" in bond_row, f"Bond row {i} missing origin"
            assert "partner" in bond_row, f"Bond row {i} missing partner"
            assert "sterimol_L" in bond_row
            assert "sterimol_B1" in bond_row
            assert "sterimol_B5" in bond_row
            # origin and partner must be valid atom indices
            assert 0 <= bond_row["origin"] < atom_count
            assert 0 <= bond_row["partner"] < atom_count

    print(f"✓ Fixture structure valid: {len(molecules)} molecules, "
          f"{sum(len(m['atomic_numbers']) for m in molecules)} atoms, "
          f"{sum(len(m['bond_rows']) for m in molecules)} bond rows.")


def test_packaged_kallisto_reference_loads() -> None:
    """Verify the packaged kallisto reference is readable via importlib.resources.

    This test verifies the package-data declaration in pyproject.toml is correct,
    ensuring kallisto_references.json ships in wheels/installs.
    """
    # Read the packaged copy (not the test fixture)
    try:
        fixture = resources.files("oefp").joinpath("kallisto_references.json")
    except ImportError:
        # If oefp can't be imported due to missing library dependencies, verify the
        # JSON file is at least present in the source tree (pyproject.toml declares it).
        source_json = Path("python/oefp/kallisto_references.json")
        assert source_json.exists(), f"Source kallisto_references.json not found: {source_json}"

        # Parse it to verify it's well-formed
        packaged_data = json.loads(source_json.read_text(encoding="utf-8"))
        assert "kallisto_version" in packaged_data
        assert packaged_data["kallisto_version"] == "1.0.10"
        return  # Skip the importlib.resources check if oefp isn't loadable

    assert fixture.is_file(), "Packaged kallisto_references.json not found via importlib.resources"

    # Load and parse the packaged fixture
    with fixture.open(encoding="utf-8") as handle:
        packaged_data = json.load(handle)

    # Verify it has the expected top-level structure
    assert "kallisto_version" in packaged_data
    assert "atom_schema" in packaged_data
    assert "bond_schema" in packaged_data
    assert "tiers" in packaged_data
    assert "molecules" in packaged_data
    assert packaged_data["kallisto_version"] == "1.0.10"


# Tolerance tiers from kallisto_references.json
TIERS = {
    "exact": 1e-8,
    "tight": 1e-4,
    "loose": 1e-2,
}


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_atom_schema() -> None:
    """Verify the kallisto atom schema has expected columns."""
    schema = kallisto_atom_schema()
    # Schema loaded from fixture includes all columns (8 for full kallisto)
    assert len(schema.names) >= 5  # At least through Task 6 (eeq)
    assert schema.names[:5] == ("cn_erf", "cn_cov", "cn_exp", "prox", "eeq")
    for defn in schema.definitions:
        assert defn.group == "kallisto"
        assert defn.source_name == "kallisto"
        assert defn.source_type == "geometric"
        assert defn.source_version == "kallisto-1.0.10"


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_atom_descriptors_conformance() -> None:
    """Compare kallisto atom descriptor results against kallisto 1.0.10 oracle."""
    fixture_path = Path("tests/python/kallisto_references.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    molecules = fixture["molecules"]
    tiers = fixture["tiers"]
    sdf_base = Path("tests/data/kallisto_panel")

    # Columns to test (Task 6 scope: cn_erf, cn_cov, cn_exp, prox, eeq)
    test_columns = ["cn_erf", "cn_cov", "cn_exp", "prox", "eeq"]
    max_deviations = {col: 0.0 for col in test_columns}

    for mol_data in molecules:
        mol_id = mol_data["id"]
        sdf_path = sdf_base / f"{mol_id}.sdf"
        assert sdf_path.exists(), f"SDF not found for {mol_id}: {sdf_path}"

        # Load molecule from SDF
        ifs = oechem.oemolistream(str(sdf_path))
        mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, mol)

        # Compute kallisto atom descriptors
        charge = mol_data.get("charge")
        result = kallisto_atom_descriptors(mol, charge=charge)

        atom_count = len(mol_data["atomic_numbers"])
        assert result.atom_count == atom_count, \
            f"Atom count mismatch for {mol_id}: expected {atom_count}, got {result.atom_count}"

        # Compare each column against fixture
        for col_name in test_columns:
            expected_vals = np.array(mol_data["atom_values"][col_name], dtype=np.float64)
            actual_vals = getattr(result, col_name)
            tier = tiers[col_name]
            tol = TIERS[tier]

            # All values should be valid
            validity = result.validity[col_name]
            assert np.all(validity), \
                f"{mol_id} {col_name}: expected all valid, got {np.sum(~validity)} invalid"

            # Check values match within tier tolerance
            deviation = np.abs(actual_vals - expected_vals)
            max_dev = np.max(deviation)
            max_deviations[col_name] = max(max_deviations[col_name], max_dev)

            assert np.allclose(actual_vals, expected_vals, atol=tol, rtol=0.0), \
                f"{mol_id} {col_name}: max deviation {max_dev:.2e} exceeds tier {tier} tolerance {tol:.2e}"

    print(f"\n✓ Kallisto atom descriptor conformance: {len(molecules)} molecules")
    for col_name in test_columns:
        tier = tiers[col_name]
        print(f"  {col_name}: max deviation {max_deviations[col_name]:.2e} (tier {tier})")


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_atom_descriptors_ineligible() -> None:
    """Verify ineligible molecules return empty results."""
    # Test 1: 2D molecule (no 3D coords)
    mol_2d = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol_2d, "CCO")
    result_2d = kallisto_atom_descriptors(mol_2d)
    assert result_2d.atom_count == 0
    assert len(result_2d["cn_erf"]) == 0
    assert len(result_2d["cn_cov"]) == 0
    assert len(result_2d["cn_exp"]) == 0
    assert len(result_2d["prox"]) == 0
    assert len(result_2d["eeq"]) == 0

    # Test 2: Molecule with Z > 86 (francium Z=87)
    mol_heavy = oechem.OEGraphMol()
    fr = mol_heavy.NewAtom(87)  # Francium, Z=87 > 86
    mol_heavy.SetDimension(3)
    coords = [0.0, 0.0, 0.0]
    mol_heavy.SetCoords(fr, coords)

    result_heavy = kallisto_atom_descriptors(mol_heavy)
    assert result_heavy.atom_count == 0
    assert len(result_heavy["cn_erf"]) == 0
    assert len(result_heavy["cn_cov"]) == 0
    assert len(result_heavy["cn_exp"]) == 0
    assert len(result_heavy["prox"]) == 0
    assert len(result_heavy["eeq"]) == 0


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_atom_descriptors_noncontiguous_indices() -> None:
    """Verify atom_indices preserves OpenEye GetIdx after DeleteAtom.

    Regression test: after deleting a middle atom, the C++ side preserves each
    atom's actual OpenEye GetIdx (e.g. [0,1,3,4]), and Python exposes it so
    callers don't silently get 0..N-1 labels when atom IDs are non-contiguous.
    """
    # Build a 5-atom molecule with simple 3D coordinates
    mol = oechem.OEGraphMol()
    mol.SetDimension(3)

    # Create 5 carbon atoms in a line
    atoms = []
    for i in range(5):
        atom = mol.NewAtom(6)  # Carbon
        atoms.append(atom)
        # Simple linear coordinates
        mol.SetCoords(atom, [float(i), 0.0, 0.0])

    # Add bonds between consecutive atoms
    for i in range(4):
        mol.NewBond(atoms[i], atoms[i + 1])

    # Collect all atom indices before deletion
    all_atoms_before = list(mol.GetAtoms())
    assert len(all_atoms_before) == 5

    # Delete the 3rd atom (index 2) → GetIdx becomes [0,1,3,4] for the 4 remaining
    atom_to_delete = atoms[2]
    mol.DeleteAtom(atom_to_delete)

    # Re-collect atoms (should be 4 now)
    all_atoms_after = list(mol.GetAtoms())
    assert len(all_atoms_after) == 4

    # Expected OpenEye GetIdx sequence (non-contiguous: [0,1,3,4])
    expected_indices = np.array([a.GetIdx() for a in all_atoms_after], dtype=np.uint32)
    assert expected_indices[0] == 0
    assert expected_indices[1] == 1
    assert expected_indices[2] == 3  # NOT 2 (that atom was deleted)
    assert expected_indices[3] == 4

    # Compute kallisto atom descriptors
    result = kallisto_atom_descriptors(mol)

    # Verify atom_count matches
    assert result.atom_count == 4

    # Verify atom_indices has the same length
    assert len(result.atom_indices) == 4

    # The returned atom_indices should be [0,1,3,4], NOT [0,1,2,3]
    np.testing.assert_array_equal(
        result.atom_indices,
        expected_indices,
        err_msg="atom_indices should preserve OpenEye GetIdx [0,1,3,4], not renumber to [0,1,2,3]"
    )


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_eeq_overlapping_coordinates() -> None:
    """Verify EEQ emits missing values (not NaN) for degenerate geometry.

    When two atoms occupy identical coordinates (r=0), the EEQ matrix assembly
    contains NaN terms (erf(0)/0 = NaN). The native code must detect this and
    emit the eeq column as all-missing rather than valid-looking NaN charges.
    """
    # Build a molecule with two carbon atoms at IDENTICAL 3D coordinates
    mol = oechem.OEGraphMol()
    mol.SetDimension(3)

    # Create two carbon atoms at the SAME position
    atom1 = mol.NewAtom(6)  # Carbon
    atom2 = mol.NewAtom(6)  # Carbon
    identical_coords = [0.0, 0.0, 0.0]
    mol.SetCoords(atom1, identical_coords)
    mol.SetCoords(atom2, identical_coords)

    # Compute kallisto atom descriptors
    result = kallisto_atom_descriptors(mol)

    # Should have 2 atoms
    assert result.atom_count == 2

    # CN/prox may still have values (they don't break on r=0 the same way),
    # but EEQ must be marked as all-missing
    eeq_validity = result.validity["eeq"]
    assert len(eeq_validity) == 2
    assert not np.any(eeq_validity), \
        "EEQ should be all-missing for overlapping coordinates, not valid NaN"

    # Verify no present value is NaN (all values should be marked invalid)
    eeq_vals = result["eeq"]
    for i in range(2):
        if eeq_validity[i]:
            # If somehow marked valid, verify it's not NaN
            assert np.isfinite(eeq_vals[i]), \
                f"EEQ value {i} is marked valid but contains NaN/inf"
