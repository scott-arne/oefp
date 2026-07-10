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
from oefp import (
    kallisto_atom_descriptors,
    kallisto_atom_descriptors_batch,
    kallisto_atom_schema,
    kallisto_bond_descriptors,
    kallisto_bond_descriptors_batch,
    kallisto_bond_schema,
    sterimol,
)

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
    # Schema loaded from fixture includes all columns (8 for complete atom schema)
    assert len(schema.names) >= 8  # At least through Task 8 (vdw_truhlar)
    assert schema.names[:8] == ("cn_erf", "cn_cov", "cn_exp", "prox", "eeq", "alp", "vdw_rahm", "vdw_truhlar")
    for defn in schema.definitions:
        assert defn.group == "kallisto"
        assert defn.source_name == "kallisto"
        assert defn.source_type == "geometric"
        assert defn.source_version == "kallisto-1.0.10"


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_atom_schema_metadata_consistency() -> None:
    """Verify Python schema metadata uses canonical units spelling.

    The Python kallisto_atom_schema() derives column names from the native
    KallistoAtomDescriptorSchema() and metadata from kallisto_references.json.
    The native C++ schema is the single source of truth for units spelling.
    This test iterates all kallisto atom columns and asserts each column's units
    match the expected canonical spelling (Bohr with capital B, etc.), preventing
    native-vs-fixture units divergence.
    """
    python_schema = kallisto_atom_schema()

    # Expected canonical units spelling for each kallisto atom column
    expected_units = {
        "cn_erf": "",
        "cn_cov": "",
        "cn_exp": "",
        "prox": "",
        "eeq": "e",
        "alp": "Bohr^3",
        "vdw_rahm": "Bohr",
        "vdw_truhlar": "Bohr",
    }

    for col_name, expected_unit in expected_units.items():
        col_idx = python_schema.names.index(col_name)
        col_defn = python_schema.definitions[col_idx]
        assert col_defn.units == expected_unit, \
            f"{col_name} units should be '{expected_unit}' (canonical spelling), got '{col_defn.units}'"


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_atom_descriptors_conformance() -> None:
    """Compare kallisto atom descriptor results against kallisto 1.0.10 oracle."""
    fixture_path = Path("tests/python/kallisto_references.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    molecules = fixture["molecules"]
    tiers = fixture["tiers"]
    sdf_base = Path("tests/data/kallisto_panel")

    # Columns to test (Task 8 scope: all 8 atom columns)
    test_columns = ["cn_erf", "cn_cov", "cn_exp", "prox", "eeq", "alp", "vdw_rahm", "vdw_truhlar"]
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
    assert len(result_2d["alp"]) == 0
    assert len(result_2d["vdw_rahm"]) == 0
    assert len(result_2d["vdw_truhlar"]) == 0

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
    assert len(result_heavy["alp"]) == 0
    assert len(result_heavy["vdw_rahm"]) == 0
    assert len(result_heavy["vdw_truhlar"]) == 0


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


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_bond_schema() -> None:
    """Verify the kallisto bond schema has expected columns."""
    schema = kallisto_bond_schema()
    # Schema loaded from fixture includes 3 bond columns (Sterimol L/B1/B5)
    assert len(schema.names) == 3
    assert schema.names == ("sterimol_L", "sterimol_B1", "sterimol_B5")
    for defn in schema.definitions:
        assert defn.group == "kallisto"
        assert defn.source_name == "kallisto"
        assert defn.source_type == "geometric"
        assert defn.source_version == "kallisto-1.0.10"
        assert defn.units == "Bohr"


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_bond_descriptors_conformance() -> None:
    """Compare kallisto bond descriptor results against kallisto 1.0.10 oracle."""
    fixture_path = Path("tests/python/kallisto_references.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    molecules = fixture["molecules"]
    tiers = fixture["tiers"]
    sdf_base = Path("tests/data/kallisto_panel")

    # Columns to test (all 3 bond columns)
    test_columns = ["sterimol_L", "sterimol_B1", "sterimol_B5"]
    max_deviations = {col: 0.0 for col in test_columns}

    for mol_data in molecules:
        mol_id = mol_data["id"]
        sdf_path = sdf_base / f"{mol_id}.sdf"
        assert sdf_path.exists(), f"SDF not found for {mol_id}: {sdf_path}"

        # Load molecule from SDF
        ifs = oechem.oemolistream(str(sdf_path))
        mol = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, mol)

        # Compute kallisto bond descriptors
        result = kallisto_bond_descriptors(mol)

        # Build a lookup by (begin, end) for comparison
        bond_map = {}
        for i in range(result.bond_count):
            begin = int(result.begin[i])
            end = int(result.end[i])
            bond_map[(begin, end)] = {
                col: result[col][i] for col in test_columns
            }

        # Check that the emitted bond set matches the fixture bond_rows
        fixture_bonds = mol_data["bond_rows"]
        fixture_bond_keys = {(row["origin"], row["partner"]) for row in fixture_bonds}
        result_bond_keys = set(bond_map.keys())

        # The result should emit exactly the same (origin, partner) set as the fixture
        assert result_bond_keys == fixture_bond_keys, \
            f"{mol_id}: emitted bond set {result_bond_keys} != fixture {fixture_bond_keys}"

        # Compare each bond row against fixture
        for bond_row in fixture_bonds:
            origin = bond_row["origin"]
            partner = bond_row["partner"]
            key = (origin, partner)

            assert key in bond_map, \
                f"{mol_id}: missing bond ({origin}, {partner}) in result"

            result_vals = bond_map[key]

            for col_name in test_columns:
                expected_val = bond_row[col_name]
                actual_val = result_vals[col_name]
                tier = tiers[col_name]
                tol = TIERS[tier]

                deviation = abs(actual_val - expected_val)
                max_deviations[col_name] = max(max_deviations[col_name], deviation)

                assert abs(actual_val - expected_val) <= tol, \
                    f"{mol_id} bond ({origin},{partner}) {col_name}: " \
                    f"deviation {deviation:.2e} exceeds tier {tier} tolerance {tol:.2e}"

    print(f"\n✓ Kallisto bond descriptor conformance: {len(molecules)} molecules")
    for col_name in test_columns:
        tier = tiers[col_name]
        print(f"  {col_name}: max deviation {max_deviations[col_name]:.2e} (tier {tier})")


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_bond_descriptors_ineligible() -> None:
    """Verify ineligible molecules return empty bond results."""
    # Test 1: 2D molecule (no 3D coords)
    mol_2d = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol_2d, "CCO")
    result_2d = kallisto_bond_descriptors(mol_2d)
    assert result_2d.bond_count == 0
    assert len(result_2d["sterimol_L"]) == 0
    assert len(result_2d["sterimol_B1"]) == 0
    assert len(result_2d["sterimol_B5"]) == 0

    # Test 2: Molecule with Z > 86
    mol_heavy = oechem.OEGraphMol()
    fr = mol_heavy.NewAtom(87)  # Francium, Z=87 > 86
    mol_heavy.SetDimension(3)
    coords = [0.0, 0.0, 0.0]
    mol_heavy.SetCoords(fr, coords)

    result_heavy = kallisto_bond_descriptors(mol_heavy)
    assert result_heavy.bond_count == 0
    assert len(result_heavy["sterimol_L"]) == 0
    assert len(result_heavy["sterimol_B1"]) == 0
    assert len(result_heavy["sterimol_B5"]) == 0


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_sterimol_function() -> None:
    """Verify the sterimol function matches bond descriptor results."""
    fixture_path = Path("tests/python/kallisto_references.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

    # Test on the first molecule with bonds
    mol_data = fixture["molecules"][0]
    mol_id = mol_data["id"]
    sdf_path = Path("tests/data/kallisto_panel") / f"{mol_id}.sdf"

    # Load molecule from SDF
    ifs = oechem.oemolistream(str(sdf_path))
    mol = oechem.OEGraphMol()
    oechem.OEReadMolecule(ifs, mol)

    # Get a bond from fixture
    if len(mol_data["bond_rows"]) > 0:
        bond_row = mol_data["bond_rows"][0]
        origin = bond_row["origin"]
        partner = bond_row["partner"]

        # Compute via sterimol function
        result = sterimol(mol, origin, partner)
        assert result is not None, f"sterimol({origin}, {partner}) returned None"

        # Compare against fixture
        tier = fixture["tiers"]["sterimol_L"]
        tol = TIERS[tier]

        assert abs(result.L - bond_row["sterimol_L"]) <= tol
        assert abs(result.B1 - bond_row["sterimol_B1"]) <= tol
        assert abs(result.B5 - bond_row["sterimol_B5"]) <= tol

    # Test ineligible call
    result_bad = sterimol(mol, 999, 1000)
    assert result_bad is None


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_atom_descriptors_batch() -> None:
    """Verify atom descriptor batch with valid, skipped, valid molecules."""
    # Create 3 molecules: valid 3D, skipped (2D), valid 3D
    mol_valid_1 = oechem.OEGraphMol()
    mol_valid_1.SetDimension(3)
    # Simple 3-atom chain
    atoms_1 = []
    for i in range(3):
        atom = mol_valid_1.NewAtom(6)  # Carbon
        mol_valid_1.SetCoords(atom, [float(i), 0.0, 0.0])
        atoms_1.append(atom)
    mol_valid_1.NewBond(atoms_1[0], atoms_1[1])
    mol_valid_1.NewBond(atoms_1[1], atoms_1[2])

    mol_skipped = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol_skipped, "CCN")  # 2D, no coords

    mol_valid_2 = oechem.OEGraphMol()
    mol_valid_2.SetDimension(3)
    # Simple 2-atom chain
    atoms_2 = []
    for i in range(2):
        atom = mol_valid_2.NewAtom(6)  # Carbon
        mol_valid_2.SetCoords(atom, [float(i), 0.0, 0.0])
        atoms_2.append(atom)
    mol_valid_2.NewBond(atoms_2[0], atoms_2[1])

    # Compute batch
    mols = [mol_valid_1, mol_skipped, mol_valid_2]
    batch = kallisto_atom_descriptors_batch(mols)

    # Check batch size
    assert len(batch) == 3
    assert batch.Size() == 3

    # Segment 0: valid molecule → should match single-molecule result
    result_single_0 = kallisto_atom_descriptors(mol_valid_1)
    segment_0 = batch[0]

    assert len(segment_0["cn_erf"]) == result_single_0.atom_count
    assert len(segment_0["atom_indices"]) == result_single_0.atom_count
    np.testing.assert_array_equal(segment_0["atom_indices"], result_single_0.atom_indices)

    # Check all columns match
    column_names = ["cn_erf", "cn_cov", "cn_exp", "prox", "eeq", "alp", "vdw_rahm", "vdw_truhlar"]
    for col in column_names:
        np.testing.assert_array_equal(
            segment_0[col], result_single_0[col],
            err_msg=f"Segment 0 {col} should match single-molecule result"
        )
        # Check validity matches
        np.testing.assert_array_equal(
            segment_0["validity"][col], result_single_0.validity[col],
            err_msg=f"Segment 0 validity[{col}] should match single-molecule result"
        )

    # Segment 1: skipped molecule → empty arrays
    segment_1 = batch[1]
    assert len(segment_1["cn_erf"]) == 0
    assert len(segment_1["atom_indices"]) == 0
    for col in column_names:
        assert len(segment_1[col]) == 0
        assert len(segment_1["validity"][col]) == 0

    # Segment 2: valid molecule → should match single-molecule result
    result_single_2 = kallisto_atom_descriptors(mol_valid_2)
    segment_2 = batch[2]

    assert len(segment_2["cn_erf"]) == result_single_2.atom_count
    assert len(segment_2["atom_indices"]) == result_single_2.atom_count
    np.testing.assert_array_equal(segment_2["atom_indices"], result_single_2.atom_indices)

    for col in column_names:
        np.testing.assert_array_equal(
            segment_2[col], result_single_2[col],
            err_msg=f"Segment 2 {col} should match single-molecule result"
        )
        np.testing.assert_array_equal(
            segment_2["validity"][col], result_single_2.validity[col],
            err_msg=f"Segment 2 validity[{col}] should match single-molecule result"
        )


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_bond_descriptors_batch() -> None:
    """Verify bond descriptor batch with valid, skipped, valid molecules."""
    # Create 3 molecules: valid 3D, skipped (2D), valid 3D
    mol_valid_1 = oechem.OEGraphMol()
    mol_valid_1.SetDimension(3)
    # Simple 3-atom chain
    atoms_1 = []
    for i in range(3):
        atom = mol_valid_1.NewAtom(6)  # Carbon
        mol_valid_1.SetCoords(atom, [float(i), 0.0, 0.0])
        atoms_1.append(atom)
    mol_valid_1.NewBond(atoms_1[0], atoms_1[1])
    mol_valid_1.NewBond(atoms_1[1], atoms_1[2])

    mol_skipped = oechem.OEGraphMol()
    oechem.OESmilesToMol(mol_skipped, "CCN")  # 2D, no coords

    mol_valid_2 = oechem.OEGraphMol()
    mol_valid_2.SetDimension(3)
    # Simple 2-atom chain
    atoms_2 = []
    for i in range(2):
        atom = mol_valid_2.NewAtom(6)  # Carbon
        mol_valid_2.SetCoords(atom, [float(i), 0.0, 0.0])
        atoms_2.append(atom)
    mol_valid_2.NewBond(atoms_2[0], atoms_2[1])

    # Compute batch
    mols = [mol_valid_1, mol_skipped, mol_valid_2]
    batch = kallisto_bond_descriptors_batch(mols)

    # Check batch size
    assert len(batch) == 3
    assert batch.Size() == 3

    # Segment 0: valid molecule → should match single-molecule result
    result_single_0 = kallisto_bond_descriptors(mol_valid_1)
    segment_0 = batch[0]

    assert len(segment_0["sterimol_L"]) == result_single_0.bond_count
    assert len(segment_0["begin"]) == result_single_0.bond_count
    assert len(segment_0["end"]) == result_single_0.bond_count
    np.testing.assert_array_equal(segment_0["begin"], result_single_0.begin)
    np.testing.assert_array_equal(segment_0["end"], result_single_0.end)

    # Check all columns match
    column_names = ["sterimol_L", "sterimol_B1", "sterimol_B5"]
    for col in column_names:
        np.testing.assert_array_equal(
            segment_0[col], result_single_0[col],
            err_msg=f"Segment 0 {col} should match single-molecule result"
        )
        np.testing.assert_array_equal(
            segment_0["validity"][col], result_single_0.validity[col],
            err_msg=f"Segment 0 validity[{col}] should match single-molecule result"
        )

    # Segment 1: skipped molecule → empty arrays
    segment_1 = batch[1]
    assert len(segment_1["sterimol_L"]) == 0
    assert len(segment_1["begin"]) == 0
    assert len(segment_1["end"]) == 0
    for col in column_names:
        assert len(segment_1[col]) == 0
        assert len(segment_1["validity"][col]) == 0

    # Segment 2: valid molecule → should match single-molecule result
    result_single_2 = kallisto_bond_descriptors(mol_valid_2)
    segment_2 = batch[2]

    assert len(segment_2["sterimol_L"]) == result_single_2.bond_count
    assert len(segment_2["begin"]) == result_single_2.bond_count
    assert len(segment_2["end"]) == result_single_2.bond_count
    np.testing.assert_array_equal(segment_2["begin"], result_single_2.begin)
    np.testing.assert_array_equal(segment_2["end"], result_single_2.end)

    for col in column_names:
        np.testing.assert_array_equal(
            segment_2[col], result_single_2[col],
            err_msg=f"Segment 2 {col} should match single-molecule result"
        )
        np.testing.assert_array_equal(
            segment_2["validity"][col], result_single_2.validity[col],
            err_msg=f"Segment 2 validity[{col}] should match single-molecule result"
        )


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_batch_threaded_smoke() -> None:
    """Verify batch functions run without error (GIL-released smoke test)."""
    # Create a few valid 3D molecules
    mols = []
    for n_atoms in [3, 2, 4, 5]:
        mol = oechem.OEGraphMol()
        mol.SetDimension(3)
        atoms = []
        for i in range(n_atoms):
            atom = mol.NewAtom(6)  # Carbon
            mol.SetCoords(atom, [float(i), 0.0, 0.0])
            atoms.append(atom)
        for i in range(n_atoms - 1):
            mol.NewBond(atoms[i], atoms[i + 1])
        mols.append(mol)

    # Call batch functions (GIL is released in the native trampoline)
    atom_batch = kallisto_atom_descriptors_batch(mols)
    assert len(atom_batch) == 4

    bond_batch = kallisto_bond_descriptors_batch(mols)
    assert len(bond_batch) == 4

    # Verify all segments are non-empty (all molecules are valid 3D)
    for i in range(4):
        atom_seg = atom_batch[i]
        assert len(atom_seg["cn_erf"]) > 0

        bond_seg = bond_batch[i]
        # Sterimol may be empty if no acyclic bonds, but the segment should exist
        assert "sterimol_L" in bond_seg


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_atom_descriptors_batch_charge_override() -> None:
    """Verify atom descriptor batch charge override matches single-molecule charge override."""
    # Create a simple 3D molecule
    mol = oechem.OEGraphMol()
    mol.SetDimension(3)
    atoms = []
    for i in range(3):
        atom = mol.NewAtom(6)  # Carbon
        mol.SetCoords(atom, [float(i), 0.0, 0.0])
        atoms.append(atom)
    mol.NewBond(atoms[0], atoms[1])
    mol.NewBond(atoms[1], atoms[2])

    # Compute with charge override for single molecule
    result_single = kallisto_atom_descriptors(mol, charge=0)

    # Compute with charge override for batch
    batch = kallisto_atom_descriptors_batch([mol], charge=0)
    assert len(batch) == 1
    segment = batch[0]

    # Both should produce identical results
    column_names = ["cn_erf", "cn_cov", "cn_exp", "prox", "eeq", "alp", "vdw_rahm", "vdw_truhlar"]
    for col in column_names:
        np.testing.assert_array_equal(
            segment[col], result_single[col],
            err_msg=f"Batch charge override {col} should match single-molecule charge override"
        )
        np.testing.assert_array_equal(
            segment["validity"][col], result_single.validity[col],
            err_msg=f"Batch charge override validity[{col}] should match single-molecule"
        )


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_sterimol_namedtuple_unpacking() -> None:
    """Verify Sterimol result is a NamedTuple supporting tuple unpacking."""
    # Create a simple 3D molecule with a bond
    mol = oechem.OEGraphMol()
    mol.SetDimension(3)
    atoms = []
    for i in range(2):
        atom = mol.NewAtom(6)  # Carbon
        mol.SetCoords(atom, [float(i), 0.0, 0.0])
        atoms.append(atom)
    mol.NewBond(atoms[0], atoms[1])

    # Compute sterimol
    result = sterimol(mol, 0, 1)
    assert result is not None

    # Test tuple unpacking
    L, B1, B5 = result
    assert isinstance(L, float)
    assert isinstance(B1, float)
    assert isinstance(B5, float)

    # Test attribute access
    assert result.L == L
    assert result.B1 == B1
    assert result.B5 == B5

    # Verify fixture row also matches (use first bond from fixture)
    fixture_path = Path("tests/python/kallisto_references.json")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    mol_data = fixture["molecules"][0]

    if len(mol_data["bond_rows"]) > 0:
        bond_row = mol_data["bond_rows"][0]
        mol_id = mol_data["id"]
        sdf_path = Path("tests/data/kallisto_panel") / f"{mol_id}.sdf"

        ifs = oechem.oemolistream(str(sdf_path))
        mol_fixture = oechem.OEGraphMol()
        oechem.OEReadMolecule(ifs, mol_fixture)

        origin = bond_row["origin"]
        partner = bond_row["partner"]
        result_fixture = sterimol(mol_fixture, origin, partner)
        assert result_fixture is not None

        # Test unpacking with fixture values
        L_fix, B1_fix, B5_fix = result_fixture

        # Verify unpacked values match attributes
        assert L_fix == result_fixture.L
        assert B1_fix == result_fixture.B1
        assert B5_fix == result_fixture.B5

        # Verify against fixture (use tight tier tolerance)
        tier = fixture["tiers"]["sterimol_L"]
        tol = TIERS[tier]
        assert abs(L_fix - bond_row["sterimol_L"]) <= tol
        assert abs(B1_fix - bond_row["sterimol_B1"]) <= tol
        assert abs(B5_fix - bond_row["sterimol_B5"]) <= tol


@pytest.mark.skipif(not HAS_OPENEYE, reason="OpenEye not available")
def test_kallisto_source_classes_not_public() -> None:
    """Verify KallistoAtomDescriptorSource and KallistoBondDescriptorSource are not public."""
    import oefp

    # These should NOT be in the public API
    assert "KallistoAtomDescriptorSource" not in dir(oefp)
    assert "KallistoBondDescriptorSource" not in dir(oefp)

    # These SHOULD be in the public API (convenience/batch functions)
    assert "kallisto_atom_descriptors" in dir(oefp)
    assert "kallisto_atom_descriptors_batch" in dir(oefp)
    assert "kallisto_bond_descriptors" in dir(oefp)
    assert "kallisto_bond_descriptors_batch" in dir(oefp)
    assert "sterimol" in dir(oefp)
    assert "kallisto_atom_schema" in dir(oefp)
    assert "kallisto_bond_schema" in dir(oefp)

    # Result classes should be public
    assert "KallistoAtomDescriptors" in dir(oefp)
    assert "KallistoAtomDescriptorsBatch" in dir(oefp)
    assert "KallistoBondDescriptors" in dir(oefp)
    assert "KallistoBondDescriptorsBatch" in dir(oefp)
    assert "Sterimol" in dir(oefp)
