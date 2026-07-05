"""Robustness test for SureChEMBL sweep: null/empty/non-string SMILES."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def test_sweep_skips_invalid_smiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Null, empty, non-string, and unparseable SMILES are skipped, not fatal."""
    # Build tiny synthetic parquet with mixed valid/invalid SMILES
    smiles_values = [
        "CC",  # valid
        None,  # null
        "",  # empty
        "   ",  # whitespace
        "not-valid-@@@",  # unparseable
        "CCO",  # valid
    ]
    table = pa.table({"smiles": smiles_values})
    parquet_path = tmp_path / "test_compounds.parquet"
    pq.write_table(table, parquet_path)

    # Point the sweep at the synthetic file
    monkeypatch.setenv("OEFP_SURECHEMBL_PARQUET", str(parquet_path))
    monkeypatch.setenv("OEFP_SURECHEMBL_SAMPLE", "100")

    # Import and run the sweep test function
    import sys
    from pathlib import Path

    sweep_path = Path(__file__).parent / "surechembl_identity_sweep.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("surechembl_identity_sweep", sweep_path)
    sweep_module = importlib.util.module_from_spec(spec)
    sys.modules["surechembl_identity_sweep"] = sweep_module
    spec.loader.exec_module(sweep_module)

    # Should NOT raise TypeError or other exception
    sweep_module.test_surechembl_identity_sweep()


def test_sweep_fails_when_all_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Sweep should FAIL if no valid molecules are processed (false-green guard)."""
    # Build synthetic parquet with ONLY invalid SMILES
    smiles_values = [
        None,  # null
        "",  # empty
        "   ",  # whitespace
        "not-valid-@@@",  # unparseable
    ]
    table = pa.table({"smiles": smiles_values})
    parquet_path = tmp_path / "test_all_invalid.parquet"
    pq.write_table(table, parquet_path)

    # Point the sweep at the all-invalid file
    monkeypatch.setenv("OEFP_SURECHEMBL_PARQUET", str(parquet_path))
    monkeypatch.setenv("OEFP_SURECHEMBL_SAMPLE", "10")

    # Import and run the sweep test function
    import sys
    from pathlib import Path

    sweep_path = Path(__file__).parent / "surechembl_identity_sweep.py"
    import importlib.util

    spec = importlib.util.spec_from_file_location("surechembl_identity_sweep_all_invalid", sweep_path)
    sweep_module = importlib.util.module_from_spec(spec)
    sys.modules["surechembl_identity_sweep_all_invalid"] = sweep_module
    spec.loader.exec_module(sweep_module)

    # Should raise AssertionError (sample-floor not met)
    with pytest.raises(AssertionError, match="no valid molecules"):
        sweep_module.test_surechembl_identity_sweep()
