"""Tests for OpenEye fingerprint interop."""

import pytest


def test_openeye_import_matches_openeye_tanimoto(aspirin_mol, ethanol_mol):
    from openeye import oegraphsim
    import oefp

    oe_a = oegraphsim.OEFingerPrint()
    oe_b = oegraphsim.OEFingerPrint()
    assert oegraphsim.OEMakeCircularFP(oe_a, aspirin_mol)
    assert oegraphsim.OEMakeCircularFP(oe_b, ethanol_mol)

    fp_a = oefp.from_openeye_fingerprint(oe_a)
    fp_b = oefp.from_openeye_fingerprint(oe_b)

    assert oefp.compare(fp_a, fp_b, oefp.Metric.tanimoto()) == pytest.approx(
        oegraphsim.OETanimoto(oe_a, oe_b)
    )


def test_openeye_export_roundtrip(ethanol_mol):
    from openeye import oegraphsim
    import oefp

    original = oegraphsim.OEFingerPrint()
    assert oegraphsim.OEMakeCircularFP(original, ethanol_mol)

    fp = oefp.from_openeye_fingerprint(original)
    exported = oefp.to_openeye_fingerprint(fp)

    assert oegraphsim.OEIsSameFPType(original, exported)
    assert oegraphsim.OETanimoto(original, exported) == pytest.approx(1.0)


def test_openeye_import_exposes_numeric_type_metadata(ethanol_mol):
    from openeye import oegraphsim
    import oefp

    original = oegraphsim.OEFingerPrint()
    assert oegraphsim.OEMakeCircularFP(original, ethanol_mol)
    assert original.GetFPTypeBase() is not None

    fp = oefp.from_openeye_fingerprint(original)
    spec = fp._native.Spec()

    assert spec.has_source_type_id
    assert spec.source_type_id == original.GetFPTypeBase().GetFPType()


def test_openeye_export_uses_numeric_type_metadata_when_string_empty(ethanol_mol):
    from openeye import oegraphsim
    import oefp

    original = oegraphsim.OEFingerPrint()
    assert oegraphsim.OEMakeCircularFP(original, ethanol_mol)

    fp = oefp.from_openeye_fingerprint(original)
    fp._native.Spec().source_type = ""

    exported = oefp.to_openeye_fingerprint(fp)

    assert oegraphsim.OEIsSameFPType(original, exported)
    assert oegraphsim.OETanimoto(original, exported) == pytest.approx(1.0)


def test_openeye_export_rejects_invalid_string_before_numeric_fallback(ethanol_mol):
    import oefp
    from openeye import oegraphsim

    original = oegraphsim.OEFingerPrint()
    assert oegraphsim.OEMakeCircularFP(original, ethanol_mol)

    fp = oefp.from_openeye_fingerprint(original)
    fp._native.Spec().source_type = "not-a-valid-openeye-fingerprint-type"

    with pytest.raises(ValueError, match="could not be resolved"):
        oefp.to_openeye_fingerprint(fp)
