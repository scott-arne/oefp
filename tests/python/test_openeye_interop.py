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
