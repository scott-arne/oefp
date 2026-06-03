#include "oefp/stereo.h"

namespace OEFP {
namespace detail {

AtomStereoLabel PerceiveAtomStereo(
    const OEChem::OEMolBase& mol,
    const OEChem::OEAtomBase& atom) {
    const auto cip = OEChem::OEPerceiveCIPStereo(mol, &atom);
    if (cip == OEChem::OECIPAtomStereo::R) {
        return AtomStereoLabel::R;
    }
    if (cip == OEChem::OECIPAtomStereo::S) {
        return AtomStereoLabel::S;
    }
    // OpenEye reports UnspecStereo for stereogenic candidates without a
    // specified stereo label. RDKit leaves those out of chirality-aware
    // fingerprint encoding, so treat them as non-chiral here.
    return AtomStereoLabel::None;
}

BondStereoLabel PerceiveBondStereo(
    const OEChem::OEMolBase& mol,
    const OEChem::OEBondBase& bond) {
    const auto cip = OEChem::OEPerceiveCIPStereo(mol, &bond);
    if (cip == OEChem::OECIPBondStereo::E) {
        return BondStereoLabel::E;
    }
    if (cip == OEChem::OECIPBondStereo::Z) {
        return BondStereoLabel::Z;
    }
    // Unspecified E/Z candidates are not equivalent to RDKit's explicit
    // stereo annotations and should not perturb Morgan bond invariants.
    return BondStereoLabel::None;
}

std::uint32_t AtomPairChiralityBits(AtomStereoLabel label) {
    if (label == AtomStereoLabel::R) {
        return 1u;
    }
    if (label == AtomStereoLabel::S) {
        return 2u;
    }
    return 0u;
}

std::uint32_t MorganAtomChiralityValue(AtomStereoLabel label) {
    if (label == AtomStereoLabel::R) {
        return 3u;
    }
    if (label == AtomStereoLabel::S) {
        return 2u;
    }
    return 1u;
}

std::int32_t MorganDoubleBondStereoValue(BondStereoLabel label) {
    if (label == BondStereoLabel::E) {
        return 3;
    }
    if (label == BondStereoLabel::Z) {
        return 2;
    }
    if (label == BondStereoLabel::Unknown) {
        return 1;
    }
    return 0;
}

} // namespace detail
} // namespace OEFP
