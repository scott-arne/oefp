#pragma once

#include <cstdint>

#include <oechem.h>

namespace OEFP {
namespace detail {

enum class AtomStereoLabel : std::uint8_t {
    None,
    S,
    R,
    Unknown,
};

enum class BondStereoLabel : std::uint8_t {
    None,
    E,
    Z,
    Unknown,
};

AtomStereoLabel PerceiveAtomStereo(
    const OEChem::OEMolBase& mol,
    const OEChem::OEAtomBase& atom);

BondStereoLabel PerceiveBondStereo(
    const OEChem::OEMolBase& mol,
    const OEChem::OEBondBase& bond);

std::uint32_t AtomPairChiralityBits(AtomStereoLabel label);

std::uint32_t MorganAtomChiralityValue(AtomStereoLabel label);

std::int32_t MorganDoubleBondStereoValue(BondStereoLabel label);

} // namespace detail
} // namespace OEFP
