#ifndef OEFP_MORDRED_H
#define OEFP_MORDRED_H

#include "oefp/descriptor.h"

#include <oechem.h>

#include <memory>

namespace OEFP {

/// \brief Return the full Mordred descriptor schema.
///
/// The schema enumerates Mordred 1.2.0 descriptors in calculator order with
/// inferred scalar value kinds and source metadata generated from the local
/// Mordred descriptor definitions.
///
/// \returns Shared immutable Mordred descriptor schema.
std::shared_ptr<const DescriptorSchema> MordredDescriptorSchema();

/// \brief Generate the supported Mordred-compatible count descriptor subset.
///
/// The initial subset covers integer AtomCount, Aromatic, and BondCount-style
/// descriptors that fit OEFP's sparse count-key descriptor model. Zero-valued
/// descriptors are omitted from storage; consumers should treat missing
/// supported keys as zero.
///
/// \param mol Molecule to describe.
/// \returns Counted string-key Mordred-compatible descriptors.
DescriptorSet MakeMordredDescriptors(const OEChem::OEMolBase& mol);

} // namespace OEFP

#endif // OEFP_MORDRED_H
