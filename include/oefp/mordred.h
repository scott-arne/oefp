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
/// Mordred descriptor definitions. Definitions whose local Mordred source
/// declares ``require_3D`` carry
/// ``kDescriptorPrerequisiteCoordinates3D``.
///
/// \returns Shared immutable Mordred descriptor schema.
std::shared_ptr<const DescriptorSchema> MordredDescriptorSchema();

/// \brief Generate supported Mordred-compatible schema-backed descriptors.
///
/// The row uses :cpp:func:`MordredDescriptorSchema` and sets implemented
/// descriptor values in that full schema. Descriptor families that have not
/// been ported remain missing. Descriptors whose prerequisites are not
/// available on the input molecule also remain missing.
///
/// Mordred descriptor calculation does not generate 2D or 3D coordinates
/// implicitly. Callers that want generated conformers should perform that
/// preparation explicitly before calling this function.
///
/// \param mol Molecule to describe.
/// \returns Schema-backed Mordred-compatible descriptor row.
DescriptorSet MakeMordredDescriptors(const OEChem::OEMolBase& mol);

} // namespace OEFP

#endif // OEFP_MORDRED_H
