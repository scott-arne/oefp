#ifndef OEFP_MORDRED_H
#define OEFP_MORDRED_H

#include "oefp/column_request.h"
#include "oefp/compute_context.h"
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

/// \brief Compute Mordred descriptors reusing shared intermediates from ``ctx``.
///
/// Behaves exactly like the single-argument overload but pulls the heavy-atom
/// graph, heavy-atom distance matrix, Gasteiger charges, and Crippen
/// contributions from ``ctx`` so those intermediates are memoized and shared
/// across descriptor sources rather than recomputed here.
///
/// \note ``ctx`` must have been constructed from the same molecule passed as
///     ``mol``; the two are consumed as one molecule.
/// \note In this phase ``request`` is accepted but all columns are computed;
///     column pruning is introduced in a later phase.
///
/// \param mol Molecule to describe.
/// \param ctx Per-molecule cache of shared computation intermediates.
/// \param request Which columns of the Mordred schema to compute.
/// \returns Schema-backed Mordred-compatible descriptor row.
DescriptorSet MakeMordredDescriptors(const OEChem::OEMolBase& mol,
                                     ComputeContext& ctx,
                                     const ColumnRequest& request);

} // namespace OEFP

#endif // OEFP_MORDRED_H
