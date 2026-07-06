#ifndef OEFP_RDKIT_DESCRIPTORS_H
#define OEFP_RDKIT_DESCRIPTORS_H

#include "oefp/column_request.h"
#include "oefp/compute_context.h"
#include "oefp/descriptor.h"

#include <oechem.h>

#include <memory>

namespace OEFP {

/// \brief Return the full RDKit 2D descriptor schema (217 descriptors).
///
/// The schema enumerates ``rdkit.Chem.Descriptors._descList`` in list order with
/// inferred scalar value kinds and source metadata generated from RDKit. All
/// descriptors are 2D and carry no coordinate prerequisite. RDKit is a
/// conformance oracle only; values are computed natively via OpenEye Toolkits.
///
/// \returns Shared immutable RDKit descriptor schema.
std::shared_ptr<const DescriptorSchema> RDKitDescriptorSchema();

/// \brief Generate supported RDKit-compatible descriptors for one molecule.
///
/// The row uses :cpp:func:`RDKitDescriptorSchema`. Descriptor families that have
/// not been ported remain missing.
///
/// \param mol Molecule to describe.
/// \returns Schema-backed RDKit-compatible descriptor row.
DescriptorSet MakeRDKitDescriptors(const OEChem::OEMolBase& mol);

/// \brief Compute RDKit descriptors reusing shared intermediates from ``ctx``.
///
/// \param mol Molecule to describe.
/// \param ctx Per-molecule cache of shared computation intermediates.
/// \param request Which columns of the RDKit schema to compute.
/// \returns Schema-backed RDKit-compatible descriptor row.
DescriptorSet MakeRDKitDescriptors(const OEChem::OEMolBase& mol,
                                   ComputeContext& ctx,
                                   const ColumnRequest& request);

} // namespace OEFP

#endif // OEFP_RDKIT_DESCRIPTORS_H
