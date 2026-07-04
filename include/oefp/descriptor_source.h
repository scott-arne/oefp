#ifndef OEFP_DESCRIPTOR_SOURCE_H
#define OEFP_DESCRIPTOR_SOURCE_H

#include "oefp/descriptor.h"
#include "oefp/descriptor_schema.h"

#include <oechem.h>

#include <memory>

namespace OEFP {

/// \brief Abstract producer of schema-backed descriptor rows.
///
/// A descriptor source pairs an immutable descriptor schema with a computation
/// that fills a row of that schema for a single molecule. Concrete sources wrap
/// a particular descriptor family (for example Mordred-compatible descriptors).
class DescriptorSource {
public:
    virtual ~DescriptorSource() = default;

    /// \brief Return the immutable schema describing computed rows.
    ///
    /// \returns Shared immutable descriptor schema produced by this source.
    virtual std::shared_ptr<const DescriptorSchema> Schema() const = 0;

    /// \brief Compute a schema-backed descriptor row for one molecule.
    ///
    /// This method must be safe to call concurrently on distinct molecules with
    /// read-only input; implementations must not mutate the input molecule or
    /// shared state.
    ///
    /// \param mol Molecule to describe.
    /// \returns Schema-backed descriptor row for the molecule.
    virtual DescriptorSet Compute(const OEChem::OEMolBase& mol) const = 0;
};

/// \brief Descriptor source producing Mordred-compatible descriptors.
class MordredDescriptorSource : public DescriptorSource {
public:
    std::shared_ptr<const DescriptorSchema> Schema() const override;
    DescriptorSet Compute(const OEChem::OEMolBase& mol) const override;
};

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_SOURCE_H
