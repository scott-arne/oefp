#ifndef OEFP_DESCRIPTOR_SOURCE_H
#define OEFP_DESCRIPTOR_SOURCE_H

#include "oefp/column_request.h"
#include "oefp/compute_context.h"
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
///
/// \note A subclass that overrides any \c Compute overload hides the base's
///     other \c Compute overloads on the concrete type (C++ name hiding). A
///     subclass that wants direct concrete-type access to all overloads should
///     bring them in with \c using \c DescriptorSource::Compute;. The calculator
///     invokes sources through a \c DescriptorSource& reference, so the base
///     overloads are always reachable there regardless of name hiding.
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

    /// \brief Optional override enabling shared-intermediate memoization and
    ///     column pruning. Base default ignores ctx and request and calls
    ///     Compute(mol) — a correct, unoptimized fallback.
    ///
    /// \param mol Molecule to describe.
    /// \param ctx Per-molecule cache of shared computation intermediates.
    /// \param request Which columns of this source's schema to compute.
    /// \returns Schema-backed descriptor row for the molecule.
    virtual DescriptorSet Compute(const OEChem::OEMolBase& mol,
                                  ComputeContext& ctx,
                                  const ColumnRequest& request) const;

    /// \brief Convenience: compute all columns with a caller-provided context.
    ///
    /// \param mol Molecule to describe.
    /// \param ctx Per-molecule cache of shared computation intermediates.
    /// \returns Schema-backed descriptor row for the molecule.
    DescriptorSet Compute(const OEChem::OEMolBase& mol, ComputeContext& ctx) const;
};

/// \brief Descriptor source producing Mordred-compatible descriptors.
class MordredDescriptorSource : public DescriptorSource {
public:
    // Bring the base convenience overloads into scope so overriding the
    // context/request method below does not hide Compute(mol) or Compute(mol, ctx).
    using DescriptorSource::Compute;

    std::shared_ptr<const DescriptorSchema> Schema() const override;
    DescriptorSet Compute(const OEChem::OEMolBase& mol) const override;
    DescriptorSet Compute(const OEChem::OEMolBase& mol,
                          ComputeContext& ctx,
                          const ColumnRequest& request) const override;
};

/// \brief Descriptor source producing OpenEye-native molecular properties.
///
/// The schema pairs seven tagged columns that share computation with Mordred
/// (so they carry a curated ``canonical_id`` and deduplicate against the
/// matching Mordred column) with four genuinely OpenEye-unique columns. The
/// unique columns use a distinct method from any Mordred column, carry an empty
/// ``canonical_id``, and are therefore never deduplicated.
class OpenEyePropertyDescriptorSource : public DescriptorSource {
public:
    // Bring the base convenience overloads into scope so overriding the
    // context/request method below does not hide Compute(mol) or Compute(mol, ctx).
    using DescriptorSource::Compute;

    std::shared_ptr<const DescriptorSchema> Schema() const override;
    DescriptorSet Compute(const OEChem::OEMolBase& mol) const override;
    DescriptorSet Compute(const OEChem::OEMolBase& mol,
                          ComputeContext& ctx,
                          const ColumnRequest& request) const override;
};

/// \brief Descriptor source reproducing RDKit 2D descriptors natively.
///
/// Values reproduce RDKit's numbers within a per-descriptor tolerance tier;
/// computation is native (OpenEye Toolkits), with RDKit used only as a
/// test-time conformance oracle. Most columns carry an empty ``canonical_id``
/// and coexist with conceptually similar Mordred/OpenEye columns; a small
/// curated set shares a ``canonical_id`` where the computation is provably
/// identical.
class RDKitDescriptorSource : public DescriptorSource {
public:
    using DescriptorSource::Compute;

    std::shared_ptr<const DescriptorSchema> Schema() const override;
    DescriptorSet Compute(const OEChem::OEMolBase& mol) const override;
    DescriptorSet Compute(const OEChem::OEMolBase& mol,
                          ComputeContext& ctx,
                          const ColumnRequest& request) const override;
};

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_SOURCE_H
