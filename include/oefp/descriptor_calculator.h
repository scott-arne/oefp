#ifndef OEFP_DESCRIPTOR_CALCULATOR_H
#define OEFP_DESCRIPTOR_CALCULATOR_H

#include "oefp/column_request.h"
#include "oefp/descriptor.h"
#include "oefp/descriptor_batch.h"
#include "oefp/descriptor_schema.h"
#include "oefp/descriptor_selection.h"
#include "oefp/descriptor_source.h"

#include <oechem.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace OEFP {

/// \brief One descriptor source paired with an optional column selection.
///
/// A selection narrows the source's schema to a subset of columns before
/// deduplication. When absent, every column the source produces participates.
struct DescriptorSourceEntry {
    std::shared_ptr<const DescriptorSource> source;
    std::optional<DescriptorSelection> selection;

    /// \brief Register every column produced by a source.
    ///
    /// \param source Descriptor source contributing columns.
    DescriptorSourceEntry(std::shared_ptr<const DescriptorSource> source);

    /// \brief Register a source narrowed to a subset of its columns.
    ///
    /// \param source Descriptor source contributing columns.
    /// \param selection Selection resolved against the source's schema before
    ///     deduplication.
    DescriptorSourceEntry(
        std::shared_ptr<const DescriptorSource> source,
        DescriptorSelection selection);
};

/// \brief Resolves several descriptor sources into one deduplicated schema.
///
/// At construction the calculator applies each entry's optional selection,
/// deduplicates columns that share a non-empty ``canonical_id`` on a first-wins
/// basis, and rejects unresolved name collisions. The merged schema and the
/// per-source copy plan are fixed once and reused for every computation.
class DescriptorCalculator {
public:
    /// \brief Resolve the merged schema and copy plan from source entries.
    ///
    /// Each entry is processed in order. For every surviving column: a
    /// non-empty ``canonical_id`` already claimed by an earlier column is
    /// dropped (first-wins); otherwise a name already present in the merged
    /// schema is an unresolved collision and is rejected. An empty merged
    /// schema (no entries, or every column selected or deduplicated away) is
    /// valid.
    ///
    /// \param entries Descriptor sources with optional selections.
    /// \throws std::invalid_argument: When a source is null, a source returns a
    ///     null schema, or two surviving columns collide on name without a
    ///     shared ``canonical_id``.
    explicit DescriptorCalculator(std::vector<DescriptorSourceEntry> entries);

    /// \brief Return the merged, deduplicated descriptor schema.
    const DescriptorSchema& Schema() const;

    /// \brief Return a shared handle to the merged, deduplicated schema.
    std::shared_ptr<const DescriptorSchema> SchemaPtr() const;

    /// \brief Compute the merged descriptor row for one molecule.
    ///
    /// \param mol Molecule to describe.
    /// \returns Schema-backed descriptor row over the merged schema.
    DescriptorSet Compute(const OEChem::OEMolBase& mol) const;

    /// \brief Compute merged descriptor rows for several molecules.
    ///
    /// \param mols Molecules to describe.
    /// \returns Descriptor batch over the merged schema.
    DescriptorBatch CalculateBatch(const std::vector<const OEChem::OEMolBase*>& mols) const;

private:
    /// \brief Per-source copy plan mapping source columns to merged slots.
    struct SourcePlan {
        std::shared_ptr<const DescriptorSource> source;
        /// \brief Pairs of (source column index, merged schema slot) that survive.
        std::vector<std::pair<std::size_t, std::size_t>> kept;
        /// \brief Request naming exactly this source's surviving columns.
        ColumnRequest request = ColumnRequest::All();
    };

    std::vector<SourcePlan> plans_;
    std::shared_ptr<const DescriptorSchema> schema_;
};

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_CALCULATOR_H
