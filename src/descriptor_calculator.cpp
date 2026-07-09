#include "oefp/descriptor_calculator.h"

#include "oefp/column_request.h"
#include "oefp/compute_context.h"
#include "thread_pool.h"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace OEFP {

DescriptorSourceEntry::DescriptorSourceEntry(std::shared_ptr<const DescriptorSource> source)
    : source(std::move(source)) {}

DescriptorSourceEntry::DescriptorSourceEntry(
    std::shared_ptr<const DescriptorSource> source,
    DescriptorSelection selection)
    : source(std::move(source)), selection(std::move(selection)) {}

DescriptorCalculator::DescriptorCalculator(std::vector<DescriptorSourceEntry> entries) {
    std::vector<DescriptorDefinition> merged;
    // Canonical ids already claimed by an earlier surviving column (first-wins).
    std::unordered_set<std::string> seen;
    // Merged column name to owning source, used for collision diagnostics.
    std::unordered_map<std::string, std::string> owning_source_by_name;

    plans_.reserve(entries.size());
    for (const auto& entry : entries) {
        if (!entry.source) {
            throw std::invalid_argument("DescriptorCalculator source must not be null.");
        }
        auto source_schema = entry.source->Schema();
        if (!source_schema) {
            throw std::invalid_argument("DescriptorSource returned a null schema.");
        }

        std::vector<std::size_t> indices;
        if (entry.selection) {
            indices = entry.selection->Resolve(*source_schema);
        } else {
            indices.reserve(source_schema->Size());
            for (std::size_t i = 0; i < source_schema->Size(); ++i) {
                indices.push_back(i);
            }
        }

        SourcePlan plan;
        plan.source = entry.source;
        for (const std::size_t i : indices) {
            const DescriptorDefinition& def = source_schema->Definition(i);
            if (!def.canonical_id.empty() && seen.count(def.canonical_id) != 0) {
                continue;  // Identical column already claimed by an earlier source.
            }
            if (owning_source_by_name.count(def.name) != 0) {
                throw std::invalid_argument(
                    "Descriptor name collision: '" + def.name + "' from source '" +
                    def.source_name + "' collides with an earlier source; tag them with a "
                    "shared canonical_id or narrow one out.");
            }
            const std::size_t merged_slot = merged.size();
            merged.push_back(def);
            plan.kept.emplace_back(i, merged_slot);
            owning_source_by_name.emplace(def.name, def.source_name);
            if (!def.canonical_id.empty()) {
                seen.insert(def.canonical_id);
            }
        }
        // Name exactly the surviving source columns so a source can prune the
        // rest. Sources ignore the request until later phases, so this is a
        // value-safe no-op today: everything is still computed and the kept
        // columns copied as before.
        std::vector<std::size_t> kept_source_indices;
        kept_source_indices.reserve(plan.kept.size());
        for (const auto& [src_idx, merged_slot] : plan.kept) {
            kept_source_indices.push_back(src_idx);
        }
        plan.request = ColumnRequest::Subset(std::move(kept_source_indices));
        plans_.push_back(std::move(plan));
    }

    schema_ = std::make_shared<const DescriptorSchema>(std::move(merged));
}

const DescriptorSchema& DescriptorCalculator::Schema() const {
    return *schema_;
}

std::shared_ptr<const DescriptorSchema> DescriptorCalculator::SchemaPtr() const {
    return schema_;
}

DescriptorSet DescriptorCalculator::Compute(const OEChem::OEMolBase& mol) const {
    DescriptorSetBuilder builder(schema_);
    // One context per molecule, shared across every source, so cross-source
    // intermediates (ring perception, heavy-atom graph, ...) are computed once
    // and reused across Mordred and OpenEye. The context is a local here and is
    // never shared across molecules or threads: CalculateBatch invokes Compute
    // per row, so each parallel worker builds its own context.
    ComputeContext ctx(mol);
    for (const SourcePlan& plan : plans_) {
        if (plan.kept.empty()) {
            // A source selected or deduplicated down to nothing contributes no
            // columns; skip its Compute so it is a genuine no-op at compute time.
            continue;
        }
        const DescriptorSet row = plan.source->Compute(ctx.NormalizedMol(), ctx, plan.request);
        for (const auto& [src_idx, merged_slot] : plan.kept) {
            // Copy only present source values; missing values stay missing.
            if (row.Has(src_idx)) {
                builder.Set(schema_->Definition(merged_slot).name, row.Value(src_idx));
            }
        }
    }
    return builder.Build();
}

DescriptorBatch DescriptorCalculator::CalculateBatch(
    const std::vector<const OEChem::OEMolBase*>& mols) const {
    // Reject null molecules up front: worker threads dereference each pointer,
    // so a null here would be undefined behavior rather than a typed error.
    for (const auto* mol : mols) {
        if (mol == nullptr) {
            throw std::invalid_argument(
                "DescriptorCalculator::CalculateBatch molecule pointers must not be null.");
        }
    }
    // Compute rows in parallel; each row is written at a disjoint index and
    // every source's Compute is a pure function of its molecule, so no locking
    // is required for the per-row work.
    std::vector<DescriptorSet> rows(mols.size());
    detail::ParallelFor(0, mols.size(), 1, 0, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            rows[i] = Compute(*mols[i]);
        }
    });

    // Assemble serially to preserve input order and because Append mutates
    // shared batch state.
    auto batch = DescriptorBatch::Empty(schema_);
    for (const auto& r : rows) {
        batch.Append(r);
    }
    return batch;
}

} // namespace OEFP
