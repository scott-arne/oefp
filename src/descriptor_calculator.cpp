#include "oefp/descriptor_calculator.h"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

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

DescriptorSet DescriptorCalculator::Compute(const OEChem::OEMolBase& /*mol*/) const {
    throw std::logic_error("not implemented");
}

DescriptorBatch DescriptorCalculator::CalculateBatch(
    const std::vector<const OEChem::OEMolBase*>& /*mols*/) const {
    throw std::logic_error("not implemented");
}

} // namespace OEFP
