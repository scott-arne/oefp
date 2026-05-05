#include "oefp/annotation.h"

#include <utility>

namespace OEFP {

void OEFPAnnotationSet::SetBitLabel(std::uint64_t bit_id, std::string label) {
    bit_labels_[bit_id] = std::move(label);
}

std::string OEFPAnnotationSet::BitLabel(std::uint64_t bit_id) const {
    const auto found = bit_labels_.find(bit_id);
    if (found == bit_labels_.end()) {
        return {};
    }
    return found->second;
}

void OEFPMappingSet::AddAtomMapping(
    std::size_t row,
    std::uint64_t bit_id,
    std::vector<std::uint32_t> atom_ids) {
    atom_mappings_[{row, bit_id}] = std::move(atom_ids);
}

std::vector<std::uint32_t> OEFPMappingSet::AtomsForBit(
    std::size_t row,
    std::uint64_t bit_id) const {
    const auto found = atom_mappings_.find({row, bit_id});
    if (found == atom_mappings_.end()) {
        return {};
    }
    return found->second;
}

} // namespace OEFP
