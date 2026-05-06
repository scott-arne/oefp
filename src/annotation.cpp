#include "oefp/annotation.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace OEFP {

OEFPBitEnvironment::OEFPBitEnvironment(std::uint32_t atom_id, std::uint32_t radius)
    : atom_id_(atom_id), radius_(radius) {
}

std::uint32_t OEFPBitEnvironment::AtomId() const {
    return atom_id_;
}

std::uint32_t OEFPBitEnvironment::Radius() const {
    return radius_;
}

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

void OEFPMappingSet::AddEnvironmentMapping(
    std::size_t row,
    std::uint64_t bit_id,
    std::uint32_t atom_id,
    std::uint32_t radius) {
    environment_mappings_[{row, bit_id}].emplace_back(atom_id, radius);
}

std::vector<std::uint32_t> OEFPMappingSet::AtomsForBit(
    std::size_t row,
    std::uint64_t bit_id) const {
    const auto found = atom_mappings_.find({row, bit_id});
    if (found != atom_mappings_.end()) {
        return found->second;
    }

    const auto environment_found = environment_mappings_.find({row, bit_id});
    if (environment_found == environment_mappings_.end()) {
        return {};
    }

    std::vector<std::uint32_t> atoms;
    atoms.reserve(environment_found->second.size());
    for (const auto& environment : environment_found->second) {
        atoms.push_back(environment.AtomId());
    }
    return atoms;
}

std::vector<OEFPBitEnvironment> OEFPMappingSet::EnvironmentsForBit(
    std::size_t row,
    std::uint64_t bit_id) const {
    const auto found = environment_mappings_.find({row, bit_id});
    if (found == environment_mappings_.end()) {
        return {};
    }
    return found->second;
}

std::vector<std::uint32_t> OEFPMappingSet::EnvironmentAtomIdsForBit(
    std::size_t row,
    std::uint64_t bit_id) const {
    const auto environments = EnvironmentsForBit(row, bit_id);
    std::vector<std::uint32_t> atom_ids;
    atom_ids.reserve(environments.size());
    for (const auto& environment : environments) {
        atom_ids.push_back(environment.AtomId());
    }
    return atom_ids;
}

std::vector<std::uint32_t> OEFPMappingSet::EnvironmentRadiiForBit(
    std::size_t row,
    std::uint64_t bit_id) const {
    const auto environments = EnvironmentsForBit(row, bit_id);
    std::vector<std::uint32_t> radii;
    radii.reserve(environments.size());
    for (const auto& environment : environments) {
        radii.push_back(environment.Radius());
    }
    return radii;
}

std::vector<std::uint64_t> OEFPMappingSet::BitIds(std::size_t row) const {
    std::vector<std::uint64_t> bit_ids;
    for (const auto& [key, environments] : environment_mappings_) {
        if (key.first == row && !environments.empty()) {
            bit_ids.push_back(key.second);
        }
    }
    for (const auto& [key, atoms] : atom_mappings_) {
        if (key.first == row && !atoms.empty()
            && environment_mappings_.find(key) == environment_mappings_.end()) {
            bit_ids.push_back(key.second);
        }
    }
    std::sort(bit_ids.begin(), bit_ids.end());
    bit_ids.erase(std::unique(bit_ids.begin(), bit_ids.end()), bit_ids.end());
    return bit_ids;
}

std::vector<std::uint32_t> OEFPMappingSet::BitIds32(std::size_t row) const {
    const auto bit_ids = BitIds(row);
    std::vector<std::uint32_t> bit_ids32;
    bit_ids32.reserve(bit_ids.size());
    for (const auto bit_id : bit_ids) {
        if (bit_id > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Mapped bit id exceeds uint32 storage.");
        }
        bit_ids32.push_back(static_cast<std::uint32_t>(bit_id));
    }
    return bit_ids32;
}

bool operator==(const OEFPBitEnvironment& lhs, const OEFPBitEnvironment& rhs) {
    return lhs.AtomId() == rhs.AtomId() && lhs.Radius() == rhs.Radius();
}

bool operator!=(const OEFPBitEnvironment& lhs, const OEFPBitEnvironment& rhs) {
    return !(lhs == rhs);
}

} // namespace OEFP
