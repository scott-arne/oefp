#include "oefp/count.h"

#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

FingerprintSpec validate_counted_spec(FingerprintSpec spec) {
    if (spec.value_type != FingerprintValueType::Counted) {
        throw std::invalid_argument("OEFPCount requires counted fingerprint metadata.");
    }
    return spec;
}

} // namespace

OEFPCount::OEFPCount(FingerprintSpec spec)
    : spec_(validate_counted_spec(std::move(spec))) {
}

OEFPCount::OEFPCount(
    FingerprintSpec spec,
    std::vector<std::uint32_t> indices,
    std::vector<std::uint32_t> counts)
    : spec_(std::move(spec)),
      indices_(std::move(indices)),
      counts_(std::move(counts)) {
    ValidateCountedSpec();
    ValidateSparseStorage();
}

const FingerprintSpec& OEFPCount::Spec() const {
    return spec_;
}

std::uint64_t OEFPCount::SizeBits() const {
    return spec_.size_bits;
}

std::size_t OEFPCount::NonzeroCount() const {
    return indices_.size();
}

std::uint64_t OEFPCount::TotalCount() const {
    std::uint64_t total = 0;
    for (const auto count : counts_) {
        total += count;
    }
    return total;
}

const std::vector<std::uint32_t>& OEFPCount::Indices() const {
    return indices_;
}

const std::vector<std::uint32_t>& OEFPCount::Counts() const {
    return counts_;
}

std::uint32_t OEFPCount::Index(std::size_t row) const {
    CheckRow(row);
    return indices_[row];
}

std::uint32_t OEFPCount::Count(std::size_t row) const {
    CheckRow(row);
    return counts_[row];
}

const std::uint32_t* OEFPCount::IndexData() const {
    if (indices_.empty()) {
        return nullptr;
    }
    return indices_.data();
}

const std::uint32_t* OEFPCount::CountData() const {
    if (counts_.empty()) {
        return nullptr;
    }
    return counts_.data();
}

std::uint64_t OEFPCount::IndexDataAddress() const {
    const auto* data = IndexData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

std::uint64_t OEFPCount::CountDataAddress() const {
    const auto* data = CountData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

void OEFPCount::ValidateCountedSpec() const {
    if (spec_.value_type != FingerprintValueType::Counted) {
        throw std::invalid_argument("OEFPCount requires counted fingerprint metadata.");
    }
}

void OEFPCount::ValidateSparseStorage() const {
    if (indices_.size() != counts_.size()) {
        throw std::invalid_argument("Count fingerprint indices and counts must have the same length.");
    }
    for (std::size_t row = 0; row < indices_.size(); ++row) {
        if (indices_[row] >= spec_.size_bits) {
            throw std::out_of_range("Count fingerprint index is out of range.");
        }
        if (counts_[row] == 0u) {
            throw std::invalid_argument("Count fingerprint sparse counts must be positive.");
        }
        if (row > 0 && indices_[row - 1] >= indices_[row]) {
            throw std::invalid_argument("Count fingerprint indices must be strictly increasing.");
        }
    }
}

void OEFPCount::CheckRow(std::size_t row) const {
    if (row >= indices_.size()) {
        throw std::out_of_range("Count fingerprint row is out of range.");
    }
}

bool operator==(const OEFPCount& lhs, const OEFPCount& rhs) {
    return lhs.Spec() == rhs.Spec() && lhs.Indices() == rhs.Indices()
        && lhs.Counts() == rhs.Counts();
}

bool operator!=(const OEFPCount& lhs, const OEFPCount& rhs) {
    return !(lhs == rhs);
}

} // namespace OEFP
