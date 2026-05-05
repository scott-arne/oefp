#include "oefp/sparse.h"

#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

FingerprintSpec validate_binary_spec(FingerprintSpec spec) {
    if (spec.value_type != FingerprintValueType::Binary) {
        throw std::invalid_argument("OEFPSparse requires binary fingerprint metadata.");
    }
    return spec;
}

} // namespace

OEFPSparse::OEFPSparse(FingerprintSpec spec)
    : spec_(validate_binary_spec(std::move(spec))) {
}

OEFPSparse::OEFPSparse(FingerprintSpec spec, std::vector<std::uint32_t> indices)
    : spec_(std::move(spec)),
      indices_(std::move(indices)) {
    ValidateBinarySpec();
    ValidateSparseStorage();
}

const FingerprintSpec& OEFPSparse::Spec() const {
    return spec_;
}

std::uint64_t OEFPSparse::SizeBits() const {
    return spec_.size_bits;
}

std::size_t OEFPSparse::CountOnBits() const {
    return indices_.size();
}

const std::vector<std::uint32_t>& OEFPSparse::Indices() const {
    return indices_;
}

std::uint32_t OEFPSparse::Index(std::size_t row) const {
    CheckRow(row);
    return indices_[row];
}

const std::uint32_t* OEFPSparse::IndexData() const {
    if (indices_.empty()) {
        return nullptr;
    }
    return indices_.data();
}

std::uint64_t OEFPSparse::IndexDataAddress() const {
    const auto* data = IndexData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

void OEFPSparse::ValidateBinarySpec() const {
    if (spec_.value_type != FingerprintValueType::Binary) {
        throw std::invalid_argument("OEFPSparse requires binary fingerprint metadata.");
    }
}

void OEFPSparse::ValidateSparseStorage() const {
    for (std::size_t row = 0; row < indices_.size(); ++row) {
        if (indices_[row] >= spec_.size_bits) {
            throw std::out_of_range("Sparse fingerprint index is out of range.");
        }
        if (row > 0 && indices_[row - 1] >= indices_[row]) {
            throw std::invalid_argument("Sparse fingerprint indices must be strictly increasing.");
        }
    }
}

void OEFPSparse::CheckRow(std::size_t row) const {
    if (row >= indices_.size()) {
        throw std::out_of_range("Sparse fingerprint row is out of range.");
    }
}

bool operator==(const OEFPSparse& lhs, const OEFPSparse& rhs) {
    return lhs.Spec() == rhs.Spec() && lhs.Indices() == rhs.Indices();
}

bool operator!=(const OEFPSparse& lhs, const OEFPSparse& rhs) {
    return !(lhs == rhs);
}

} // namespace OEFP
