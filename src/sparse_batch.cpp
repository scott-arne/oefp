#include "oefp/sparse_batch.h"

#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

FingerprintSpec validate_sparse_batch_spec(FingerprintSpec spec) {
    if (spec.value_type != FingerprintValueType::Binary) {
        throw std::invalid_argument("OEFPSparseBatch supports binary fingerprints only.");
    }
    return spec;
}

std::uint64_t pointer_address(const void* data) {
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

} // namespace

OEFPSparseBatch::OEFPSparseBatch(FingerprintSpec spec)
    : spec_(validate_sparse_batch_spec(std::move(spec))) {
}

OEFPSparseBatch OEFPSparseBatch::FromFingerprints(
    const std::vector<::OEFP::OEFPSparse>& fingerprints) {
    if (fingerprints.empty()) {
        return OEFPSparseBatch();
    }

    OEFPSparseBatch batch(fingerprints.front().Spec());
    for (const auto& fp : fingerprints) {
        batch.Append(fp);
    }
    return batch;
}

void OEFPSparseBatch::Append(const OEFPSparse& fp) {
    ValidateFingerprint(fp);

    auto next_indices = indices_;
    auto next_offsets = row_offsets_;
    const auto& fp_indices = fp.Indices();
    next_indices.insert(next_indices.end(), fp_indices.begin(), fp_indices.end());
    next_offsets.push_back(static_cast<std::uint64_t>(next_indices.size()));

    indices_ = std::move(next_indices);
    row_offsets_ = std::move(next_offsets);
}

const FingerprintSpec& OEFPSparseBatch::Spec() const {
    return spec_;
}

std::size_t OEFPSparseBatch::Size() const {
    return row_offsets_.empty() ? 0u : row_offsets_.size() - 1u;
}

std::uint64_t OEFPSparseBatch::SizeBits() const {
    return spec_.size_bits;
}

std::size_t OEFPSparseBatch::EntryCount() const {
    return indices_.size();
}

std::uint64_t OEFPSparseBatch::RowOffset(std::size_t row) const {
    CheckOffsetIndex(row);
    return row_offsets_[row];
}

std::size_t OEFPSparseBatch::RowEntryCount(std::size_t row) const {
    CheckRowIndex(row);
    return static_cast<std::size_t>(row_offsets_[row + 1u] - row_offsets_[row]);
}

const std::uint32_t* OEFPSparseBatch::RowIndices(std::size_t row) const {
    const auto count = RowEntryCount(row);
    if (count == 0u) {
        return nullptr;
    }
    return indices_.data() + static_cast<std::size_t>(row_offsets_[row]);
}

const std::vector<std::uint32_t>& OEFPSparseBatch::Indices() const {
    return indices_;
}

const std::vector<std::uint64_t>& OEFPSparseBatch::RowOffsets() const {
    return row_offsets_;
}

const std::uint32_t* OEFPSparseBatch::IndexData() const {
    if (indices_.empty()) {
        return nullptr;
    }
    return indices_.data();
}

const std::uint64_t* OEFPSparseBatch::RowOffsetData() const {
    if (row_offsets_.empty()) {
        return nullptr;
    }
    return row_offsets_.data();
}

std::uint64_t OEFPSparseBatch::IndexDataAddress() const {
    return pointer_address(IndexData());
}

std::uint64_t OEFPSparseBatch::RowOffsetDataAddress() const {
    return pointer_address(RowOffsetData());
}

void OEFPSparseBatch::ValidateFingerprint(const OEFPSparse& fp) const {
    if (fp.Spec() != spec_) {
        throw std::invalid_argument("Sparse fingerprint spec does not match batch spec.");
    }
    if (fp.SizeBits() == 0u) {
        throw std::invalid_argument("Cannot append zero-width sparse fingerprints to a batch.");
    }
}

void OEFPSparseBatch::CheckRowIndex(std::size_t row) const {
    if (row >= Size()) {
        throw std::out_of_range("Sparse batch row index is out of range.");
    }
}

void OEFPSparseBatch::CheckOffsetIndex(std::size_t row) const {
    if (row > Size()) {
        throw std::out_of_range("Sparse batch row offset index is out of range.");
    }
}

bool operator==(const OEFPSparseBatch& lhs, const OEFPSparseBatch& rhs) {
    return lhs.Spec() == rhs.Spec() && lhs.Indices() == rhs.Indices()
        && lhs.RowOffsets() == rhs.RowOffsets();
}

bool operator!=(const OEFPSparseBatch& lhs, const OEFPSparseBatch& rhs) {
    return !(lhs == rhs);
}

} // namespace OEFP
