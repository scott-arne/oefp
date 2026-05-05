#include "oefp/count_batch.h"

#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

FingerprintSpec validate_counted_batch_spec(FingerprintSpec spec) {
    if (spec.value_type != FingerprintValueType::Counted) {
        throw std::invalid_argument("OEFPCountBatch supports counted fingerprints only.");
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

OEFPCountBatch::OEFPCountBatch(FingerprintSpec spec)
    : spec_(validate_counted_batch_spec(std::move(spec))) {
}

OEFPCountBatch OEFPCountBatch::FromFingerprints(
    const std::vector<::OEFP::OEFPCount>& fingerprints) {
    if (fingerprints.empty()) {
        return OEFPCountBatch();
    }

    OEFPCountBatch batch(fingerprints.front().Spec());
    for (const auto& fp : fingerprints) {
        batch.Append(fp);
    }
    return batch;
}

void OEFPCountBatch::Append(const OEFPCount& fp) {
    ValidateFingerprint(fp);

    auto next_indices = indices_;
    auto next_counts = counts_;
    auto next_offsets = row_offsets_;
    const auto& fp_indices = fp.Indices();
    const auto& fp_counts = fp.Counts();
    next_indices.insert(next_indices.end(), fp_indices.begin(), fp_indices.end());
    next_counts.insert(next_counts.end(), fp_counts.begin(), fp_counts.end());
    next_offsets.push_back(static_cast<std::uint64_t>(next_indices.size()));

    indices_ = std::move(next_indices);
    counts_ = std::move(next_counts);
    row_offsets_ = std::move(next_offsets);
}

const FingerprintSpec& OEFPCountBatch::Spec() const {
    return spec_;
}

std::size_t OEFPCountBatch::Size() const {
    return row_offsets_.empty() ? 0u : row_offsets_.size() - 1u;
}

std::uint64_t OEFPCountBatch::SizeBits() const {
    return spec_.size_bits;
}

std::size_t OEFPCountBatch::EntryCount() const {
    return indices_.size();
}

std::uint64_t OEFPCountBatch::RowOffset(std::size_t row) const {
    CheckOffsetIndex(row);
    return row_offsets_[row];
}

std::size_t OEFPCountBatch::RowEntryCount(std::size_t row) const {
    CheckRowIndex(row);
    return static_cast<std::size_t>(row_offsets_[row + 1u] - row_offsets_[row]);
}

const std::uint32_t* OEFPCountBatch::RowIndices(std::size_t row) const {
    const auto count = RowEntryCount(row);
    if (count == 0u) {
        return nullptr;
    }
    return indices_.data() + static_cast<std::size_t>(row_offsets_[row]);
}

const std::uint32_t* OEFPCountBatch::RowCounts(std::size_t row) const {
    const auto count = RowEntryCount(row);
    if (count == 0u) {
        return nullptr;
    }
    return counts_.data() + static_cast<std::size_t>(row_offsets_[row]);
}

const std::vector<std::uint32_t>& OEFPCountBatch::Indices() const {
    return indices_;
}

const std::vector<std::uint32_t>& OEFPCountBatch::Counts() const {
    return counts_;
}

const std::vector<std::uint64_t>& OEFPCountBatch::RowOffsets() const {
    return row_offsets_;
}

const std::uint32_t* OEFPCountBatch::IndexData() const {
    if (indices_.empty()) {
        return nullptr;
    }
    return indices_.data();
}

const std::uint32_t* OEFPCountBatch::CountData() const {
    if (counts_.empty()) {
        return nullptr;
    }
    return counts_.data();
}

const std::uint64_t* OEFPCountBatch::RowOffsetData() const {
    if (row_offsets_.empty()) {
        return nullptr;
    }
    return row_offsets_.data();
}

std::uint64_t OEFPCountBatch::IndexDataAddress() const {
    return pointer_address(IndexData());
}

std::uint64_t OEFPCountBatch::CountDataAddress() const {
    return pointer_address(CountData());
}

std::uint64_t OEFPCountBatch::RowOffsetDataAddress() const {
    return pointer_address(RowOffsetData());
}

void OEFPCountBatch::ValidateFingerprint(const OEFPCount& fp) const {
    if (fp.Spec() != spec_) {
        throw std::invalid_argument("Count fingerprint spec does not match batch spec.");
    }
    if (fp.SizeBits() == 0u) {
        throw std::invalid_argument("Cannot append zero-width count fingerprints to a batch.");
    }
}

void OEFPCountBatch::CheckRowIndex(std::size_t row) const {
    if (row >= Size()) {
        throw std::out_of_range("Count batch row index is out of range.");
    }
}

void OEFPCountBatch::CheckOffsetIndex(std::size_t row) const {
    if (row > Size()) {
        throw std::out_of_range("Count batch row offset index is out of range.");
    }
}

bool operator==(const OEFPCountBatch& lhs, const OEFPCountBatch& rhs) {
    return lhs.Spec() == rhs.Spec() && lhs.Indices() == rhs.Indices()
        && lhs.Counts() == rhs.Counts() && lhs.RowOffsets() == rhs.RowOffsets();
}

bool operator!=(const OEFPCountBatch& lhs, const OEFPCountBatch& rhs) {
    return !(lhs == rhs);
}

} // namespace OEFP
