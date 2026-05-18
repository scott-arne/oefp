#include "oefp/descriptor_batch.h"

#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

std::size_t total_entry_count(const std::vector<::OEFP::DescriptorSet>& descriptors) {
    std::size_t entry_count = 0u;
    for (const auto& descriptor_set : descriptors) {
        entry_count += descriptor_set.Size();
    }
    return entry_count;
}

} // namespace

DescriptorBatch::DescriptorBatch(DescriptorSpec spec)
    : spec_(std::move(spec)),
      has_spec_(true) {
}

DescriptorBatch DescriptorBatch::FromDescriptorSets(
    const std::vector<::OEFP::DescriptorSet>& descriptors) {
    if (descriptors.empty()) {
        return DescriptorBatch();
    }

    DescriptorBatch batch(descriptors.front().Spec());
    batch.ReserveRowsAndEntries(descriptors.size(), total_entry_count(descriptors));
    for (const auto& descriptor_set : descriptors) {
        batch.Append(descriptor_set);
    }
    return batch;
}

void DescriptorBatch::Append(const DescriptorSet& descriptors) {
    ValidateDescriptorSet(descriptors);

    if (!has_spec_) {
        spec_ = descriptors.Spec();
        has_spec_ = true;
    }

    AppendActiveKeys(descriptors);
    const auto& descriptor_counts = descriptors.Counts();
    counts_.insert(counts_.end(), descriptor_counts.begin(), descriptor_counts.end());
    row_offsets_.push_back(static_cast<std::uint64_t>(EntryCount()));
}

void DescriptorBatch::ReserveRowsAndEntries(std::size_t row_count, std::size_t entry_count) {
    row_offsets_.reserve(row_count + 1u);
    counts_.reserve(entry_count);

    switch (ValueType()) {
    case DescriptorValueType::Integer:
        integer_keys_.reserve(entry_count);
        break;
    case DescriptorValueType::Float:
        float_keys_.reserve(entry_count);
        break;
    case DescriptorValueType::String:
        string_keys_.reserve(entry_count);
        break;
    }
}

void DescriptorBatch::AppendActiveKeys(const DescriptorSet& descriptors) {
    switch (ValueType()) {
    case DescriptorValueType::Integer: {
        const auto& keys = descriptors.IntegerKeys();
        integer_keys_.insert(integer_keys_.end(), keys.begin(), keys.end());
        break;
    }
    case DescriptorValueType::Float: {
        const auto& keys = descriptors.FloatKeys();
        float_keys_.insert(float_keys_.end(), keys.begin(), keys.end());
        break;
    }
    case DescriptorValueType::String: {
        const auto& keys = descriptors.StringKeys();
        string_keys_.insert(string_keys_.end(), keys.begin(), keys.end());
        break;
    }
    }
}

const DescriptorSpec& DescriptorBatch::Spec() const {
    return spec_;
}

DescriptorValueType DescriptorBatch::ValueType() const {
    return spec_.value_type;
}

std::size_t DescriptorBatch::Size() const {
    return row_offsets_.empty() ? 0u : row_offsets_.size() - 1u;
}

std::size_t DescriptorBatch::EntryCount() const {
    switch (ValueType()) {
    case DescriptorValueType::Integer:
        return integer_keys_.size();
    case DescriptorValueType::Float:
        return float_keys_.size();
    case DescriptorValueType::String:
        return string_keys_.size();
    }
    return 0u;
}

std::uint64_t DescriptorBatch::RowOffset(std::size_t row) const {
    CheckOffsetIndex(row);
    return row_offsets_[row];
}

std::size_t DescriptorBatch::RowEntryCount(std::size_t row) const {
    CheckRowIndex(row);
    return static_cast<std::size_t>(row_offsets_[row + 1u] - row_offsets_[row]);
}

const std::vector<std::string>& DescriptorBatch::StringKeys() const {
    return string_keys_;
}

const std::vector<std::int64_t>& DescriptorBatch::IntegerKeys() const {
    return integer_keys_;
}

const std::vector<double>& DescriptorBatch::FloatKeys() const {
    return float_keys_;
}

const std::vector<std::uint32_t>& DescriptorBatch::Counts() const {
    return counts_;
}

const std::vector<std::uint64_t>& DescriptorBatch::RowOffsets() const {
    return row_offsets_;
}

const std::uint32_t* DescriptorBatch::CountData() const {
    if (counts_.empty()) {
        return nullptr;
    }
    return counts_.data();
}

std::uint64_t DescriptorBatch::CountDataAddress() const {
    const auto* data = CountData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const std::uint64_t* DescriptorBatch::RowOffsetData() const {
    if (row_offsets_.empty()) {
        return nullptr;
    }
    return row_offsets_.data();
}

std::uint64_t DescriptorBatch::RowOffsetDataAddress() const {
    const auto* data = RowOffsetData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const std::int64_t* DescriptorBatch::IntegerKeyData() const {
    if (integer_keys_.empty()) {
        return nullptr;
    }
    return integer_keys_.data();
}

std::uint64_t DescriptorBatch::IntegerKeyDataAddress() const {
    const auto* data = IntegerKeyData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const double* DescriptorBatch::FloatKeyData() const {
    if (float_keys_.empty()) {
        return nullptr;
    }
    return float_keys_.data();
}

std::uint64_t DescriptorBatch::FloatKeyDataAddress() const {
    const auto* data = FloatKeyData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

void DescriptorBatch::ValidateDescriptorSet(const DescriptorSet& descriptors) const {
    if (has_spec_ && descriptors.Spec() != spec_) {
        throw std::invalid_argument("Descriptor set spec does not match batch spec.");
    }
}

void DescriptorBatch::CheckRowIndex(std::size_t row) const {
    if (row >= Size()) {
        throw std::out_of_range("Descriptor batch row index is out of range.");
    }
}

void DescriptorBatch::CheckOffsetIndex(std::size_t row) const {
    if (row > Size()) {
        throw std::out_of_range("Descriptor batch row offset index is out of range.");
    }
}

bool operator==(const DescriptorBatch& lhs, const DescriptorBatch& rhs) {
    return lhs.has_spec_ == rhs.has_spec_ && lhs.Spec() == rhs.Spec()
        && lhs.StringKeys() == rhs.StringKeys()
        && lhs.IntegerKeys() == rhs.IntegerKeys() && lhs.FloatKeys() == rhs.FloatKeys()
        && lhs.Counts() == rhs.Counts() && lhs.RowOffsets() == rhs.RowOffsets();
}

bool operator!=(const DescriptorBatch& lhs, const DescriptorBatch& rhs) {
    return !(lhs == rhs);
}

} // namespace OEFP
