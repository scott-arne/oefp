#ifndef OEFP_DESCRIPTOR_BATCH_H
#define OEFP_DESCRIPTOR_BATCH_H

#include "oefp/descriptor.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace OEFP {

/// \brief Contiguous descriptor batch storage.
///
/// Rows are stored in CSR-like form. ``row_offsets_[row]`` and
/// ``row_offsets_[row + 1]`` bound each descriptor row in the flattened active
/// key vector and count array.
class DescriptorBatch {
public:
    DescriptorBatch() = default;

    /// \brief Construct an empty descriptor batch with an explicit spec.
    explicit DescriptorBatch(DescriptorSpec spec);

    /// \brief Build a descriptor batch from compatible descriptor sets.
    ///
    /// \throws std::invalid_argument: When descriptor sets have mismatched specs.
    static DescriptorBatch FromDescriptorSets(
        const std::vector<::OEFP::DescriptorSet>& descriptors);

    /// \brief Append one descriptor row.
    ///
    /// \throws std::invalid_argument: When the descriptor set is incompatible.
    void Append(const DescriptorSet& descriptors);

    /// \brief Return the shared descriptor specification.
    const DescriptorSpec& Spec() const;

    /// \brief Return the shared descriptor value type.
    DescriptorValueType ValueType() const;

    /// \brief Return the number of descriptor rows.
    std::size_t Size() const;

    /// \brief Return the total number of flattened descriptor entries.
    std::size_t EntryCount() const;

    /// \brief Return one row offset.
    ///
    /// ``row`` may equal ``Size()`` to access the sentinel final offset.
    ///
    /// \throws std::out_of_range: When row is greater than Size().
    std::uint64_t RowOffset(std::size_t row) const;

    /// \brief Return the number of entries in one row.
    ///
    /// \throws std::out_of_range: When row is greater than or equal to Size().
    std::size_t RowEntryCount(std::size_t row) const;

    /// \brief Return read-only access to flattened string keys.
    const std::vector<std::string>& StringKeys() const;

    /// \brief Return read-only access to flattened integer keys.
    const std::vector<std::int64_t>& IntegerKeys() const;

    /// \brief Return read-only access to flattened float keys.
    const std::vector<double>& FloatKeys() const;

    /// \brief Return read-only access to flattened descriptor counts.
    const std::vector<std::uint32_t>& Counts() const;

    /// \brief Return read-only access to row offsets.
    const std::vector<std::uint64_t>& RowOffsets() const;

    /// \brief Return read-only access to flattened descriptor counts.
    const std::uint32_t* CountData() const;
    std::uint64_t CountDataAddress() const;

    /// \brief Return read-only access to row offsets.
    const std::uint64_t* RowOffsetData() const;
    std::uint64_t RowOffsetDataAddress() const;

    /// \brief Return read-only access to flattened integer keys.
    const std::int64_t* IntegerKeyData() const;
    std::uint64_t IntegerKeyDataAddress() const;

    /// \brief Return read-only access to flattened float keys.
    const double* FloatKeyData() const;
    std::uint64_t FloatKeyDataAddress() const;

private:
    DescriptorSpec spec_;
    std::vector<std::string> string_keys_;
    std::vector<std::int64_t> integer_keys_;
    std::vector<double> float_keys_;
    std::vector<std::uint32_t> counts_;
    std::vector<std::uint64_t> row_offsets_{0u};
    bool has_spec_ = false;

    void ReserveRowsAndEntries(std::size_t row_count, std::size_t entry_count);
    void AppendActiveKeys(const DescriptorSet& descriptors);
    void ValidateDescriptorSet(const DescriptorSet& descriptors) const;
    void CheckRowIndex(std::size_t row) const;
    void CheckOffsetIndex(std::size_t row) const;

    friend bool operator==(const DescriptorBatch& lhs, const DescriptorBatch& rhs);
};

bool operator==(const DescriptorBatch& lhs, const DescriptorBatch& rhs);
bool operator!=(const DescriptorBatch& lhs, const DescriptorBatch& rhs);

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_BATCH_H
