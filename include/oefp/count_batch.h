#ifndef OEFP_COUNT_BATCH_H
#define OEFP_COUNT_BATCH_H

#include "oefp/count.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Contiguous sparse counted fingerprint batch.
///
/// Rows are stored in CSR-like form. ``row_offsets_[row]`` and
/// ``row_offsets_[row + 1]`` bound each row in the flattened index and count
/// arrays.
class OEFPCountBatch {
public:
    OEFPCountBatch() = default;

    /// \brief Construct an empty counted batch with an explicit spec.
    ///
    /// \param spec Shared counted fingerprint metadata.
    /// \throws std::invalid_argument: When spec.value_type is not Counted.
    explicit OEFPCountBatch(FingerprintSpec spec);

    /// \brief Build a counted batch from compatible counted fingerprints.
    ///
    /// \throws std::invalid_argument: When fingerprints have mismatched specs.
    static OEFPCountBatch FromFingerprints(const std::vector<::OEFP::OEFPCount>& fingerprints);

    /// \brief Append one counted fingerprint row.
    ///
    /// \throws std::invalid_argument: When the fingerprint is incompatible.
    void Append(const OEFPCount& fp);

    /// \brief Return the shared fingerprint specification.
    const FingerprintSpec& Spec() const;

    /// \brief Return the number of fingerprint rows.
    std::size_t Size() const;

    /// \brief Return the fixed folded fingerprint size.
    std::uint64_t SizeBits() const;

    /// \brief Return the total number of sparse entries.
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

    /// \brief Return a pointer to one row's sparse indices.
    ///
    /// Returns nullptr for empty rows.
    const std::uint32_t* RowIndices(std::size_t row) const;

    /// \brief Return a pointer to one row's sparse counts.
    ///
    /// Returns nullptr for empty rows.
    const std::uint32_t* RowCounts(std::size_t row) const;

    /// \brief Return read-only access to flattened sparse indices.
    const std::vector<std::uint32_t>& Indices() const;

    /// \brief Return read-only access to flattened sparse counts.
    const std::vector<std::uint32_t>& Counts() const;

    /// \brief Return read-only access to row offsets.
    const std::vector<std::uint64_t>& RowOffsets() const;

    /// \brief Return a raw pointer to flattened sparse indices.
    const std::uint32_t* IndexData() const;

    /// \brief Return a raw pointer to flattened sparse counts.
    const std::uint32_t* CountData() const;

    /// \brief Return a raw pointer to row offsets.
    const std::uint64_t* RowOffsetData() const;

    /// \brief Return the index data pointer as an integer address.
    std::uint64_t IndexDataAddress() const;

    /// \brief Return the count data pointer as an integer address.
    std::uint64_t CountDataAddress() const;

    /// \brief Return the row-offset data pointer as an integer address.
    std::uint64_t RowOffsetDataAddress() const;

private:
    FingerprintSpec spec_;
    std::vector<std::uint32_t> indices_;
    std::vector<std::uint32_t> counts_;
    std::vector<std::uint64_t> row_offsets_{0u};

    void ValidateFingerprint(const OEFPCount& fp) const;
    void CheckRowIndex(std::size_t row) const;
    void CheckOffsetIndex(std::size_t row) const;
};

bool operator==(const OEFPCountBatch& lhs, const OEFPCountBatch& rhs);
bool operator!=(const OEFPCountBatch& lhs, const OEFPCountBatch& rhs);

} // namespace OEFP

#endif // OEFP_COUNT_BATCH_H
