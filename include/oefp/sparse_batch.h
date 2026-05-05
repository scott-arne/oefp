#ifndef OEFP_SPARSE_BATCH_H
#define OEFP_SPARSE_BATCH_H

#include "oefp/sparse.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Contiguous sparse binary fingerprint batch.
///
/// Rows are stored in CSR-like form. ``row_offsets_[row]`` and
/// ``row_offsets_[row + 1]`` bound each row in the flattened index array.
class OEFPSparseBatch {
public:
    OEFPSparseBatch() = default;

    /// \brief Construct an empty sparse binary batch with an explicit spec.
    ///
    /// \param spec Shared sparse binary fingerprint metadata.
    /// \throws std::invalid_argument: When spec.value_type is not Binary.
    explicit OEFPSparseBatch(FingerprintSpec spec);

    /// \brief Build a sparse binary batch from compatible fingerprints.
    ///
    /// \throws std::invalid_argument: When fingerprints have mismatched specs.
    static OEFPSparseBatch FromFingerprints(const std::vector<::OEFP::OEFPSparse>& fingerprints);

    /// \brief Append one sparse binary fingerprint row.
    ///
    /// \throws std::invalid_argument: When the fingerprint is incompatible.
    void Append(const OEFPSparse& fp);

    /// \brief Return the shared fingerprint specification.
    const FingerprintSpec& Spec() const;

    /// \brief Return the number of fingerprint rows.
    std::size_t Size() const;

    /// \brief Return the sparse fingerprint identifier domain size.
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

    /// \brief Return read-only access to flattened sparse indices.
    const std::vector<std::uint32_t>& Indices() const;

    /// \brief Return read-only access to row offsets.
    const std::vector<std::uint64_t>& RowOffsets() const;

    /// \brief Return a raw pointer to flattened sparse indices.
    const std::uint32_t* IndexData() const;

    /// \brief Return a raw pointer to row offsets.
    const std::uint64_t* RowOffsetData() const;

    /// \brief Return the index data pointer as an integer address.
    std::uint64_t IndexDataAddress() const;

    /// \brief Return the row-offset data pointer as an integer address.
    std::uint64_t RowOffsetDataAddress() const;

private:
    FingerprintSpec spec_;
    std::vector<std::uint32_t> indices_;
    std::vector<std::uint64_t> row_offsets_{0u};

    void ValidateFingerprint(const OEFPSparse& fp) const;
    void CheckRowIndex(std::size_t row) const;
    void CheckOffsetIndex(std::size_t row) const;
};

bool operator==(const OEFPSparseBatch& lhs, const OEFPSparseBatch& rhs);
bool operator!=(const OEFPSparseBatch& lhs, const OEFPSparseBatch& rhs);

} // namespace OEFP

#endif // OEFP_SPARSE_BATCH_H
