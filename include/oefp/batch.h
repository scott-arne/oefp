#ifndef OEFP_BATCH_H
#define OEFP_BATCH_H

#include "oefp/fingerprint.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Contiguous row-major storage for dense binary fingerprints.
class OEFPBatch {
public:
    OEFPBatch() = default;

    /// \brief Construct an empty batch with a known fingerprint specification.
    ///
    /// Zero-bit specifications are allowed for empty batches, but appending
    /// zero-width fingerprints is rejected.
    ///
    /// \raises std::invalid_argument: When spec.value_type is not Binary.
    explicit OEFPBatch(FingerprintSpec spec);

    /// \brief Build a batch by copying fingerprints into contiguous rows.
    ///
    /// Returns a default empty batch when fingerprints is empty.
    ///
    /// \raises std::invalid_argument: When any fingerprint has an incompatible
    ///     spec or zero-width storage.
    static OEFPBatch FromFingerprints(const std::vector<OEFP>& fingerprints);

    /// \brief Append one fingerprint row.
    ///
    /// The fingerprint spec must exactly match the batch spec. Zero-width
    /// fingerprints are rejected so row pointers always identify real storage.
    ///
    /// \raises std::invalid_argument: When the fingerprint spec is incompatible
    ///     or has zero-width storage.
    void Append(const OEFP& fp);

    /// \brief Return the shared fingerprint specification.
    const FingerprintSpec& Spec() const;

    /// \brief Return the number of stored fingerprint rows.
    std::size_t Size() const;

    /// \brief Return the fixed fingerprint size in bits.
    std::uint64_t SizeBits() const;

    /// \brief Return the number of uint64 words in each row.
    std::size_t WordsPerFingerprint() const;

    /// \brief Return the total number of stored uint64 words.
    std::size_t WordCount() const;

    /// \brief Return a pointer to one row in the row-major word storage.
    ///
    /// The returned pointer remains valid until the batch is modified or
    /// destroyed.
    ///
    /// \raises std::out_of_range: When row is greater than or equal to Size().
    const std::uint64_t* RowWords(std::size_t row) const;

    /// \brief Return read-only access to all row-major fingerprint words.
    const std::vector<std::uint64_t>& Words() const;

    /// \brief Return read-only access to one cached popcount per row.
    const std::vector<std::uint32_t>& PopCounts() const;

    /// \brief Return the cached popcount for one row.
    ///
    /// \raises std::out_of_range: When row is greater than or equal to Size().
    std::uint32_t PopCount(std::size_t row) const;

    /// \brief Return a raw pointer to the contiguous word storage.
    ///
    /// Returns nullptr for an empty batch. The pointer remains valid until the
    /// batch is modified or destroyed.
    const std::uint64_t* WordData() const;

    /// \brief Return a raw pointer to the contiguous popcount storage.
    ///
    /// Returns nullptr for an empty batch. The pointer remains valid until the
    /// batch is modified or destroyed.
    const std::uint32_t* PopCountData() const;

    /// \brief Return the word data pointer as an integer address.
    std::uintptr_t WordDataAddress() const;

    /// \brief Return the popcount data pointer as an integer address.
    std::uintptr_t PopCountDataAddress() const;

private:
    FingerprintSpec spec_;
    std::size_t words_per_fingerprint_ = 0;
    std::vector<std::uint64_t> words_;
    std::vector<std::uint32_t> popcounts_;

    void ReserveRows(std::size_t row_count);
    void ValidateFingerprint(const OEFP& fp) const;
    void CheckRowIndex(std::size_t row) const;
};

} // namespace OEFP

#endif // OEFP_BATCH_H
