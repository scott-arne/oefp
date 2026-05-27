#ifndef OEFP_COUNT_H
#define OEFP_COUNT_H

#include "oefp/fingerprint.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Sparse counted fingerprint with sorted nonzero entries.
///
/// Counted fingerprints store only nonzero folded feature counts. The index and
/// count arrays are parallel, sorted by index, and contain no duplicate indices.
class OEFPCount {
public:
    OEFPCount() = default;

    /// \brief Construct an empty counted fingerprint.
    ///
    /// \param spec Fingerprint size and provenance metadata.
    /// \throws std::invalid_argument: When spec.value_type is not Counted.
    explicit OEFPCount(FingerprintSpec spec);

    /// \brief Construct a counted fingerprint from sparse count arrays.
    ///
    /// \param spec Fingerprint size and provenance metadata.
    /// \param indices Sorted nonzero folded bit indices.
    /// \param counts Positive counts parallel to indices.
    /// \throws std::invalid_argument: When the spec or sparse arrays are invalid.
    /// \throws std::out_of_range: When an index is outside the fingerprint size.
    OEFPCount(
        FingerprintSpec spec,
        std::vector<std::uint32_t> indices,
        std::vector<std::uint32_t> counts);

    /// \brief Return the fingerprint specification.
    const FingerprintSpec& Spec() const;

    /// \brief Return the fixed folded fingerprint size.
    std::uint64_t SizeBits() const;

    /// \brief Return the number of nonzero count entries.
    std::size_t NonzeroCount() const;

    /// \brief Return the sum of all sparse counts.
    std::uint64_t TotalCount() const;

    /// \brief Return read-only access to sparse indices.
    const std::vector<std::uint32_t>& Indices() const;

    /// \brief Return read-only access to sparse counts.
    const std::vector<std::uint32_t>& Counts() const;

    /// \brief Return one sparse index by row.
    ///
    /// \throws std::out_of_range: When row is greater than or equal to
    ///     NonzeroCount().
    std::uint32_t Index(std::size_t row) const;

    /// \brief Return one sparse count by row.
    ///
    /// \throws std::out_of_range: When row is greater than or equal to
    ///     NonzeroCount().
    std::uint32_t Count(std::size_t row) const;

    /// \brief Return a raw pointer to sparse indices.
    ///
    /// Returns nullptr for empty counted fingerprints.
    const std::uint32_t* IndexData() const;

    /// \brief Return a raw pointer to sparse counts.
    ///
    /// Returns nullptr for empty counted fingerprints.
    const std::uint32_t* CountData() const;

    /// \brief Return the sparse index data pointer as an integer address.
    std::uint64_t IndexDataAddress() const;

    /// \brief Return the sparse count data pointer as an integer address.
    std::uint64_t CountDataAddress() const;

private:
    FingerprintSpec spec_;
    std::vector<std::uint32_t> indices_;
    std::vector<std::uint32_t> counts_;

    void ValidateCountedSpec() const;
    void ValidateSparseStorage() const;
    void CheckRow(std::size_t row) const;
};

bool operator==(const OEFPCount& lhs, const OEFPCount& rhs);
bool operator!=(const OEFPCount& lhs, const OEFPCount& rhs);

/// \brief Sparse counted fingerprint with 64-bit sorted nonzero indices.
///
/// This is intended for raw sparse-count algorithms whose native identifier
/// domain exceeds uint32_t. Counts remain uint32_t to match existing counted
/// fingerprint storage and RDKit sparse-count value semantics.
class OEFPCount64 {
public:
    OEFPCount64() = default;

    /// \brief Construct an empty counted fingerprint.
    ///
    /// \param spec Fingerprint size and provenance metadata.
    /// \throws std::invalid_argument: When spec.value_type is not Counted.
    explicit OEFPCount64(FingerprintSpec spec);

    /// \brief Construct a counted fingerprint from sparse count arrays.
    ///
    /// \param spec Fingerprint size and provenance metadata.
    /// \param indices Sorted nonzero raw feature indices.
    /// \param counts Positive counts parallel to indices.
    /// \throws std::invalid_argument: When the spec or sparse arrays are invalid.
    /// \throws std::out_of_range: When an index is outside the fingerprint size.
    OEFPCount64(
        FingerprintSpec spec,
        std::vector<std::uint64_t> indices,
        std::vector<std::uint32_t> counts);

    /// \brief Return the fingerprint specification.
    const FingerprintSpec& Spec() const;

    /// \brief Return the raw fingerprint identifier domain size.
    std::uint64_t SizeBits() const;

    /// \brief Return the number of nonzero count entries.
    std::size_t NonzeroCount() const;

    /// \brief Return the sum of all sparse counts.
    std::uint64_t TotalCount() const;

    /// \brief Return read-only access to sparse indices.
    const std::vector<std::uint64_t>& Indices() const;

    /// \brief Return read-only access to sparse counts.
    const std::vector<std::uint32_t>& Counts() const;

    /// \brief Return one sparse index by row.
    ///
    /// \throws std::out_of_range: When row is greater than or equal to
    ///     NonzeroCount().
    std::uint64_t Index(std::size_t row) const;

    /// \brief Return one sparse count by row.
    ///
    /// \throws std::out_of_range: When row is greater than or equal to
    ///     NonzeroCount().
    std::uint32_t Count(std::size_t row) const;

    /// \brief Return a raw pointer to sparse indices.
    ///
    /// Returns nullptr for empty counted fingerprints.
    const std::uint64_t* IndexData() const;

    /// \brief Return a raw pointer to sparse counts.
    ///
    /// Returns nullptr for empty counted fingerprints.
    const std::uint32_t* CountData() const;

    /// \brief Return the sparse index data pointer as an integer address.
    std::uint64_t IndexDataAddress() const;

    /// \brief Return the sparse count data pointer as an integer address.
    std::uint64_t CountDataAddress() const;

private:
    FingerprintSpec spec_;
    std::vector<std::uint64_t> indices_;
    std::vector<std::uint32_t> counts_;

    void ValidateCountedSpec() const;
    void ValidateSparseStorage() const;
    void CheckRow(std::size_t row) const;
};

bool operator==(const OEFPCount64& lhs, const OEFPCount64& rhs);
bool operator!=(const OEFPCount64& lhs, const OEFPCount64& rhs);

} // namespace OEFP

#endif // OEFP_COUNT_H
