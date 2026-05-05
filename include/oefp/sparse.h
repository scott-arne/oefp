#ifndef OEFP_SPARSE_H
#define OEFP_SPARSE_H

#include "oefp/fingerprint.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Sparse binary fingerprint with sorted on-bit identifiers.
///
/// Sparse binary fingerprints store only on-bit identifiers. The index array is
/// sorted and contains no duplicate indices.
class OEFPSparse {
public:
    OEFPSparse() = default;

    /// \brief Construct an empty sparse binary fingerprint.
    ///
    /// \param spec Fingerprint size and provenance metadata.
    /// \throws std::invalid_argument: When spec.value_type is not Binary.
    explicit OEFPSparse(FingerprintSpec spec);

    /// \brief Construct a sparse binary fingerprint from on-bit indices.
    ///
    /// \param spec Fingerprint size and provenance metadata.
    /// \param indices Sorted on-bit identifiers.
    /// \throws std::invalid_argument: When the spec or sparse indices are
    ///     invalid.
    /// \throws std::out_of_range: When an index is outside the fingerprint
    ///     size.
    OEFPSparse(FingerprintSpec spec, std::vector<std::uint32_t> indices);

    /// \brief Return the fingerprint specification.
    const FingerprintSpec& Spec() const;

    /// \brief Return the fixed sparse fingerprint size.
    std::uint64_t SizeBits() const;

    /// \brief Return the number of on-bit entries.
    std::size_t CountOnBits() const;

    /// \brief Return read-only access to sorted on-bit identifiers.
    const std::vector<std::uint32_t>& Indices() const;

    /// \brief Return one on-bit identifier by row.
    ///
    /// \throws std::out_of_range: When row is greater than or equal to
    ///     CountOnBits().
    std::uint32_t Index(std::size_t row) const;

    /// \brief Return a raw pointer to sparse on-bit identifiers.
    ///
    /// Returns nullptr for empty sparse fingerprints.
    const std::uint32_t* IndexData() const;

    /// \brief Return the sparse index data pointer as an integer address.
    std::uint64_t IndexDataAddress() const;

private:
    FingerprintSpec spec_;
    std::vector<std::uint32_t> indices_;

    void ValidateBinarySpec() const;
    void ValidateSparseStorage() const;
    void CheckRow(std::size_t row) const;
};

bool operator==(const OEFPSparse& lhs, const OEFPSparse& rhs);
bool operator!=(const OEFPSparse& lhs, const OEFPSparse& rhs);

} // namespace OEFP

#endif // OEFP_SPARSE_H
