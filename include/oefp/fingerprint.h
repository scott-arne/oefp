#ifndef OEFP_FINGERPRINT_H
#define OEFP_FINGERPRINT_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace OEFP {

/// \brief Value representation used by a fingerprint.
enum class FingerprintValueType {
    Binary,
    Counted,
};

/// \brief Lightweight value-semantic description of a fingerprint.
///
/// The metadata fields are intentionally optional and descriptive. They provide
/// a stable place for future OpenEye or RDKit provenance without making the
/// dense fingerprint object responsible for annotations or feature mappings.
struct FingerprintSpec {
    std::uint64_t size_bits = 0;
    FingerprintValueType value_type = FingerprintValueType::Binary;
    std::string source_name;
    std::string source_type;
    std::string source_version;
    std::string parameters;
};

bool operator==(const FingerprintSpec& lhs, const FingerprintSpec& rhs);
bool operator!=(const FingerprintSpec& lhs, const FingerprintSpec& rhs);

/// \brief Return the number of uint64 words required for dense bit storage.
std::size_t DenseWordCount(std::uint64_t size_bits);

/// \brief Dense fixed-length binary fingerprint.
///
/// Bits are stored in uint64 words using little-endian bit numbering within
/// each word: bit index i maps to word i / 64 and mask 1ULL << (i % 64).
class OEFP {
public:
    /// \brief Construct a zero-initialized binary fingerprint.
    ///
    /// \param spec Fingerprint size and provenance metadata.
    /// \raises std::invalid_argument: When spec.value_type is not Binary.
    explicit OEFP(FingerprintSpec spec);

    /// \brief Construct a binary fingerprint from precomputed words.
    ///
    /// The supplied word vector must match the word count implied by the
    /// fingerprint size. Unused high bits in the final word are cleared.
    ///
    /// \param spec Fingerprint size and provenance metadata.
    /// \param words Dense uint64 storage words.
    /// \raises std::invalid_argument: When spec.value_type is not Binary or
    ///     words has an incompatible length.
    OEFP(FingerprintSpec spec, std::vector<std::uint64_t> words);

    /// \brief Return the fingerprint specification.
    const FingerprintSpec& Spec() const;

    /// \brief Return the fixed fingerprint size in bits.
    std::uint64_t SizeBits() const;

    /// \brief Return the number of uint64 storage words.
    std::size_t WordCount() const;

    /// \brief Return read-only access to the dense storage words.
    const std::vector<std::uint64_t>& Words() const;

    /// \brief Return one storage word by zero-based word index.
    ///
    /// \raises std::out_of_range: When word_index is greater than or equal to
    ///     WordCount().
    std::uint64_t Word(std::size_t word_index) const;

    /// \brief Set one storage word by zero-based word index.
    ///
    /// Writes through this method preserve the fixed-size invariant by masking
    /// unused high bits when the final word is changed.
    ///
    /// \raises std::out_of_range: When word_index is greater than or equal to
    ///     WordCount().
    void SetWord(std::size_t word_index, std::uint64_t value);

    /// \brief Set a bit by zero-based index.
    ///
    /// \raises std::out_of_range: When index is greater than or equal to
    ///     SizeBits().
    void SetBit(std::uint64_t index);

    /// \brief Clear a bit by zero-based index.
    ///
    /// \raises std::out_of_range: When index is greater than or equal to
    ///     SizeBits().
    void ClearBit(std::uint64_t index);

    /// \brief Test whether a bit is set by zero-based index.
    ///
    /// \raises std::out_of_range: When index is greater than or equal to
    ///     SizeBits().
    bool TestBit(std::uint64_t index) const;

    /// \brief Count set bits in the dense binary fingerprint.
    std::uint64_t CountOnBits() const;

    /// \brief Clear unused high bits in the final storage word.
    void MaskUnusedBits();

private:
    FingerprintSpec spec_;
    std::vector<std::uint64_t> words_;

    void ValidateBinarySpec() const;
    void CheckBitIndex(std::uint64_t index) const;
    void CheckWordIndex(std::size_t word_index) const;
};

bool operator==(const OEFP& lhs, const OEFP& rhs);
bool operator!=(const OEFP& lhs, const OEFP& rhs);

} // namespace OEFP

#endif // OEFP_FINGERPRINT_H
