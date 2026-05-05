#include "oefp/fingerprint.h"

#include <stdexcept>
#include <utility>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace OEFP {
namespace {

constexpr std::uint64_t BITS_PER_WORD = 64;

std::uint64_t count_bits(std::uint64_t word) {
#if defined(_MSC_VER)
    return static_cast<std::uint64_t>(__popcnt64(word));
#elif defined(__clang__) || defined(__GNUC__)
    return static_cast<std::uint64_t>(
        __builtin_popcountll(static_cast<unsigned long long>(word)));
#else
    std::uint64_t count = 0;
    while (word != 0) {
        word &= word - 1;
        ++count;
    }
    return count;
#endif
}

FingerprintSpec validate_binary_spec(FingerprintSpec spec) {
    if (spec.value_type != FingerprintValueType::Binary) {
        throw std::invalid_argument("OEFP currently supports binary fingerprints only.");
    }
    return spec;
}

} // namespace

bool operator==(const FingerprintSpec& lhs, const FingerprintSpec& rhs) {
    return lhs.size_bits == rhs.size_bits && lhs.value_type == rhs.value_type
        && lhs.source_name == rhs.source_name && lhs.source_type == rhs.source_type
        && lhs.source_version == rhs.source_version && lhs.parameters == rhs.parameters;
}

bool operator!=(const FingerprintSpec& lhs, const FingerprintSpec& rhs) {
    return !(lhs == rhs);
}

std::size_t DenseWordCount(std::uint64_t size_bits) {
    return static_cast<std::size_t>(
        (size_bits / BITS_PER_WORD) + (size_bits % BITS_PER_WORD != 0 ? 1 : 0));
}

OEFP::OEFP(FingerprintSpec spec)
    : spec_(validate_binary_spec(std::move(spec))),
      words_(DenseWordCount(spec_.size_bits), 0ULL) {
}

OEFP::OEFP(FingerprintSpec spec, std::vector<std::uint64_t> words)
    : spec_(std::move(spec)),
      words_(std::move(words)) {
    ValidateBinarySpec();
    if (words_.size() != DenseWordCount(spec_.size_bits)) {
        throw std::invalid_argument("Fingerprint word count does not match size_bits.");
    }
    MaskUnusedBits();
}

const FingerprintSpec& OEFP::Spec() const {
    return spec_;
}

std::uint64_t OEFP::SizeBits() const {
    return spec_.size_bits;
}

std::size_t OEFP::WordCount() const {
    return words_.size();
}

const std::vector<std::uint64_t>& OEFP::Words() const {
    return words_;
}

std::uint64_t OEFP::Word(std::size_t word_index) const {
    CheckWordIndex(word_index);
    return words_[word_index];
}

void OEFP::SetWord(std::size_t word_index, std::uint64_t value) {
    CheckWordIndex(word_index);
    words_[word_index] = value;
    if (word_index + 1 == words_.size()) {
        MaskUnusedBits();
    }
}

void OEFP::SetBit(std::uint64_t index) {
    CheckBitIndex(index);
    words_[index / BITS_PER_WORD] |= 1ULL << (index % BITS_PER_WORD);
}

void OEFP::ClearBit(std::uint64_t index) {
    CheckBitIndex(index);
    words_[index / BITS_PER_WORD] &= ~(1ULL << (index % BITS_PER_WORD));
}

bool OEFP::TestBit(std::uint64_t index) const {
    CheckBitIndex(index);
    return (words_[index / BITS_PER_WORD] & (1ULL << (index % BITS_PER_WORD))) != 0;
}

std::uint64_t OEFP::CountOnBits() const {
    std::uint64_t count = 0;
    for (const auto word : words_) {
        count += count_bits(word);
    }
    return count;
}

void OEFP::MaskUnusedBits() {
    const auto used_bits = spec_.size_bits % BITS_PER_WORD;
    if (used_bits == 0 || words_.empty()) {
        return;
    }

    const auto final_word_mask = (1ULL << used_bits) - 1ULL;
    words_.back() &= final_word_mask;
}

void OEFP::ValidateBinarySpec() const {
    if (spec_.value_type != FingerprintValueType::Binary) {
        throw std::invalid_argument("OEFP currently supports binary fingerprints only.");
    }
}

void OEFP::CheckBitIndex(std::uint64_t index) const {
    if (index >= spec_.size_bits) {
        throw std::out_of_range("Fingerprint bit index is out of range.");
    }
}

void OEFP::CheckWordIndex(std::size_t word_index) const {
    if (word_index >= words_.size()) {
        throw std::out_of_range("Fingerprint word index is out of range.");
    }
}

bool operator==(const OEFP& lhs, const OEFP& rhs) {
    return lhs.Spec() == rhs.Spec() && lhs.Words() == rhs.Words();
}

bool operator!=(const OEFP& lhs, const OEFP& rhs) {
    return !(lhs == rhs);
}

} // namespace OEFP
