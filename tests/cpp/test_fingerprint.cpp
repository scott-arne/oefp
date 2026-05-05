#include <gtest/gtest.h>

#include "oefp/oefp.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace OEFP {
namespace test {

FingerprintSpec binary_spec(std::uint64_t size_bits) {
    FingerprintSpec spec;
    spec.size_bits = size_bits;
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "unit-test";
    spec.source_type = "dense";
    spec.source_version = "1";
    spec.parameters = "size=" + std::to_string(size_bits);
    return spec;
}

FingerprintSpec counted_spec(std::uint64_t size_bits) {
    auto spec = binary_spec(size_bits);
    spec.value_type = FingerprintValueType::Counted;
    return spec;
}

TEST(FingerprintTest, DenseWordCountReturnsExpectedStorageWidth) {
    EXPECT_EQ(DenseWordCount(0), 0u);
    EXPECT_EQ(DenseWordCount(1), 1u);
    EXPECT_EQ(DenseWordCount(64), 1u);
    EXPECT_EQ(DenseWordCount(65), 2u);
    EXPECT_EQ(DenseWordCount(128), 2u);
}

TEST(FingerprintTest, InitializesZeroWordsForBoundarySizes) {
    const std::vector<std::pair<std::uint64_t, std::size_t>> cases = {
        {0, 0},
        {1, 1},
        {64, 1},
        {65, 2},
        {128, 2},
    };

    for (const auto& [size_bits, word_count] : cases) {
        OEFP fingerprint(binary_spec(size_bits));

        EXPECT_EQ(fingerprint.SizeBits(), size_bits);
        EXPECT_EQ(fingerprint.WordCount(), word_count);
        EXPECT_EQ(fingerprint.Words().size(), word_count);
        for (const auto word : fingerprint.Words()) {
            EXPECT_EQ(word, 0ULL);
        }
    }
}

TEST(FingerprintTest, DefaultConstructsAsEmptyBinaryFingerprint) {
    const OEFP fingerprint;

    EXPECT_EQ(fingerprint.SizeBits(), 0u);
    EXPECT_EQ(fingerprint.WordCount(), 0u);
    EXPECT_EQ(fingerprint.CountOnBits(), 0u);
    EXPECT_EQ(fingerprint.WordData(), nullptr);
}

TEST(FingerprintTest, SetsTestsAndClearsBitsAcrossWordBoundaries) {
    OEFP fingerprint(binary_spec(130));

    fingerprint.SetBit(0);
    fingerprint.SetBit(63);
    fingerprint.SetBit(64);
    fingerprint.SetBit(129);

    EXPECT_TRUE(fingerprint.TestBit(0));
    EXPECT_TRUE(fingerprint.TestBit(63));
    EXPECT_TRUE(fingerprint.TestBit(64));
    EXPECT_TRUE(fingerprint.TestBit(129));
    EXPECT_FALSE(fingerprint.TestBit(1));
    EXPECT_FALSE(fingerprint.TestBit(128));

    EXPECT_EQ(fingerprint.Words()[0], (1ULL << 0) | (1ULL << 63));
    EXPECT_EQ(fingerprint.Words()[1], 1ULL);
    EXPECT_EQ(fingerprint.Words()[2], 1ULL << 1);

    fingerprint.ClearBit(63);
    fingerprint.ClearBit(64);

    EXPECT_TRUE(fingerprint.TestBit(0));
    EXPECT_FALSE(fingerprint.TestBit(63));
    EXPECT_FALSE(fingerprint.TestBit(64));
    EXPECT_TRUE(fingerprint.TestBit(129));
}

TEST(FingerprintTest, BitOperationsRejectOutOfRangeIndexes) {
    OEFP empty(binary_spec(0));
    OEFP fingerprint(binary_spec(3));

    EXPECT_THROW(empty.SetBit(0), std::out_of_range);
    EXPECT_THROW(empty.ClearBit(0), std::out_of_range);
    EXPECT_THROW(empty.TestBit(0), std::out_of_range);
    EXPECT_THROW(fingerprint.SetBit(3), std::out_of_range);
    EXPECT_THROW(fingerprint.ClearBit(3), std::out_of_range);
    EXPECT_THROW(fingerprint.TestBit(3), std::out_of_range);
}

TEST(FingerprintTest, WordAccessRejectsOutOfRangeIndexes) {
    OEFP fingerprint(binary_spec(65));

    EXPECT_THROW(static_cast<void>(fingerprint.Word(2)), std::out_of_range);
    EXPECT_THROW(fingerprint.SetWord(2, 0ULL), std::out_of_range);
}

TEST(FingerprintTest, ConstructorValidatesWordCount) {
    EXPECT_THROW(OEFP(binary_spec(65), std::vector<std::uint64_t>{0ULL}), std::invalid_argument);
    EXPECT_THROW(
        OEFP(binary_spec(65), std::vector<std::uint64_t>{0ULL, 0ULL, 0ULL}),
        std::invalid_argument);

    EXPECT_NO_THROW(OEFP(binary_spec(65), std::vector<std::uint64_t>{0ULL, 0ULL}));
}

TEST(FingerprintTest, ConstructorsRejectCountedSpecs) {
    EXPECT_THROW(OEFP(counted_spec(64)), std::invalid_argument);
    EXPECT_THROW(OEFP(counted_spec(64), std::vector<std::uint64_t>{0ULL}), std::invalid_argument);
}

TEST(FingerprintTest, MasksUnusedHighBitsInFinalWord) {
    OEFP fingerprint(binary_spec(65), std::vector<std::uint64_t>{~0ULL, ~0ULL});

    EXPECT_EQ(fingerprint.Word(0), ~0ULL);
    EXPECT_EQ(fingerprint.Word(1), 1ULL);

    fingerprint.SetWord(1, ~0ULL);

    EXPECT_EQ(fingerprint.Word(1), 1ULL);
    EXPECT_EQ(fingerprint.Words()[1], 1ULL);
}

TEST(FingerprintTest, SetsAndReadsWordsWithFinalWordMasking) {
    OEFP fingerprint(binary_spec(130));

    fingerprint.SetWord(0, ~0ULL);
    fingerprint.SetWord(1, 0x1234ULL);
    fingerprint.SetWord(2, ~0ULL);

    EXPECT_EQ(fingerprint.Word(0), ~0ULL);
    EXPECT_EQ(fingerprint.Word(1), 0x1234ULL);
    EXPECT_EQ(fingerprint.Word(2), 0x3ULL);
}

TEST(FingerprintTest, CountsOnBits) {
    OEFP fingerprint(binary_spec(130));

    EXPECT_EQ(fingerprint.CountOnBits(), 0U);

    fingerprint.SetBit(0);
    fingerprint.SetBit(63);
    fingerprint.SetBit(64);
    fingerprint.SetBit(129);

    EXPECT_EQ(fingerprint.CountOnBits(), 4U);

    fingerprint.ClearBit(64);

    EXPECT_EQ(fingerprint.CountOnBits(), 3U);
}

TEST(FingerprintTest, ComparesSpecsAndWordsForEquality) {
    OEFP first(binary_spec(65));
    OEFP same(binary_spec(65));
    OEFP different_size(binary_spec(64));

    auto different_metadata_spec = binary_spec(65);
    different_metadata_spec.source_version = "2";
    OEFP different_metadata(different_metadata_spec);

    first.SetBit(64);
    same.SetBit(64);

    EXPECT_EQ(first, same);
    EXPECT_NE(first, different_size);
    EXPECT_NE(first, different_metadata);

    same.ClearBit(64);
    EXPECT_NE(first, same);
}

} // namespace test
} // namespace OEFP
