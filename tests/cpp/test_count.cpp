#include <gtest/gtest.h>

#include "oefp/count.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace OEFP {
namespace test {
namespace {

FingerprintSpec counted_spec(std::uint64_t size_bits = 128) {
    FingerprintSpec spec;
    spec.size_bits = size_bits;
    spec.value_type = FingerprintValueType::Counted;
    spec.source_name = "test";
    spec.source_type = "count";
    return spec;
}

FingerprintSpec binary_spec() {
    auto spec = counted_spec();
    spec.value_type = FingerprintValueType::Binary;
    return spec;
}

} // namespace

TEST(CountFingerprintTest, StoresSparseCountsAndTotals) {
    const OEFPCount fp(counted_spec(), {3u, 7u, 64u}, {2u, 5u, 1u});

    EXPECT_EQ(fp.SizeBits(), 128u);
    EXPECT_EQ(fp.NonzeroCount(), 3u);
    EXPECT_EQ(fp.TotalCount(), 8u);
    EXPECT_EQ(fp.Index(0), 3u);
    EXPECT_EQ(fp.Index(1), 7u);
    EXPECT_EQ(fp.Index(2), 64u);
    EXPECT_EQ(fp.Count(0), 2u);
    EXPECT_EQ(fp.Count(1), 5u);
    EXPECT_EQ(fp.Count(2), 1u);
    EXPECT_EQ(fp.Indices(), std::vector<std::uint32_t>({3u, 7u, 64u}));
    EXPECT_EQ(fp.Counts(), std::vector<std::uint32_t>({2u, 5u, 1u}));
    EXPECT_NE(fp.IndexData(), nullptr);
    EXPECT_NE(fp.CountData(), nullptr);
}

TEST(CountFingerprintTest, EmptyCountFingerprintKeepsSpecAndNullPointers) {
    const OEFPCount fp(counted_spec(64));

    EXPECT_EQ(fp.SizeBits(), 64u);
    EXPECT_EQ(fp.NonzeroCount(), 0u);
    EXPECT_EQ(fp.TotalCount(), 0u);
    EXPECT_EQ(fp.IndexData(), nullptr);
    EXPECT_EQ(fp.CountData(), nullptr);
    EXPECT_EQ(fp.IndexDataAddress(), 0u);
    EXPECT_EQ(fp.CountDataAddress(), 0u);
}

TEST(CountFingerprintTest, RejectsInvalidSparseCountStorage) {
    EXPECT_THROW((void)OEFPCount{binary_spec()}, std::invalid_argument);
    EXPECT_THROW(OEFPCount(counted_spec(), {1u}, {1u, 2u}), std::invalid_argument);
    EXPECT_THROW(OEFPCount(counted_spec(), {1u, 1u}, {1u, 2u}), std::invalid_argument);
    EXPECT_THROW(OEFPCount(counted_spec(), {2u, 1u}, {1u, 2u}), std::invalid_argument);
    EXPECT_THROW(OEFPCount(counted_spec(), {1u}, {0u}), std::invalid_argument);
    EXPECT_THROW(OEFPCount(counted_spec(4), {4u}, {1u}), std::out_of_range);
}

TEST(CountFingerprintTest, RowAccessRejectsOutOfRangeRows) {
    const OEFPCount fp(counted_spec(), {3u}, {2u});

    EXPECT_THROW(fp.Index(1), std::out_of_range);
    EXPECT_THROW(fp.Count(1), std::out_of_range);
}

TEST(CountFingerprint64Test, StoresSparseCountsAboveUint32Range) {
    const auto large_index = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 17u;
    const OEFPCount64 fp(
        counted_spec(1ULL << 40u),
        std::vector<std::uint64_t>{3u, large_index},
        std::vector<std::uint32_t>{2u, 5u});

    EXPECT_EQ(fp.SizeBits(), 1ULL << 40u);
    EXPECT_EQ(fp.NonzeroCount(), 2u);
    EXPECT_EQ(fp.TotalCount(), 7u);
    EXPECT_EQ(fp.Index(0), 3u);
    EXPECT_EQ(fp.Index(1), large_index);
    EXPECT_EQ(fp.Count(0), 2u);
    EXPECT_EQ(fp.Count(1), 5u);
    EXPECT_EQ(fp.Indices(), std::vector<std::uint64_t>({3u, large_index}));
    EXPECT_EQ(fp.Counts(), std::vector<std::uint32_t>({2u, 5u}));
    EXPECT_NE(fp.IndexData(), nullptr);
    EXPECT_NE(fp.CountData(), nullptr);
}

TEST(CountFingerprint64Test, EmptyCountFingerprintKeepsSpecAndNullPointers) {
    const OEFPCount64 fp(counted_spec(1ULL << 36u));

    EXPECT_EQ(fp.SizeBits(), 1ULL << 36u);
    EXPECT_EQ(fp.NonzeroCount(), 0u);
    EXPECT_EQ(fp.TotalCount(), 0u);
    EXPECT_EQ(fp.IndexData(), nullptr);
    EXPECT_EQ(fp.CountData(), nullptr);
    EXPECT_EQ(fp.IndexDataAddress(), 0u);
    EXPECT_EQ(fp.CountDataAddress(), 0u);
}

TEST(CountFingerprint64Test, RejectsInvalidSparseCountStorage) {
    EXPECT_THROW((void)OEFPCount64{binary_spec()}, std::invalid_argument);
    EXPECT_THROW(
        OEFPCount64(
            counted_spec(),
            std::vector<std::uint64_t>{1u},
            std::vector<std::uint32_t>{1u, 2u}),
        std::invalid_argument);
    EXPECT_THROW(
        OEFPCount64(
            counted_spec(),
            std::vector<std::uint64_t>{1u, 1u},
            std::vector<std::uint32_t>{1u, 2u}),
        std::invalid_argument);
    EXPECT_THROW(
        OEFPCount64(
            counted_spec(),
            std::vector<std::uint64_t>{2u, 1u},
            std::vector<std::uint32_t>{1u, 2u}),
        std::invalid_argument);
    EXPECT_THROW(
        OEFPCount64(
            counted_spec(),
            std::vector<std::uint64_t>{1u},
            std::vector<std::uint32_t>{0u}),
        std::invalid_argument);
    EXPECT_THROW(
        OEFPCount64(
            counted_spec(4),
            std::vector<std::uint64_t>{4u},
            std::vector<std::uint32_t>{1u}),
        std::out_of_range);
}

TEST(CountFingerprint64Test, RowAccessRejectsOutOfRangeRows) {
    const OEFPCount64 fp(
        counted_spec(),
        std::vector<std::uint64_t>{3u},
        std::vector<std::uint32_t>{2u});

    EXPECT_THROW(fp.Index(1), std::out_of_range);
    EXPECT_THROW(fp.Count(1), std::out_of_range);
}

} // namespace test
} // namespace OEFP
