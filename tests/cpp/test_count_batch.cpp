#include <gtest/gtest.h>

#include "oefp/count_batch.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {
namespace {

FingerprintSpec counted_spec(std::uint64_t size_bits = 128) {
    FingerprintSpec spec;
    spec.size_bits = size_bits;
    spec.value_type = FingerprintValueType::Counted;
    spec.source_name = "unit-test";
    spec.source_type = "count";
    spec.source_version = "1";
    spec.parameters = "size=" + std::to_string(size_bits);
    return spec;
}

FingerprintSpec binary_spec() {
    auto spec = counted_spec();
    spec.value_type = FingerprintValueType::Binary;
    return spec;
}

OEFPCount count_fp(
    std::vector<std::uint32_t> indices,
    std::vector<std::uint32_t> counts,
    std::uint64_t size_bits = 128) {
    return OEFPCount(counted_spec(size_bits), std::move(indices), std::move(counts));
}

} // namespace

TEST(OEFPCountBatchTest, BuildsContiguousSparseRowsFromCountFingerprints) {
    const auto first = count_fp({1u, 7u}, {2u, 1u});
    const auto second = count_fp({0u, 7u, 64u}, {3u, 4u, 5u});
    const auto empty = OEFPCount(counted_spec());

    const auto batch = OEFPCountBatch::FromFingerprints({first, second, empty});

    EXPECT_EQ(batch.Size(), 3u);
    EXPECT_EQ(batch.SizeBits(), 128u);
    EXPECT_EQ(batch.EntryCount(), 5u);
    EXPECT_EQ(batch.RowEntryCount(0), 2u);
    EXPECT_EQ(batch.RowEntryCount(1), 3u);
    EXPECT_EQ(batch.RowEntryCount(2), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
    EXPECT_EQ(batch.RowOffset(1), 2u);
    EXPECT_EQ(batch.RowOffset(2), 5u);
    EXPECT_EQ(batch.RowOffset(3), 5u);
    EXPECT_EQ(batch.Indices(), std::vector<std::uint32_t>({1u, 7u, 0u, 7u, 64u}));
    EXPECT_EQ(batch.Counts(), std::vector<std::uint32_t>({2u, 1u, 3u, 4u, 5u}));
    EXPECT_EQ(batch.RowIndices(1)[0], 0u);
    EXPECT_EQ(batch.RowCounts(1)[2], 5u);
    EXPECT_NE(batch.IndexData(), nullptr);
    EXPECT_NE(batch.CountData(), nullptr);
    EXPECT_NE(batch.RowOffsetData(), nullptr);
}

TEST(OEFPCountBatchTest, EmptyBatchKeepsSpecAndSingleZeroOffset) {
    const OEFPCountBatch batch(counted_spec(256));

    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.SizeBits(), 256u);
    EXPECT_EQ(batch.EntryCount(), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
    EXPECT_TRUE(batch.Indices().empty());
    EXPECT_TRUE(batch.Counts().empty());
    EXPECT_EQ(batch.IndexData(), nullptr);
    EXPECT_EQ(batch.CountData(), nullptr);
    EXPECT_NE(batch.RowOffsetData(), nullptr);
    EXPECT_NE(batch.RowOffsetDataAddress(), 0u);
}

TEST(OEFPCountBatchTest, EmptyFromFingerprintsReturnsDefaultBatch) {
    const auto batch = OEFPCountBatch::FromFingerprints({});

    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.SizeBits(), 0u);
    EXPECT_EQ(batch.EntryCount(), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
}

TEST(OEFPCountBatchTest, RejectsInvalidOrMismatchedCountFingerprints) {
    OEFPCountBatch batch(counted_spec());
    batch.Append(count_fp({1u}, {2u}));

    EXPECT_THROW(batch.Append(count_fp({1u}, {2u}, 64)), std::invalid_argument);

    auto metadata_spec = counted_spec();
    metadata_spec.source_version = "2";
    EXPECT_THROW(batch.Append(OEFPCount(metadata_spec, {1u}, {2u})), std::invalid_argument);

    EXPECT_THROW(static_cast<void>(OEFPCountBatch{binary_spec()}), std::invalid_argument);
    EXPECT_THROW(batch.Append(OEFPCount(counted_spec(0))), std::invalid_argument);
}

TEST(OEFPCountBatchTest, RowAccessRejectsOutOfRangeRows) {
    const auto batch = OEFPCountBatch::FromFingerprints({count_fp({1u}, {2u})});

    EXPECT_THROW(batch.RowOffset(2), std::out_of_range);
    EXPECT_THROW(batch.RowEntryCount(1), std::out_of_range);
    EXPECT_THROW(batch.RowIndices(1), std::out_of_range);
    EXPECT_THROW(batch.RowCounts(1), std::out_of_range);
}

} // namespace test
} // namespace OEFP
