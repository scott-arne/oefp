#include <gtest/gtest.h>

#include "oefp/sparse_batch.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {
namespace {

FingerprintSpec sparse_binary_spec(
    std::uint64_t size_bits = std::numeric_limits<std::uint64_t>::max()) {
    FingerprintSpec spec;
    spec.size_bits = size_bits;
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "unit-test";
    spec.source_type = "sparse-binary";
    spec.source_version = "1";
    spec.parameters = "size=" + std::to_string(size_bits);
    return spec;
}

FingerprintSpec counted_spec() {
    auto spec = sparse_binary_spec(128);
    spec.value_type = FingerprintValueType::Counted;
    return spec;
}

OEFPSparse sparse_fp(
    std::vector<std::uint32_t> indices,
    std::uint64_t size_bits = std::numeric_limits<std::uint64_t>::max()) {
    return OEFPSparse(sparse_binary_spec(size_bits), std::move(indices));
}

} // namespace

TEST(OEFPSparseBatchTest, BuildsContiguousSparseRowsFromSparseFingerprints) {
    const auto first = sparse_fp({1u, 7u});
    const auto second = sparse_fp({0u, 7u, 64u});
    const OEFPSparse empty(sparse_binary_spec());

    const auto batch = OEFPSparseBatch::FromFingerprints({first, second, empty});

    EXPECT_EQ(batch.Size(), 3u);
    EXPECT_EQ(batch.SizeBits(), std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(batch.EntryCount(), 5u);
    EXPECT_EQ(batch.RowEntryCount(0), 2u);
    EXPECT_EQ(batch.RowEntryCount(1), 3u);
    EXPECT_EQ(batch.RowEntryCount(2), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
    EXPECT_EQ(batch.RowOffset(1), 2u);
    EXPECT_EQ(batch.RowOffset(2), 5u);
    EXPECT_EQ(batch.RowOffset(3), 5u);
    EXPECT_EQ(batch.Indices(), std::vector<std::uint32_t>({1u, 7u, 0u, 7u, 64u}));
    EXPECT_EQ(batch.RowIndices(1)[0], 0u);
    EXPECT_EQ(batch.RowIndices(1)[2], 64u);
    EXPECT_NE(batch.IndexData(), nullptr);
    EXPECT_NE(batch.RowOffsetData(), nullptr);
}

TEST(OEFPSparseBatchTest, EmptyBatchKeepsSpecAndSingleZeroOffset) {
    const OEFPSparseBatch batch(sparse_binary_spec(256));

    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.SizeBits(), 256u);
    EXPECT_EQ(batch.EntryCount(), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
    EXPECT_TRUE(batch.Indices().empty());
    EXPECT_EQ(batch.IndexData(), nullptr);
    EXPECT_NE(batch.RowOffsetData(), nullptr);
    EXPECT_NE(batch.RowOffsetDataAddress(), 0u);
}

TEST(OEFPSparseBatchTest, EmptyFromFingerprintsReturnsDefaultBatch) {
    const auto batch = OEFPSparseBatch::FromFingerprints({});

    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.SizeBits(), 0u);
    EXPECT_EQ(batch.EntryCount(), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
}

TEST(OEFPSparseBatchTest, RejectsInvalidOrMismatchedSparseFingerprints) {
    OEFPSparseBatch batch(sparse_binary_spec());
    batch.Append(sparse_fp({1u}));

    EXPECT_THROW(batch.Append(sparse_fp({1u}, 64)), std::invalid_argument);

    auto metadata_spec = sparse_binary_spec();
    metadata_spec.source_version = "2";
    EXPECT_THROW(batch.Append(OEFPSparse(metadata_spec, {1u})), std::invalid_argument);

    EXPECT_THROW(static_cast<void>(OEFPSparseBatch{counted_spec()}), std::invalid_argument);
    EXPECT_THROW(batch.Append(OEFPSparse(sparse_binary_spec(0))), std::invalid_argument);
}

TEST(OEFPSparseBatchTest, RowAccessRejectsOutOfRangeRows) {
    const auto batch = OEFPSparseBatch::FromFingerprints({sparse_fp({1u})});

    EXPECT_THROW(batch.RowOffset(2), std::out_of_range);
    EXPECT_THROW(batch.RowEntryCount(1), std::out_of_range);
    EXPECT_THROW(batch.RowIndices(1), std::out_of_range);
}

} // namespace test
} // namespace OEFP
