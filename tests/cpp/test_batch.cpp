#include <gtest/gtest.h>

#include "oefp/batch.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {
namespace {

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
} // namespace

TEST(OEFPBatchTest, BuildsContiguousRowsFromFingerprints) {
    auto spec = binary_spec(128);
    auto a = OEFP(spec);
    a.SetBit(0);
    a.SetBit(64);
    auto b = OEFP(spec);
    b.SetBit(1);
    b.SetBit(65);
    b.SetBit(127);

    OEFPBatch batch = OEFPBatch::FromFingerprints({a, b});

    EXPECT_EQ(batch.Spec(), spec);
    EXPECT_EQ(batch.Size(), 2u);
    EXPECT_EQ(batch.SizeBits(), 128u);
    EXPECT_EQ(batch.WordsPerFingerprint(), 2u);
    EXPECT_EQ(batch.WordCount(), 4u);
    EXPECT_EQ(batch.PopCount(0), 2u);
    EXPECT_EQ(batch.PopCount(1), 3u);
    EXPECT_EQ(batch.RowWords(0)[0], a.Words()[0]);
    EXPECT_EQ(batch.RowWords(0)[1], a.Words()[1]);
    EXPECT_EQ(batch.RowWords(1)[0], b.Words()[0]);
    EXPECT_EQ(batch.RowWords(1)[1], b.Words()[1]);
    EXPECT_EQ(batch.RowWords(1), batch.WordData() + 2);
    EXPECT_EQ(batch.PopCounts(), (std::vector<std::uint32_t>{2u, 3u}));
}

TEST(OEFPBatchTest, AppendsToExplicitSpecEmptyBatch) {
    auto spec = binary_spec(128);
    OEFP fp(spec);
    fp.SetBit(7);
    fp.SetBit(70);
    OEFPBatch batch(spec);

    batch.Append(fp);

    EXPECT_EQ(batch.Size(), 1u);
    EXPECT_EQ(batch.WordsPerFingerprint(), 2u);
    EXPECT_EQ(batch.WordCount(), 2u);
    EXPECT_EQ(batch.PopCount(0), 2u);
    EXPECT_EQ(batch.RowWords(0)[0], fp.Words()[0]);
    EXPECT_EQ(batch.RowWords(0)[1], fp.Words()[1]);
}

TEST(OEFPBatchTest, EmptyBatchKeepsSpecAndHasNoRows) {
    auto spec = binary_spec(256);

    OEFPBatch batch(spec);

    EXPECT_EQ(batch.Spec(), spec);
    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.SizeBits(), 256u);
    EXPECT_EQ(batch.WordsPerFingerprint(), 4u);
    EXPECT_EQ(batch.WordCount(), 0u);
    EXPECT_TRUE(batch.Words().empty());
    EXPECT_TRUE(batch.PopCounts().empty());
}

TEST(OEFPBatchTest, ExplicitZeroBitEmptyBatchIsAllowed) {
    auto spec = binary_spec(0);

    OEFPBatch batch(spec);

    EXPECT_EQ(batch.Spec(), spec);
    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.SizeBits(), 0u);
    EXPECT_EQ(batch.WordsPerFingerprint(), 0u);
    EXPECT_EQ(batch.WordCount(), 0u);
}

TEST(OEFPBatchTest, EmptyFromFingerprintsReturnsDefaultBatch) {
    OEFPBatch batch = OEFPBatch::FromFingerprints({});

    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.SizeBits(), 0u);
    EXPECT_EQ(batch.WordsPerFingerprint(), 0u);
    EXPECT_EQ(batch.WordCount(), 0u);
    EXPECT_TRUE(batch.Words().empty());
    EXPECT_TRUE(batch.PopCounts().empty());
}

TEST(OEFPBatchTest, AppendRejectsZeroWidthFingerprints) {
    auto spec = binary_spec(0);
    OEFP fp(spec);
    OEFPBatch batch(spec);

    EXPECT_THROW(batch.Append(fp), std::invalid_argument);
    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.WordCount(), 0u);
    EXPECT_TRUE(batch.PopCounts().empty());
}

TEST(OEFPBatchTest, FromFingerprintsRejectsZeroWidthFingerprints) {
    OEFP fp(binary_spec(0));

    EXPECT_THROW(OEFPBatch::FromFingerprints({fp}), std::invalid_argument);
}

TEST(OEFPBatchTest, AppendRejectsMismatchedSpecs) {
    OEFPBatch batch(binary_spec(64));
    OEFP mismatched(binary_spec(128));

    EXPECT_THROW(batch.Append(mismatched), std::invalid_argument);
}

TEST(OEFPBatchTest, AppendRejectsMetadataMismatchWithSameBitSize) {
    auto spec = binary_spec(64);
    auto mismatched_spec = spec;
    mismatched_spec.source_version = "2";
    OEFPBatch batch(spec);
    OEFP mismatched(mismatched_spec);

    EXPECT_THROW(batch.Append(mismatched), std::invalid_argument);
}

TEST(OEFPBatchTest, FromFingerprintsRejectsMismatchedSpecs) {
    auto a = OEFP(binary_spec(64));
    auto b = OEFP(binary_spec(128));

    EXPECT_THROW(OEFPBatch::FromFingerprints({a, b}), std::invalid_argument);
}

TEST(OEFPBatchTest, FromFingerprintsRejectsMetadataMismatchWithSameBitSize) {
    auto spec = binary_spec(64);
    auto mismatched_spec = spec;
    mismatched_spec.parameters = "different";
    auto a = OEFP(spec);
    auto b = OEFP(mismatched_spec);

    EXPECT_THROW(OEFPBatch::FromFingerprints({a, b}), std::invalid_argument);
}

TEST(OEFPBatchTest, RowAccessRejectsOutOfRangeRows) {
    OEFPBatch empty(binary_spec(64));
    OEFPBatch batch = OEFPBatch::FromFingerprints({OEFP(binary_spec(64))});

    EXPECT_THROW(static_cast<void>(empty.RowWords(0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(empty.PopCount(0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(batch.RowWords(1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(batch.PopCount(1)), std::out_of_range);
}

TEST(OEFPBatchTest, PointerHelpersExposeEmptyAndNonEmptyStorage) {
    OEFPBatch empty(binary_spec(64));

    EXPECT_EQ(empty.WordData(), nullptr);
    EXPECT_EQ(empty.PopCountData(), nullptr);
    EXPECT_EQ(empty.WordDataAddress(), 0u);
    EXPECT_EQ(empty.PopCountDataAddress(), 0u);

    OEFP fp(binary_spec(64));
    fp.SetBit(3);
    OEFPBatch batch = OEFPBatch::FromFingerprints({fp});

    EXPECT_EQ(batch.WordData(), batch.Words().data());
    EXPECT_EQ(batch.PopCountData(), batch.PopCounts().data());
    EXPECT_EQ(batch.WordDataAddress(), reinterpret_cast<std::uintptr_t>(batch.WordData()));
    EXPECT_EQ(batch.PopCountDataAddress(), reinterpret_cast<std::uintptr_t>(batch.PopCountData()));
}

} // namespace test
} // namespace OEFP
