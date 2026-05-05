#include <gtest/gtest.h>

#include "oefp/sparse.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace OEFP {
namespace test {
namespace {

FingerprintSpec sparse_binary_spec(std::uint64_t size_bits = std::numeric_limits<std::uint64_t>::max()) {
    FingerprintSpec spec;
    spec.size_bits = size_bits;
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "test";
    spec.source_type = "sparse";
    return spec;
}

FingerprintSpec counted_spec() {
    auto spec = sparse_binary_spec(128);
    spec.value_type = FingerprintValueType::Counted;
    return spec;
}

} // namespace

TEST(SparseFingerprintTest, StoresSparseOnBits) {
    const OEFPSparse fp(sparse_binary_spec(), {3u, 7u, 64u});

    EXPECT_EQ(fp.SizeBits(), std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(fp.CountOnBits(), 3u);
    EXPECT_EQ(fp.Index(0), 3u);
    EXPECT_EQ(fp.Index(1), 7u);
    EXPECT_EQ(fp.Index(2), 64u);
    EXPECT_EQ(fp.Indices(), std::vector<std::uint32_t>({3u, 7u, 64u}));
    EXPECT_NE(fp.IndexData(), nullptr);
    EXPECT_NE(fp.IndexDataAddress(), 0u);
}

TEST(SparseFingerprintTest, EmptySparseFingerprintKeepsSpecAndNullPointer) {
    const OEFPSparse fp(sparse_binary_spec(64));

    EXPECT_EQ(fp.SizeBits(), 64u);
    EXPECT_EQ(fp.CountOnBits(), 0u);
    EXPECT_EQ(fp.IndexData(), nullptr);
    EXPECT_EQ(fp.IndexDataAddress(), 0u);
}

TEST(SparseFingerprintTest, RejectsInvalidSparseBinaryStorage) {
    EXPECT_THROW((void)OEFPSparse{counted_spec()}, std::invalid_argument);
    EXPECT_THROW(OEFPSparse(sparse_binary_spec(), {1u, 1u}), std::invalid_argument);
    EXPECT_THROW(OEFPSparse(sparse_binary_spec(), {2u, 1u}), std::invalid_argument);
    EXPECT_THROW(OEFPSparse(sparse_binary_spec(4), {4u}), std::out_of_range);
}

TEST(SparseFingerprintTest, RowAccessRejectsOutOfRangeRows) {
    const OEFPSparse fp(sparse_binary_spec(), {3u});

    EXPECT_THROW(fp.Index(1), std::out_of_range);
}

} // namespace test
} // namespace OEFP
