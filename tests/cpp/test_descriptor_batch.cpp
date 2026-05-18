#include <gtest/gtest.h>

#include "oefp/descriptor_batch.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {
namespace {

DescriptorSpec string_spec() {
    DescriptorSpec spec;
    spec.value_type = DescriptorValueType::String;
    spec.source_name = "unit-test";
    spec.source_type = "descriptor";
    spec.source_version = "1";
    spec.parameters = "kind=string";
    return spec;
}

DescriptorSpec integer_spec() {
    auto spec = string_spec();
    spec.value_type = DescriptorValueType::Integer;
    spec.parameters = "kind=integer";
    return spec;
}

DescriptorSpec float_spec() {
    auto spec = string_spec();
    spec.value_type = DescriptorValueType::Float;
    spec.parameters = "kind=float";
    return spec;
}

} // namespace

TEST(DescriptorBatchTest, BuildsCsrStorageFromDescriptorSets) {
    const auto first = DescriptorSet::FromStrings(string_spec(), {"beta", "alpha", "beta"});
    const auto second = DescriptorSet::FromStrings(string_spec(), {"gamma", "beta"});
    const auto empty = DescriptorSet::FromStrings(string_spec(), {});

    const auto batch = DescriptorBatch::FromDescriptorSets({first, second, empty});

    EXPECT_EQ(batch.Spec(), string_spec());
    EXPECT_EQ(batch.ValueType(), DescriptorValueType::String);
    EXPECT_EQ(batch.Size(), 3u);
    EXPECT_EQ(batch.EntryCount(), 4u);
    EXPECT_EQ(batch.RowEntryCount(0), 2u);
    EXPECT_EQ(batch.RowEntryCount(1), 2u);
    EXPECT_EQ(batch.RowEntryCount(2), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
    EXPECT_EQ(batch.RowOffset(1), 2u);
    EXPECT_EQ(batch.RowOffset(2), 4u);
    EXPECT_EQ(batch.RowOffset(3), 4u);
    EXPECT_EQ(batch.StringKeys(), std::vector<std::string>({"alpha", "beta", "beta", "gamma"}));
    EXPECT_TRUE(batch.IntegerKeys().empty());
    EXPECT_TRUE(batch.FloatKeys().empty());
    EXPECT_EQ(batch.Counts(), std::vector<std::uint32_t>({1u, 2u, 1u, 1u}));
    EXPECT_EQ(batch.RowOffsets(), std::vector<std::uint64_t>({0u, 2u, 4u, 4u}));
    ASSERT_NE(batch.CountData(), nullptr);
    EXPECT_EQ(batch.CountData()[1], 2u);
    EXPECT_NE(batch.CountDataAddress(), 0u);
    ASSERT_NE(batch.RowOffsetData(), nullptr);
    EXPECT_EQ(batch.RowOffsetData()[2], 4u);
    EXPECT_NE(batch.RowOffsetDataAddress(), 0u);

    auto appended = DescriptorBatch(string_spec());
    appended.Append(first);
    appended.Append(second);
    appended.Append(empty);
    EXPECT_EQ(appended, batch);
}

TEST(DescriptorBatchTest, StoresOnlyActiveKeyVector) {
    const auto integers = DescriptorBatch::FromDescriptorSets(
        {DescriptorSet::FromIntegers(integer_spec(), {7, -2, 7})});
    EXPECT_EQ(integers.ValueType(), DescriptorValueType::Integer);
    EXPECT_EQ(integers.EntryCount(), 2u);
    EXPECT_TRUE(integers.StringKeys().empty());
    EXPECT_EQ(integers.IntegerKeys(), std::vector<std::int64_t>({-2, 7}));
    EXPECT_TRUE(integers.FloatKeys().empty());
    EXPECT_EQ(integers.Counts(), std::vector<std::uint32_t>({1u, 2u}));
    ASSERT_NE(integers.IntegerKeyData(), nullptr);
    EXPECT_EQ(integers.IntegerKeyData()[0], -2);
    EXPECT_NE(integers.IntegerKeyDataAddress(), 0u);
    EXPECT_EQ(integers.FloatKeyData(), nullptr);
    EXPECT_EQ(integers.FloatKeyDataAddress(), 0u);

    const auto floats =
        DescriptorBatch::FromDescriptorSets({DescriptorSet::FromFloats(float_spec(), {2.5, 1.0})});
    EXPECT_EQ(floats.ValueType(), DescriptorValueType::Float);
    EXPECT_EQ(floats.EntryCount(), 2u);
    EXPECT_TRUE(floats.StringKeys().empty());
    EXPECT_TRUE(floats.IntegerKeys().empty());
    EXPECT_EQ(floats.FloatKeys(), std::vector<double>({1.0, 2.5}));
    EXPECT_EQ(floats.Counts(), std::vector<std::uint32_t>({1u, 1u}));
    ASSERT_NE(floats.FloatKeyData(), nullptr);
    EXPECT_EQ(floats.FloatKeyData()[0], 1.0);
    EXPECT_NE(floats.FloatKeyDataAddress(), 0u);
    EXPECT_EQ(floats.IntegerKeyData(), nullptr);
    EXPECT_EQ(floats.IntegerKeyDataAddress(), 0u);
}

TEST(DescriptorBatchTest, EmptyBatchKeepsSpecAndSingleZeroOffset) {
    const DescriptorBatch batch(integer_spec());

    EXPECT_EQ(batch.Spec(), integer_spec());
    EXPECT_EQ(batch.ValueType(), DescriptorValueType::Integer);
    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.EntryCount(), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
    EXPECT_TRUE(batch.StringKeys().empty());
    EXPECT_TRUE(batch.IntegerKeys().empty());
    EXPECT_TRUE(batch.FloatKeys().empty());
    EXPECT_TRUE(batch.Counts().empty());
    EXPECT_EQ(batch.RowOffsets(), std::vector<std::uint64_t>({0u}));
    EXPECT_EQ(batch.CountData(), nullptr);
    EXPECT_EQ(batch.CountDataAddress(), 0u);
    ASSERT_NE(batch.RowOffsetData(), nullptr);
    EXPECT_EQ(batch.RowOffsetData()[0], 0u);
    EXPECT_NE(batch.RowOffsetDataAddress(), 0u);
}

TEST(DescriptorBatchTest, EmptyFromDescriptorSetsReturnsDefaultBatch) {
    const auto batch = DescriptorBatch::FromDescriptorSets({});

    EXPECT_EQ(batch.Spec(), DescriptorSpec());
    EXPECT_EQ(batch.ValueType(), DescriptorValueType::String);
    EXPECT_EQ(batch.Size(), 0u);
    EXPECT_EQ(batch.EntryCount(), 0u);
    EXPECT_EQ(batch.RowOffset(0), 0u);
    EXPECT_EQ(batch.RowOffsets(), std::vector<std::uint64_t>({0u}));
}

TEST(DescriptorBatchTest, EqualityIncludesWhetherSpecIsBound) {
    DescriptorBatch unbound;
    DescriptorBatch bound{DescriptorSpec()};

    EXPECT_NE(unbound, bound);

    unbound.Append(DescriptorSet::FromIntegers(integer_spec(), {1}));
    EXPECT_THROW(
        bound.Append(DescriptorSet::FromIntegers(integer_spec(), {1})),
        std::invalid_argument);
}

TEST(DescriptorBatchTest, RejectsMismatchedSpecs) {
    DescriptorBatch batch(string_spec());
    batch.Append(DescriptorSet::FromStrings(string_spec(), {"alpha"}));

    auto metadata_spec = string_spec();
    metadata_spec.source_version = "2";
    EXPECT_THROW(
        batch.Append(DescriptorSet::FromStrings(metadata_spec, {"beta"})),
        std::invalid_argument);

    EXPECT_THROW(
        batch.Append(DescriptorSet::FromIntegers(integer_spec(), {1})),
        std::invalid_argument);
}

TEST(DescriptorBatchTest, DefaultBatchTakesFirstRowSpec) {
    DescriptorBatch batch;
    batch.Append(DescriptorSet::FromIntegers(integer_spec(), {1, 2, 1}));

    EXPECT_EQ(batch.Spec(), integer_spec());
    EXPECT_EQ(batch.ValueType(), DescriptorValueType::Integer);
    EXPECT_EQ(batch.Size(), 1u);
    EXPECT_EQ(batch.IntegerKeys(), std::vector<std::int64_t>({1, 2}));
    EXPECT_EQ(batch.Counts(), std::vector<std::uint32_t>({2u, 1u}));
}

TEST(DescriptorBatchTest, RowAccessRejectsOutOfRangeRows) {
    const auto batch =
        DescriptorBatch::FromDescriptorSets({DescriptorSet::FromStrings(string_spec(), {"alpha"})});

    EXPECT_EQ(batch.RowOffset(1), 1u);
    EXPECT_THROW(batch.RowOffset(2), std::out_of_range);
    EXPECT_THROW(batch.RowEntryCount(1), std::out_of_range);
}

TEST(DescriptorBatchTest, EqualityIncludesSpecKeysCountsAndOffsets) {
    const auto first = DescriptorBatch::FromDescriptorSets(
        {DescriptorSet::FromStrings(string_spec(), {"alpha", "beta"})});
    const auto same = DescriptorBatch::FromDescriptorSets(
        {DescriptorSet::FromStrings(string_spec(), {"beta", "alpha"})});
    const auto different_counts =
        DescriptorBatch::FromDescriptorSets({DescriptorSet::FromStrings(string_spec(), {"alpha"})});

    auto different_spec = string_spec();
    different_spec.source_version = "2";
    const auto with_different_spec =
        DescriptorBatch::FromDescriptorSets({DescriptorSet::FromStrings(different_spec, {"alpha"})});
    const auto different_offsets = DescriptorBatch::FromDescriptorSets(
        {DescriptorSet::FromStrings(string_spec(), {"alpha"}),
         DescriptorSet::FromStrings(string_spec(), {"beta"})});

    EXPECT_EQ(first, same);
    EXPECT_NE(first, different_counts);
    EXPECT_NE(first, with_different_spec);
    EXPECT_NE(first, different_offsets);
}

} // namespace test
} // namespace OEFP
