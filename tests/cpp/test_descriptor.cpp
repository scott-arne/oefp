#include <gtest/gtest.h>

#include "oefp/descriptor.h"

#include <cstdint>
#include <limits>
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

TEST(DescriptorSetTest, AggregatesStringKeysIntoCanonicalCounts) {
    const auto descriptors =
        DescriptorSet::FromStrings(string_spec(), {"beta", "alpha", "beta", "gamma", "alpha"});

    EXPECT_EQ(descriptors.ValueType(), DescriptorValueType::String);
    EXPECT_EQ(descriptors.Size(), 3u);
    EXPECT_EQ(descriptors.TotalCount(), 5u);
    EXPECT_EQ(descriptors.StringKeys(), std::vector<std::string>({"alpha", "beta", "gamma"}));
    EXPECT_EQ(descriptors.Counts(), std::vector<std::uint32_t>({2u, 2u, 1u}));
    EXPECT_TRUE(descriptors.IntegerKeys().empty());
    EXPECT_TRUE(descriptors.FloatKeys().empty());
    EXPECT_NE(descriptors.CountData(), nullptr);
    EXPECT_NE(descriptors.CountDataAddress(), 0u);

    const auto empty = DescriptorSet::FromStrings(string_spec(), {});
    EXPECT_EQ(empty.Size(), 0u);
    EXPECT_EQ(empty.TotalCount(), 0u);
    EXPECT_EQ(empty.CountData(), nullptr);
    EXPECT_EQ(empty.CountDataAddress(), 0u);
}

TEST(DescriptorSetTest, StoresIntegerAndFloatKeys) {
    const DescriptorSet integers(
        integer_spec(),
        std::vector<std::int64_t>{-2, 7, 11},
        std::vector<std::uint32_t>{1u, 4u, 2u});
    EXPECT_EQ(integers.ValueType(), DescriptorValueType::Integer);
    EXPECT_EQ(integers.IntegerKeys(), std::vector<std::int64_t>({-2, 7, 11}));
    EXPECT_EQ(integers.Counts(), std::vector<std::uint32_t>({1u, 4u, 2u}));
    EXPECT_EQ(integers.TotalCount(), 7u);

    const auto floats = DescriptorSet::FromFloats(float_spec(), {2.5, 1.0, 2.5});
    EXPECT_EQ(floats.ValueType(), DescriptorValueType::Float);
    EXPECT_EQ(floats.FloatKeys(), std::vector<double>({1.0, 2.5}));
    EXPECT_EQ(floats.Counts(), std::vector<std::uint32_t>({1u, 2u}));
    EXPECT_EQ(floats.TotalCount(), 3u);
}

TEST(DescriptorSetTest, RejectsInvalidCanonicalStorage) {
    EXPECT_THROW(
        DescriptorSet(
            string_spec(),
            std::vector<std::string>{"beta", "alpha"},
            std::vector<std::uint32_t>{1u, 1u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            string_spec(),
            std::vector<std::string>{"alpha", "alpha"},
            std::vector<std::uint32_t>{1u, 2u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            string_spec(),
            std::vector<std::string>{"alpha"},
            std::vector<std::uint32_t>{0u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            string_spec(),
            std::vector<std::string>{"alpha"},
            std::vector<std::uint32_t>{1u, 2u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            integer_spec(),
            std::vector<std::string>{"alpha"},
            std::vector<std::uint32_t>{1u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            float_spec(),
            std::vector<std::int64_t>{1},
            std::vector<std::uint32_t>{1u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            string_spec(),
            std::vector<double>{1.0},
            std::vector<std::uint32_t>{1u}),
        std::invalid_argument);
}

TEST(DescriptorSetTest, RejectsNonFiniteFloatKeys) {
    EXPECT_THROW(
        DescriptorSet(
            float_spec(),
            std::vector<double>{std::numeric_limits<double>::infinity()},
            std::vector<std::uint32_t>{1u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet::FromFloats(float_spec(), {std::numeric_limits<double>::quiet_NaN()}),
        std::invalid_argument);
}

TEST(DescriptorSetTest, EqualityIncludesSpecKeysAndCounts) {
    const auto first = DescriptorSet::FromStrings(string_spec(), {"alpha", "beta", "alpha"});
    const auto same = DescriptorSet::FromStrings(string_spec(), {"beta", "alpha", "alpha"});

    auto different_spec = string_spec();
    different_spec.source_version = "2";
    const auto with_different_spec =
        DescriptorSet::FromStrings(different_spec, {"alpha", "beta", "alpha"});
    const auto with_different_keys = DescriptorSet::FromStrings(string_spec(), {"alpha", "gamma", "alpha"});
    const DescriptorSet with_different_counts(
        string_spec(),
        std::vector<std::string>{"alpha", "beta"},
        std::vector<std::uint32_t>{1u, 2u});

    EXPECT_EQ(first.Spec(), string_spec());
    EXPECT_EQ(first, same);
    EXPECT_NE(first, with_different_spec);
    EXPECT_NE(first, with_different_keys);
    EXPECT_NE(first, with_different_counts);
}

} // namespace test
} // namespace OEFP
