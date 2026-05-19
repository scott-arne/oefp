#include "oefp/descriptor_value.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

using namespace OEFP;

TEST(DescriptorValueTest, StoresScalarKinds) {
    EXPECT_TRUE(DescriptorValue::Bool(true).AsBool());
    EXPECT_EQ(DescriptorValue::Int(42).AsInt(), 42);
    EXPECT_DOUBLE_EQ(DescriptorValue::Float(3.5).AsFloat(), 3.5);
    EXPECT_EQ(DescriptorValue::String("acid").AsString(), "acid");
}

TEST(DescriptorValueTest, StoresVectorAndMatrixKinds) {
    const auto vector = DescriptorValue::FloatVector({1.0, 2.0, 3.0});
    EXPECT_EQ(vector.AsFloatVector(), std::vector<double>({1.0, 2.0, 3.0}));

    const auto matrix = DescriptorValue::FloatMatrix({2, 2}, {1.0, 2.0, 3.0, 4.0});
    EXPECT_EQ(matrix.Shape(), std::vector<std::uint64_t>({2u, 2u}));
    EXPECT_EQ(matrix.AsFloatVector(), std::vector<double>({1.0, 2.0, 3.0, 4.0}));
}

TEST(DescriptorValueTest, StoresCountedKeyKinds) {
    const auto strings = DescriptorValue::CountedStringKeys({"alpha", "beta"}, {2u, 1u});
    EXPECT_EQ(strings.Kind(), DescriptorValueKind::CountedStringKeys);
    EXPECT_EQ(strings.CountedStringKeys().keys, std::vector<std::string>({"alpha", "beta"}));
    EXPECT_EQ(strings.CountedStringKeys().counts, std::vector<std::uint32_t>({2u, 1u}));

    const auto integers = DescriptorValue::CountedIntegerKeys({7, 11}, {1u, 3u});
    EXPECT_EQ(integers.Kind(), DescriptorValueKind::CountedIntegerKeys);
    EXPECT_EQ(integers.CountedIntegerKeys().keys, std::vector<std::int64_t>({7, 11}));
    EXPECT_EQ(integers.CountedIntegerKeys().counts, std::vector<std::uint32_t>({1u, 3u}));
}

TEST(DescriptorValueTest, RejectsInvalidCountedKeyStorage) {
    EXPECT_THROW(
        DescriptorValue::CountedStringKeys({"alpha"}, {1u, 2u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::CountedIntegerKeys({1, 2}, {1u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::CountedStringKeys({"alpha"}, {0u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::CountedIntegerKeys({1}, {0u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::CountedStringKeys({"beta", "alpha"}, {1u, 1u}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::CountedIntegerKeys({7, 7}, {1u, 1u}),
        std::invalid_argument);
}

TEST(DescriptorValueTest, RejectsWrongAccessorsAndInvalidMatrixShapes) {
    EXPECT_THROW(static_cast<void>(DescriptorValue::Float(3.5).AsInt()), std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(DescriptorValue::CountedStringKeys({"alpha"}, {1u}).CountedIntegerKeys()),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(DescriptorValue::CountedIntegerKeys({1}, {1u}).CountedStringKeys()),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::FloatMatrix({2, 2}, {1.0, 2.0, 3.0}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::IntMatrix({2, 0}, {1, 2}),
        std::invalid_argument);
}
