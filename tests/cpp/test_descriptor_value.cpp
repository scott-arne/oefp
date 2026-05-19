#include "oefp/descriptor_value.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
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

TEST(DescriptorValueTest, RejectsWrongAccessorsAndInvalidMatrixShapes) {
    EXPECT_THROW(static_cast<void>(DescriptorValue::Float(3.5).AsInt()), std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::FloatMatrix({2, 2}, {1.0, 2.0, 3.0}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorValue::IntMatrix({2, 0}, {1, 2}),
        std::invalid_argument);
}
