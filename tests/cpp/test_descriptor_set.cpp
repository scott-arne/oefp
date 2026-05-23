#include "oefp/descriptor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace OEFP;

namespace {

std::shared_ptr<const DescriptorSchema> mixed_schema() {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float, "mordred:constitutional"});
    builder.Add(DescriptorDefinition{"nAtom", DescriptorValueKind::Int, "mordred:atom_count"});
    DescriptorDefinition lipinski{"Lipinski", DescriptorValueKind::Bool, "mordred:filter"};
    lipinski.prerequisites = kDescriptorPrerequisiteCoordinates3D;
    builder.Add(lipinski);
    builder.Add(DescriptorDefinition{"Class", DescriptorValueKind::String, "manual:category"});
    return builder.Build();
}

std::shared_ptr<const DescriptorSchema> shaped_schema() {
    DescriptorDefinition matrix{"jacobian", DescriptorValueKind::FloatMatrix};
    matrix.shape = DescriptorShape{{2u, 2u}};

    DescriptorSchemaBuilder builder;
    builder.Add(matrix);
    return builder.Build();
}

std::shared_ptr<const DescriptorSchema> counted_string_schema() {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{
        "raw",
        DescriptorValueKind::CountedStringKeys,
        "test",
        "unit-test",
        "descriptor",
        "1",
        "kind=string"});
    return builder.Build();
}

} // namespace

TEST(TypedDescriptorSetTest, StoresTypedValuesAgainstSharedSchema) {
    DescriptorSetBuilder builder(mixed_schema());
    builder.Set("MW", DescriptorValue::Float(46.069));
    builder.Set("nAtom", DescriptorValue::Int(9));
    builder.Set("Lipinski", DescriptorValue::Bool(true));
    builder.Set("Class", DescriptorValue::String("alcohol"));

    const auto row = builder.Build();

    EXPECT_DOUBLE_EQ(row.Float("MW"), 46.069);
    EXPECT_EQ(row.Int("nAtom"), 9);
    EXPECT_TRUE(row.Bool("Lipinski"));
    EXPECT_EQ(row.String("Class"), "alcohol");
    EXPECT_TRUE(row.Has("MW"));
}

TEST(TypedDescriptorSetTest, PreservesMissingValues) {
    DescriptorSetBuilder builder(mixed_schema());
    builder.Set("MW", DescriptorValue::Float(46.069));

    const auto row = builder.Build();

    EXPECT_TRUE(row.Has("MW"));
    EXPECT_FALSE(row.Has("nAtom"));
    EXPECT_THROW(static_cast<void>(row.Int("nAtom")), std::invalid_argument);
}

TEST(TypedDescriptorSetTest, LeavesValuesMissingWhenPrerequisitesAreUnavailable) {
    DescriptorSetBuilder builder(
        mixed_schema(),
        kDescriptorPrerequisiteGraph);
    builder.Set("MW", DescriptorValue::Float(46.069));
    builder.Set("Lipinski", DescriptorValue::Bool(true));

    const auto row = builder.Build();

    EXPECT_EQ(builder.AvailablePrerequisites(), kDescriptorPrerequisiteGraph);
    EXPECT_TRUE(row.Has("MW"));
    EXPECT_FALSE(row.Has("Lipinski"));
    EXPECT_EQ(
        MissingDescriptorPrerequisites(
            row.Schema().Definition(row.Schema().IndexOf("Lipinski")).prerequisites,
            builder.AvailablePrerequisites()),
        kDescriptorPrerequisiteCoordinates3D);
    EXPECT_FALSE(DescriptorPrerequisitesSatisfied(
        kDescriptorPrerequisiteCoordinates3D,
        builder.AvailablePrerequisites()));
}

TEST(TypedDescriptorSetTest, RejectsWrongValueKind) {
    DescriptorSetBuilder builder(mixed_schema());
    EXPECT_THROW(builder.Set("MW", DescriptorValue::String("wrong")), std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            mixed_schema(),
            {DescriptorValue::String("wrong")}),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            mixed_schema(),
            {DescriptorValue::Float(46.069), DescriptorValue::Int(9)}),
        std::invalid_argument);
}

TEST(TypedDescriptorSetTest, RejectsWrongShape) {
    DescriptorSetBuilder builder(shaped_schema());
    EXPECT_THROW(
        builder.Set("jacobian", DescriptorValue::FloatMatrix({4u}, {1.0, 2.0, 3.0, 4.0})),
        std::invalid_argument);
    EXPECT_THROW(
        DescriptorSet(
            shaped_schema(),
            {DescriptorValue::FloatMatrix({1u, 4u}, {1.0, 2.0, 3.0, 4.0})}),
        std::invalid_argument);
}

TEST(TypedDescriptorSetTest, RejectsLegacyCountedKeyAccessors) {
    DescriptorSetBuilder builder(mixed_schema());
    builder.Set("MW", DescriptorValue::Float(46.069));

    const auto row = builder.Build();

    EXPECT_THROW(static_cast<void>(row.Spec()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.ValueType()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.TotalCount()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.StringKeys()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.IntegerKeys()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.FloatKeys()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.Counts()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.CountData()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.CountDataAddress()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.IntegerKeyData()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.IntegerKeyDataAddress()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.FloatKeyData()), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row.FloatKeyDataAddress()), std::invalid_argument);
}

TEST(TypedDescriptorSetTest, OneColumnCountedRowExposesLegacyStorage) {
    DescriptorSetBuilder builder(counted_string_schema());
    builder.Set("raw", DescriptorValue::CountedStringKeys({"alpha", "beta"}, {2u, 1u}));

    const auto row = builder.Build("row-1");

    EXPECT_EQ(row.ValueType(), DescriptorValueType::String);
    EXPECT_EQ(row.Size(), 2u);
    EXPECT_EQ(row.TotalCount(), 3u);
    EXPECT_EQ(row.StringKeys(), std::vector<std::string>({"alpha", "beta"}));
    EXPECT_EQ(row.Counts(), std::vector<std::uint32_t>({2u, 1u}));
    EXPECT_EQ(row.Spec().source_name, "unit-test");
    EXPECT_EQ(row.Spec().source_type, "descriptor");
    EXPECT_EQ(row.Spec().source_version, "1");
    EXPECT_EQ(row.Spec().parameters, "kind=string");
    EXPECT_EQ(row.CountData(), row.Counts().data());
}

TEST(TypedDescriptorSetTest, ProjectsNamedSubsets) {
    DescriptorSetBuilder builder(mixed_schema());
    builder.Set("MW", DescriptorValue::Float(46.069));
    builder.Set("Class", DescriptorValue::String("alcohol"));

    const auto row = builder.Build("ethanol");
    const auto subset = row.Subset({"Class", "MW"});

    EXPECT_EQ(subset.RowId(), "ethanol");
    ASSERT_EQ(subset.Schema().Size(), 2u);
    EXPECT_EQ(subset.Schema().Definition(0).name, "Class");
    EXPECT_EQ(subset.Schema().Definition(1).name, "MW");
    EXPECT_EQ(subset.String("Class"), "alcohol");
    EXPECT_DOUBLE_EQ(subset.Float("MW"), 46.069);
}
