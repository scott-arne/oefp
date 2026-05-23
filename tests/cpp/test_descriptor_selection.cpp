#include "oefp/descriptor.h"
#include "oefp/descriptor_selection.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

using namespace OEFP;

TEST(DescriptorSelectionTest, SelectsByNamesGroupAndIndices) {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float, "mordred:constitutional"});
    builder.Add(DescriptorDefinition{"TopoPSA", DescriptorValueKind::Float, "mordred:surface"});
    builder.Add(DescriptorDefinition{"nAtom", DescriptorValueKind::Int, "mordred:constitutional"});
    const auto schema = builder.Build();

    const auto by_names = DescriptorSelection::Names({"TopoPSA", "MW"}).Resolve(*schema);
    EXPECT_EQ(by_names, std::vector<std::size_t>({1u, 0u}));

    const auto by_group = DescriptorSelection::Group("mordred:constitutional").Resolve(*schema);
    EXPECT_EQ(by_group, std::vector<std::size_t>({0u, 2u}));

    const auto by_indices = DescriptorSelection::Indices({2u, 0u}).Resolve(*schema);
    EXPECT_EQ(by_indices, std::vector<std::size_t>({2u, 0u}));
}

TEST(DescriptorSelectionTest, DescriptorSetSubsetUsesProjectionOrder) {
    DescriptorSchemaBuilder schema_builder;
    schema_builder.Add(
        DescriptorDefinition{"MW", DescriptorValueKind::Float, "mordred:constitutional"});
    schema_builder.Add(DescriptorDefinition{"TopoPSA", DescriptorValueKind::Float, "mordred:surface"});
    schema_builder.Add(
        DescriptorDefinition{"nAtom", DescriptorValueKind::Int, "mordred:constitutional"});
    const auto schema = schema_builder.Build();

    DescriptorSetBuilder row_builder(schema);
    row_builder.Set("MW", DescriptorValue::Float(46.0));
    row_builder.Set("TopoPSA", DescriptorValue::Float(20.2));
    row_builder.Set("nAtom", DescriptorValue::Int(9));

    const auto subset = row_builder.Build("ethanol")
                            .Subset(DescriptorSelection::Names({"nAtom", "MW"}));

    EXPECT_EQ(subset.RowId(), "ethanol");
    EXPECT_EQ(subset.Schema().Size(), 2u);
    EXPECT_EQ(subset.Schema().Definition(0).name, "nAtom");
    EXPECT_EQ(subset.Int("nAtom"), 9);
    EXPECT_DOUBLE_EQ(subset.Float("MW"), 46.0);
}

TEST(DescriptorSelectionTest, RejectsOutOfRangeExplicitIndex) {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float});
    const auto schema = builder.Build();

    EXPECT_THROW(
        static_cast<void>(DescriptorSelection::Indices({1u}).Resolve(*schema)),
        std::out_of_range);
}
