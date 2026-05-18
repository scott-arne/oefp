#include "oefp/descriptor_schema.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

using namespace OEFP;

TEST(DescriptorSchemaTest, BuildsStableSchemaWithNameAndGroupLookup) {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{
        "MW",
        DescriptorValueKind::Float,
        "mordred:constitutional",
        "Mordred",
        "Weight",
        "1.2.0",
        "",
        "Da",
        "molecular weight",
        {}
    });
    builder.Add(DescriptorDefinition{
        "nAtom",
        DescriptorValueKind::Int,
        "mordred:atom_count",
        "Mordred",
        "AtomCount",
        "1.2.0",
        "type=Atom",
        "",
        "number of atoms",
        {}
    });

    const auto schema = builder.Build();

    EXPECT_EQ(schema->Size(), 2u);
    EXPECT_EQ(schema->IndexOf("MW"), 0u);
    EXPECT_EQ(schema->IndexOf("nAtom"), 1u);
    EXPECT_EQ(schema->Definition(0).name, "MW");
    EXPECT_EQ(schema->IndicesForGroup("mordred:constitutional"), std::vector<std::size_t>({0u}));
    EXPECT_EQ(schema->SchemaId(), "806f12e60900ac40");
}

TEST(DescriptorSchemaTest, RejectsDuplicateNames) {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float});
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float});

    EXPECT_THROW(static_cast<void>(builder.Build()), std::invalid_argument);
}

TEST(DescriptorSchemaTest, RejectsInvalidValueKinds) {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"bad", static_cast<DescriptorValueKind>(999)});

    EXPECT_THROW(static_cast<void>(builder.Build()), std::invalid_argument);
}

TEST(DescriptorSchemaTest, ProjectsNamedSubsets) {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float, "mordred:constitutional"});
    builder.Add(DescriptorDefinition{"nAtom", DescriptorValueKind::Int, "mordred:atom_count"});
    builder.Add(DescriptorDefinition{"TopoPSA", DescriptorValueKind::Float, "mordred:surface"});

    const auto schema = builder.Build();
    const auto projected = schema->Project({"TopoPSA", "MW"});

    EXPECT_EQ(projected->Size(), 2u);
    EXPECT_EQ(projected->Definition(0).name, "TopoPSA");
    EXPECT_EQ(projected->Definition(1).name, "MW");
    EXPECT_NE(projected->SchemaId(), schema->SchemaId());
}
