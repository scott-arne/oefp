#include "oefp/mordred.h"

#include <gtest/gtest.h>

using namespace OEFP;

TEST(MordredSchemaTest, FullSchemaContainsAllDescriptors) {
    const auto schema = MordredDescriptorSchema();

    EXPECT_EQ(schema->Size(), 1826u);
    EXPECT_EQ(schema->SchemaId(), "1fa9a8e86a7c8731");
    EXPECT_EQ(schema->Definition(0).name, "ABC");
    EXPECT_EQ(schema->Definition(1825).name, "mZagreb2");
    EXPECT_TRUE(schema->Contains("ABC"));
    EXPECT_TRUE(schema->Contains("nAtom"));
    EXPECT_TRUE(schema->Contains("Lipinski"));
    EXPECT_TRUE(schema->Contains("GhoseFilter"));
    EXPECT_EQ(schema->IndexOf("ABC"), 0u);
    EXPECT_EQ(schema->IndexOf("nAtom"), 18u);
    EXPECT_EQ(schema->IndexOf("Lipinski"), 1351u);
    EXPECT_EQ(schema->IndexOf("GhoseFilter"), 1352u);
    EXPECT_EQ(schema->IndexOf("mZagreb2"), 1825u);
    EXPECT_EQ(schema->Definition(schema->IndexOf("ABC")).source_name, "Mordred");
    EXPECT_EQ(schema->Definition(schema->IndexOf("ABC")).source_version, "Mordred-1.2.0");
    EXPECT_EQ(
        schema->Definition(schema->IndexOf("ABC")).value_kind,
        DescriptorValueKind::Float);
    EXPECT_EQ(
        schema->Definition(schema->IndexOf("nAtom")).value_kind,
        DescriptorValueKind::Int);
    EXPECT_EQ(
        schema->Definition(schema->IndexOf("Lipinski")).value_kind,
        DescriptorValueKind::Bool);
}
