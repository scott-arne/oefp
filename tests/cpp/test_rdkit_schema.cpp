#include "oefp/rdkit_descriptors.h"

#include <gtest/gtest.h>

using namespace OEFP;

// 217 RDKit _descList descriptors minus 3 always-zero VSA bins (SMR_VSA8,
// SlogP_VSA9, EState_VSA11) = 214 schema descriptors.
TEST(RDKitSchemaTest, HasAll214Descriptors) {
    EXPECT_EQ(RDKitDescriptorSchema()->Size(), 214u);
}

// Pin the native schema to the committed reference fixture (schema_id, column
// order, and source version), mirroring the Mordred schema test. Without this,
// a drift in src/rdkit_schema.cpp — a bumped source version, edited metadata,
// or reordered columns — would ship green because the conformance and pruning
// tests couple to the schema by name, never by identity or position.
TEST(RDKitSchemaTest, MatchesCommittedFixtureIdentity) {
    const auto schema = RDKitDescriptorSchema();

    EXPECT_EQ(schema->SchemaId(), "fe975cfea4084c34");

    // First and last columns pin the schema's ordering endpoints.
    EXPECT_EQ(schema->Definition(0).name, "MaxAbsEStateIndex");
    EXPECT_EQ(schema->Definition(213).name, "fr_urea");

    // Representative interior columns pin the ordering across families.
    EXPECT_EQ(schema->IndexOf("qed"), 4u);
    EXPECT_EQ(schema->IndexOf("MolWt"), 6u);
    EXPECT_EQ(schema->IndexOf("BCUT2D_MWHI"), 18u);
    EXPECT_EQ(schema->IndexOf("TPSA"), 81u);
    EXPECT_EQ(schema->IndexOf("fr_benzene"), 163u);

    EXPECT_EQ(schema->Definition(0).source_version, "RDKit-2026.03.3");
}

TEST(RDKitSchemaTest, SourceNameIsRDKit) {
    const auto schema = RDKitDescriptorSchema();
    for (std::size_t i = 0; i < schema->Size(); ++i) {
        EXPECT_EQ(schema->Definition(i).source_name, "RDKit") << i;
    }
}

TEST(RDKitSchemaTest, IntegerAndFragmentCountsAreIntKind) {
    const auto schema = RDKitDescriptorSchema();
    EXPECT_EQ(schema->Definition(schema->IndexOf("HeavyAtomCount")).value_kind,
              DescriptorValueKind::Int);
    EXPECT_EQ(schema->Definition(schema->IndexOf("fr_benzene")).value_kind,
              DescriptorValueKind::Int);
    EXPECT_EQ(schema->Definition(schema->IndexOf("MolLogP")).value_kind,
              DescriptorValueKind::Float);
}

TEST(RDKitSchemaTest, NoCoordinatePrerequisites) {
    const auto schema = RDKitDescriptorSchema();
    for (std::size_t i = 0; i < schema->Size(); ++i) {
        EXPECT_EQ(schema->Definition(i).prerequisites, kDescriptorPrerequisiteNone) << i;
    }
}

// Verify that structurally always-zero VSA bins are excluded from the schema.
TEST(RDKitSchemaTest, ExcludesAlwaysZeroVsaBins) {
    const auto schema = RDKitDescriptorSchema();
    EXPECT_FALSE(schema->Contains("SMR_VSA8"));
    EXPECT_FALSE(schema->Contains("SlogP_VSA9"));
    EXPECT_FALSE(schema->Contains("EState_VSA11"));
}
