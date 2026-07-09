#include "oefp/rdkit_descriptors.h"

#include <gtest/gtest.h>

using namespace OEFP;

// 217 RDKit _descList descriptors minus 4 (3 always-zero VSA bins — SMR_VSA8,
// SlogP_VSA9, EState_VSA11 — plus SPS, excluded because it diverges from RDKit
// on exotic aromaticity-model cases) = 213 schema descriptors.
TEST(RDKitSchemaTest, HasAll213Descriptors) {
    EXPECT_EQ(RDKitDescriptorSchema()->Size(), 213u);
}

// Pin the native schema to the committed reference fixture (schema_id, column
// order, and source version), mirroring the Mordred schema test. Without this,
// a drift in src/rdkit_schema.cpp — a bumped source version, edited metadata,
// or reordered columns — would ship green because the conformance and pruning
// tests couple to the schema by name, never by identity or position.
TEST(RDKitSchemaTest, MatchesCommittedFixtureIdentity) {
    const auto schema = RDKitDescriptorSchema();

    EXPECT_EQ(schema->SchemaId(), "1faa8f3e6f579c15");

    // First and last columns pin the schema's ordering endpoints.
    EXPECT_EQ(schema->Definition(0).name, "MaxAbsEStateIndex");
    EXPECT_EQ(schema->Definition(212).name, "fr_urea");

    // Representative interior columns pin the ordering across families. Every
    // index after SPS (old index 5) shifts down by one now that SPS is excluded.
    EXPECT_EQ(schema->IndexOf("qed"), 4u);
    EXPECT_EQ(schema->IndexOf("MolWt"), 5u);
    EXPECT_EQ(schema->IndexOf("BCUT2D_MWHI"), 17u);
    EXPECT_EQ(schema->IndexOf("TPSA"), 80u);
    EXPECT_EQ(schema->IndexOf("fr_benzene"), 162u);

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

// Verify that excluded descriptors are absent from the schema: the three
// structurally always-zero VSA bins and SPS (excluded because it diverges from
// RDKit on exotic aromaticity-model cases).
TEST(RDKitSchemaTest, ExcludesAlwaysZeroVsaBins) {
    const auto schema = RDKitDescriptorSchema();
    EXPECT_FALSE(schema->Contains("SMR_VSA8"));
    EXPECT_FALSE(schema->Contains("SlogP_VSA9"));
    EXPECT_FALSE(schema->Contains("EState_VSA11"));
    EXPECT_FALSE(schema->Contains("SPS"));
}
