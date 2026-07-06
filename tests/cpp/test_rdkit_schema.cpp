#include "oefp/rdkit_descriptors.h"

#include <gtest/gtest.h>

using namespace OEFP;

// 217 RDKit _descList descriptors minus 3 always-zero VSA bins (SMR_VSA8,
// SlogP_VSA9, EState_VSA11) = 214 schema descriptors.
TEST(RDKitSchemaTest, HasAll214Descriptors) {
    EXPECT_EQ(RDKitDescriptorSchema()->Size(), 214u);
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
