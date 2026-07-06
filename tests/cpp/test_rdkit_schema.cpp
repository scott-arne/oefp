#include "oefp/rdkit_descriptors.h"

#include <gtest/gtest.h>

using namespace OEFP;

TEST(RDKitSchemaTest, HasAll217Descriptors) {
    EXPECT_EQ(RDKitDescriptorSchema()->Size(), 217u);
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
