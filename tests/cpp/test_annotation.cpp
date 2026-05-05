#include <gtest/gtest.h>

#include "oefp/annotation.h"

namespace OEFP {
namespace test {

TEST(AnnotationTest, StoresAndRetrievesBitLabels) {
    OEFPAnnotationSet annotations;
    annotations.SetBitLabel(42, "phenolic oxygen");

    EXPECT_EQ(annotations.BitLabel(42), "phenolic oxygen");
    EXPECT_EQ(annotations.BitLabel(7), "");
}

TEST(MappingTest, StoresBitProvenanceForRows) {
    OEFPMappingSet mappings;
    mappings.AddAtomMapping(3, 42, {1, 2, 5});

    const auto atoms = mappings.AtomsForBit(3, 42);

    ASSERT_EQ(atoms.size(), 3u);
    EXPECT_EQ(atoms[0], 1u);
    EXPECT_EQ(atoms[1], 2u);
    EXPECT_EQ(atoms[2], 5u);
    EXPECT_TRUE(mappings.AtomsForBit(2, 42).empty());
}

} // namespace test
} // namespace OEFP
