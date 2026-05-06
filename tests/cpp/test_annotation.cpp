#include <gtest/gtest.h>

#include "oefp/annotation.h"

#include <cstdint>
#include <vector>

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

TEST(MappingTest, StoresBitEnvironmentProvenanceForRows) {
    OEFPMappingSet mappings;
    mappings.AddEnvironmentMapping(3, 42, 1, 0);
    mappings.AddEnvironmentMapping(3, 42, 5, 2);
    mappings.AddEnvironmentMapping(3, 7, 2, 1);

    const auto bit_ids = mappings.BitIds(3);
    ASSERT_EQ(bit_ids.size(), 2u);
    EXPECT_EQ(bit_ids[0], 7u);
    EXPECT_EQ(bit_ids[1], 42u);

    const auto environments = mappings.EnvironmentsForBit(3, 42);
    ASSERT_EQ(environments.size(), 2u);
    EXPECT_EQ(environments[0].AtomId(), 1u);
    EXPECT_EQ(environments[0].Radius(), 0u);
    EXPECT_EQ(environments[1].AtomId(), 5u);
    EXPECT_EQ(environments[1].Radius(), 2u);

    EXPECT_EQ(mappings.AtomsForBit(3, 42), std::vector<std::uint32_t>({1u, 5u}));
    EXPECT_TRUE(mappings.EnvironmentsForBit(2, 42).empty());
}

} // namespace test
} // namespace OEFP
