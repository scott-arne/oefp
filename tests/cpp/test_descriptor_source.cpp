#include "oefp/descriptor_source.h"
#include "oefp/mordred.h"

#include <gtest/gtest.h>

#include <oechem.h>

using namespace OEFP;

TEST(MordredDescriptorSourceTest, SchemaMatchesMordredSchema) {
    MordredDescriptorSource source;
    EXPECT_EQ(source.Schema()->SchemaId(), MordredDescriptorSchema()->SchemaId());
}

TEST(MordredDescriptorSourceTest, ComputeMatchesFreeFunction) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CCO"));
    MordredDescriptorSource source;
    const auto row = source.Compute(mol);
    EXPECT_DOUBLE_EQ(row.Float("MW"), MakeMordredDescriptors(mol).Float("MW"));
}
