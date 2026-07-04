#include "oefp/descriptor_source.h"
#include "oefp/molecular_properties.h"
#include "oefp/mordred.h"

#include <gtest/gtest.h>

#include <cstdint>

#include <oechem.h>
#include <oemolprop.h>

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

TEST(OpenEyePropertyDescriptorSourceTest, TaggedColumnsMatchMordredByConstruction) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));  // aspirin
    OpenEyePropertyDescriptorSource oe;
    MordredDescriptorSource mordred;
    const auto oe_row = oe.Compute(mol);
    const auto md_row = mordred.Compute(mol);
    EXPECT_DOUBLE_EQ(oe_row.Float("MolecularWeight"), md_row.Float("MW"));
    EXPECT_DOUBLE_EQ(oe_row.Float("TopologicalPSA"), md_row.Float("TopoPSA"));
    EXPECT_EQ(oe_row.Int("HeavyAtomCount"), md_row.Int("nHeavyAtom"));
    EXPECT_EQ(oe_row.Int("LipinskiHBD"), md_row.Int("nHBDon"));
    EXPECT_EQ(oe_row.Int("HBA"), md_row.Int("nHBAcc"));
}

TEST(OpenEyePropertyDescriptorSourceTest, SchemaTagsMatchMordred) {
    OpenEyePropertyDescriptorSource oe;
    const auto& schema = *oe.Schema();
    EXPECT_EQ(schema.Definition(schema.IndexOf("MolecularWeight")).canonical_id,
              "quantity:exact_molecular_weight");
    // OpenEye-unique columns are intentionally untagged (never deduplicated).
    EXPECT_TRUE(schema.Definition(schema.IndexOf("XLogP")).canonical_id.empty());
    EXPECT_TRUE(schema.Definition(schema.IndexOf("RotatableBondCount")).canonical_id.empty());
}

TEST(OpenEyePropertyDescriptorSourceTest, UntaggedColumnsMatchOpenEyeToolkit) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));  // aspirin
    OpenEyePropertyDescriptorSource oe;
    const auto row = oe.Compute(mol);
    float xlogp = 0.0f;
    ASSERT_TRUE(OEMolProp::OEGetXLogP(mol, xlogp));
    EXPECT_DOUBLE_EQ(row.Float("XLogP"), static_cast<double>(xlogp));
    EXPECT_EQ(row.Int("RotatableBondCount"),
              static_cast<std::int64_t>(OEMolProp::OEGetRotatableBondCount(mol)));
    EXPECT_EQ(row.Int("AromaticRingCount"),
              static_cast<std::int64_t>(OEMolProp::OEGetAromaticRingCount(mol)));
}
