#include "oefp/molecular_properties.h"

#include <gtest/gtest.h>

#include <oechem.h>

using namespace OEFP;

namespace {
OEChem::OEGraphMol mol_from_smiles(const char* smiles) {
    OEChem::OEGraphMol mol;
    EXPECT_TRUE(OEChem::OESmilesToMol(mol, smiles));
    return mol;
}
} // namespace

TEST(MolecularPropertiesTest, EthanolWeightsAndCounts) {
    const auto ethanol = mol_from_smiles("CCO");
    EXPECT_NEAR(ExactMolecularWeight(ethanol), 46.04186, 1e-4);
    EXPECT_NEAR(AverageMolecularWeight(ethanol), ExactMolecularWeight(ethanol) / 9.0, 1e-9);
    EXPECT_EQ(HeavyAtomCount(ethanol), 3u);
    EXPECT_EQ(TotalAtomCount(ethanol), 9u);
}

TEST(MolecularPropertiesTest, EmptyMoleculeAverageIsZero) {
    OEChem::OEGraphMol empty;
    EXPECT_DOUBLE_EQ(AverageMolecularWeight(empty), 0.0);
    EXPECT_EQ(TotalAtomCount(empty), 0u);
}
