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

TEST(MolecularPropertiesTest, IsotopeHydrogenCountedOnce) {
    // OESuppressHydrogens retains isotope hydrogens ([2H]) as explicit atoms, so
    // the weight/count helpers must count such a hydrogen exactly once with its
    // isotope mass, not double-count it via the heavy atom's GetTotalHCount.
    const auto heavy_water = mol_from_smiles("[2H]O[H]");  // one deuterium, one protium
    EXPECT_NEAR(ExactMolecularWeight(heavy_water), 19.01684, 1e-4);  // O + D + H, each once
    EXPECT_EQ(TotalAtomCount(heavy_water), 3u);  // O plus two hydrogens
    EXPECT_EQ(HeavyAtomCount(heavy_water), 1u);  // just the oxygen
    // HeavyAtomMolWt strips all hydrogen mass, leaving the oxygen average weight.
    EXPECT_NEAR(heavy_atom_standard_weight(heavy_water), OEChem::OEGetAverageWeight(8u), 1e-4);
}

TEST(MolecularPropertiesTest, HydrogenOnlyAndMixtureCountedInFull) {
    // Weight/count run on the un-suppressed molecule, so a hydrogen-only molecule
    // (no heavy atom to hold implicit hydrogens) and an H-only fragment of a
    // mixture are counted in full rather than collapsed.
    const auto h2 = mol_from_smiles("[H][H]");
    EXPECT_NEAR(ExactMolecularWeight(h2), 2.01565, 1e-4);
    EXPECT_EQ(TotalAtomCount(h2), 2u);
    EXPECT_EQ(HeavyAtomCount(h2), 0u);

    const auto mixture = mol_from_smiles("[H][H].CCO");  // molecular hydrogen plus ethanol
    EXPECT_NEAR(ExactMolecularWeight(mixture),
                ExactMolecularWeight(h2) + ExactMolecularWeight(mol_from_smiles("CCO")), 1e-4);
    EXPECT_EQ(TotalAtomCount(mixture), 11u);  // two (H2) plus nine (ethanol)
}
