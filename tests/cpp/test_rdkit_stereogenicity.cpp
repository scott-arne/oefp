#include "oefp/rdkit_stereogenicity.h"

#include <gtest/gtest.h>
#include <oechem.h>

#include <set>
#include <vector>

using namespace OEFP;

namespace {

// Parse, perceive (ring/aromatic/hybridization), and return flagged heavy-atom
// indices (heavy-iteration order) for the potential-stereocenter vector.
std::set<std::size_t> stereo_atoms(const char* smiles) {
    OEChem::OEGraphMol mol;
    EXPECT_TRUE(OEChem::OESmilesToMol(mol, smiles)) << smiles;
    OEChem::OEFindRingAtomsAndBonds(mol);
    OEChem::OEAssignAromaticFlags(mol);
    OEChem::OEAssignHybridization(mol);
    const auto flags = rdkit_potential_stereogenicity(mol);
    std::set<std::size_t> out;
    for (std::size_t i = 0u; i < flags.atom_is_potential_stereocenter.size(); ++i) {
        if (flags.atom_is_potential_stereocenter[i]) out.insert(i);
    }
    return out;
}

}  // namespace

TEST(RDKitStereogenicityTest, ClassicTetrahedralCenters) {
    EXPECT_EQ(stereo_atoms("C[C@H](N)C(=O)O"), (std::set<std::size_t>{1}));  // L-alanine C1
    EXPECT_EQ(stereo_atoms("FC(Cl)Br"), (std::set<std::size_t>{1}));        // assigned-agnostic
}

TEST(RDKitStereogenicityTest, SimpleNegatives) {
    EXPECT_TRUE(stereo_atoms("c1ccccc1").empty());   // benzene
    EXPECT_TRUE(stereo_atoms("C1CCCCC1").empty());   // cyclohexane
    EXPECT_TRUE(stereo_atoms("CC(C)C").empty());     // isobutane
    EXPECT_TRUE(stereo_atoms("C1CC1").empty());      // cyclopropane
    EXPECT_TRUE(stereo_atoms("CC1CCCCC1").empty());  // methylcyclohexane (substituted C not stereogenic)
}
