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

std::set<std::size_t> stereo_bond_atoms(const char* smiles) {
    OEChem::OEGraphMol mol;
    EXPECT_TRUE(OEChem::OESmilesToMol(mol, smiles)) << smiles;
    OEChem::OEFindRingAtomsAndBonds(mol);
    OEChem::OEAssignAromaticFlags(mol);
    OEChem::OEAssignHybridization(mol);
    const auto flags = rdkit_potential_stereogenicity(mol);
    std::set<std::size_t> out;
    for (std::size_t i = 0u; i < flags.atom_on_potential_stereo_bond.size(); ++i) {
        if (flags.atom_on_potential_stereo_bond[i]) out.insert(i);
    }
    return out;
}

}  // namespace

TEST(RDKitStereogenicityTest, ClassicTetrahedralCenters) {
    EXPECT_EQ(stereo_atoms("C[C@H](N)C(=O)O"), (std::set<std::size_t>{1}));  // L-alanine C1
    EXPECT_EQ(stereo_atoms("FC(Cl)Br"), (std::set<std::size_t>{1}));        // assigned-agnostic
}

// Specified-stereo atom symbols: a specified centre contributes a parity symbol
// (_CW/_CCW), not a unique per-index label, so ring/dependent centres are
// distinguished by the RELATIONSHIP between neighbouring specified centres.
// meso vs like configurations therefore give different centre sets. Redundant
// (pseudo-asymmetric) specified tags that OE preserves but RDKit strips at parse
// time are cleaned up by the duplicate rule. All expected sets verified against
// the RDKit 2026.03.3 SPS oracle (FindMolChiralCenters, includeUnassigned=True,
// includeCIP=False, useLegacyImplementation=False, on the sanitized molecule).
TEST(RDKitStereogenicityTest, SpecifiedStereoAtomSymbols) {
    // Opposite-parity (meso-like): central atom 3 is NOT a stereocenter because
    // the two [C@H]/[C@@H] arms become equivalent under the parity symbols.
    EXPECT_EQ(stereo_atoms("C[C@H](F)C(Cl)(Br)[C@@H](F)C"),
              (std::set<std::size_t>{1, 6}));
    // Same-parity: central atom 3 IS a stereocenter (arms distinguishable).
    EXPECT_EQ(stereo_atoms("C[C@H](F)C(Cl)(Br)[C@H](F)C"),
              (std::set<std::size_t>{1, 3, 6}));
    // Butane-2,3-diol: both centres survive regardless of relative parity.
    EXPECT_EQ(stereo_atoms("C[C@H](O)[C@@H](O)C"), (std::set<std::size_t>{1, 3}));
    EXPECT_EQ(stereo_atoms("C[C@H](O)[C@H](O)C"), (std::set<std::size_t>{1, 3}));
    // Pentitol with a specified middle carbon: the middle is a redundant
    // pseudo-asymmetric tag (enantiomeric arms) and is dropped -> {2,6}.
    EXPECT_EQ(stereo_atoms("OC[C@@H](O)[C@@H](O)[C@H](O)CO"),
              (std::set<std::size_t>{2, 6}));
    // Trihydroxyglutaric acid: the middle carbon's arms are distinguishable
    // (COOH-terminated), so the middle IS retained -> {3,5,7}.
    EXPECT_EQ(stereo_atoms("OC(=O)[C@@H](O)[C@H](O)[C@@H](O)C(=O)O"),
              (std::set<std::size_t>{3, 5, 7}));
    // 2,3,4-trichloropentane: middle carbon retained (arms distinguishable).
    EXPECT_EQ(stereo_atoms("C[C@@H](Cl)[C@H](Cl)[C@@H](Cl)C"),
              (std::set<std::size_t>{1, 3, 5}));
}

TEST(RDKitStereogenicityTest, SimpleNegatives) {
    EXPECT_TRUE(stereo_atoms("c1ccccc1").empty());   // benzene
    EXPECT_TRUE(stereo_atoms("C1CCCCC1").empty());   // cyclohexane
    EXPECT_TRUE(stereo_atoms("CC(C)C").empty());     // isobutane
    EXPECT_TRUE(stereo_atoms("C1CC1").empty());      // cyclopropane
    EXPECT_TRUE(stereo_atoms("CC1CCCCC1").empty());  // methylcyclohexane (substituted C not stereogenic)
}

TEST(RDKitStereogenicityTest, ParaAndCageStereocenters) {
    EXPECT_EQ(stereo_atoms("CC1CCC(C)CC1"),
              (std::set<std::size_t>{1, 4}));                 // 1,4-dimethylcyclohexane
    EXPECT_EQ(stereo_atoms("C1C2CC3CC1CC(C2)C3"),
              (std::set<std::size_t>{1, 3, 5, 7}));           // adamantane bridgeheads
    EXPECT_EQ(stereo_atoms("C12C3C4C1C5C2C3C45"),
              (std::set<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7}));  // cubane (all 8)
    EXPECT_EQ(stereo_atoms("C1CC23CCC(C1)(CC2)CC3"),
              (std::set<std::size_t>{2, 5}));                 // bridged bicyclo bridgeheads
}

TEST(RDKitStereogenicityTest, HypervalentHydride) {
    EXPECT_EQ(stereo_atoms("[AlH4-]"), (std::set<std::size_t>{0}));  // flagged despite SPS=0
    EXPECT_EQ(stereo_atoms("[SiH4]"), (std::set<std::size_t>{0}));   // total-degree-4 kept
    EXPECT_TRUE(stereo_atoms("[BH4-]").empty());                     // boron fails element gate
    EXPECT_TRUE(stereo_atoms("[PH5]").empty());                      // total-degree-5 -> TBP, filtered
}

// Special gate branches (isotope symbol, three-coordinate N, S/Se lone pair)
// verified against the RDKit oracle; not covered by the plan's pinned vectors.
TEST(RDKitStereogenicityTest, SpecialGateCases) {
    EXPECT_EQ(stereo_atoms("[13CH3]C(O)C"), (std::set<std::size_t>{1}));  // isotope desymmetrizes
    EXPECT_TRUE(stereo_atoms("CC(O)C").empty());                         // same skeleton, no isotope
    EXPECT_EQ(stereo_atoms("CN1CC1C"), (std::set<std::size_t>{1, 3}));   // N in 3-ring + ring C
    EXPECT_EQ(stereo_atoms("C1CN2CCC1CC2"), (std::set<std::size_t>{2, 5}));  // bridgehead N (quinuclidine)
    EXPECT_EQ(stereo_atoms("O=S(C)CC"), (std::set<std::size_t>{1}));     // sulfoxide S lone pair
    EXPECT_TRUE(stereo_atoms("C[S+](C)C").empty());                      // sulfonium: 3 equal methyls
}

TEST(RDKitStereogenicityTest, PotentialStereoDoubleBonds) {
    EXPECT_EQ(stereo_bond_atoms("CC=CCO"), (std::set<std::size_t>{1, 2}));    // crotyl
    EXPECT_EQ(stereo_bond_atoms("CC=CC"), (std::set<std::size_t>{1, 2}));     // 2-butene
    EXPECT_EQ(stereo_bond_atoms("O=CC=CC=O"), (std::set<std::size_t>{2, 3})); // but-2-enedial
    EXPECT_EQ(stereo_bond_atoms("CN=NC"), (std::set<std::size_t>{1, 2}));     // diazene N=N
    EXPECT_EQ(stereo_bond_atoms("c1ccccc1N=Nc1ccccc1"),
              (std::set<std::size_t>{6, 7}));                                 // azobenzene N=N
}

TEST(RDKitStereogenicityTest, NonStereoDoubleBonds) {
    EXPECT_TRUE(stereo_bond_atoms("C=C").empty());         // ethylene
    EXPECT_TRUE(stereo_bond_atoms("CC=C").empty());        // propene (=CH2 end has 2 H)
    EXPECT_TRUE(stereo_bond_atoms("CC(C)=C").empty());     // isobutene (2 identical CH3)
    EXPECT_TRUE(stereo_bond_atoms("CC(C)=O").empty());     // acetone (terminal O)
    EXPECT_TRUE(stereo_bond_atoms("c1ccccc1C=C").empty()); // styrene vinyl
}

// Additional fused/bridged/spiro topologies verified against the RDKit oracle,
// guarding the port beyond the plan's pinned cages.
TEST(RDKitStereogenicityTest, ExtraRingTopologies) {
    EXPECT_EQ(stereo_atoms("C1CC2CCC1CC2"), (std::set<std::size_t>{2, 5}));       // bicyclo[2.2.2]octane
    EXPECT_EQ(stereo_atoms("C1CCC2CCCCC2C1"), (std::set<std::size_t>{3, 8}));     // decalin ring fusion
    EXPECT_EQ(stereo_atoms("C1CC2CCC3CC1CC(C2)C3"),
              (std::set<std::size_t>{2, 5, 7, 9}));                              // diamantane bridgeheads
    EXPECT_TRUE(stereo_atoms("C1CC2(CC1)CCCCC2").empty());                       // spiro (no dependent pair)
    EXPECT_TRUE(stereo_atoms("C1CCC(CC1)C1CCCCC1").empty());                     // bicyclohexyl (single each)
}

// The legacy findPotentialStereoBonds ignores ring double bonds completely (any
// size), so SPS never flags an unmarked ring C=C. The new-algorithm bond path
// (ring>=8) would WRONGLY flag cyclooctene/cyclodecene -- this test guards that.
TEST(RDKitStereogenicityTest, RingDoubleBondsNotFlagged) {
    EXPECT_TRUE(stereo_bond_atoms("C1CCC=CC1").empty());     // cyclohexene (6)
    EXPECT_TRUE(stereo_bond_atoms("C1CCCC=CCC1").empty());   // cyclooctene (8)
    EXPECT_TRUE(stereo_bond_atoms("C1CCCC=CCCCC1").empty()); // cyclodecene (10)
}

// RDKit clears specified stereo on ring double bonds in rings smaller than
// MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO (8) because trans is geometrically
// impossible in small rings. Ring size >= 8 retains specified stereo.
TEST(RDKitStereogenicityTest, SpecifiedStereoDoubleBonds) {
    EXPECT_EQ(stereo_bond_atoms("F/C=C/F"), (std::set<std::size_t>{1, 2}));        // acyclic, flagged
    EXPECT_EQ(stereo_bond_atoms("C1CCCC/C=C/C1"), (std::set<std::size_t>{5, 6}));  // 8-ring, flagged
    EXPECT_TRUE(stereo_bond_atoms("C1CCC/C=C/C1").empty());                        // 7-ring, NOT flagged
    EXPECT_TRUE(stereo_bond_atoms("C1CC/C=C/C1").empty());                         // 6-ring, NOT flagged
}

// flagRingStereo otherFoundByBondCount branch: a ring stereocenter is supported
// by an exocyclic (out-of-ring) double bond at the across-ring atom. The support
// uses the LOCAL bond gate (degree/H/ring-size only), NOT distinguishability, so
// an exocyclic =C(C)C with two identical methyls still qualifies. All expected
// sets verified against the RDKit 2026.03.3 oracle (FindMolChiralCenters,
// includeUnassigned=True, includeCIP=False, useLegacyImplementation=False).
TEST(RDKitStereogenicityTest, ExocyclicDoubleBondRingStereo) {
    EXPECT_EQ(stereo_atoms("CC1CCC(=C(F)Cl)CC1"), (std::set<std::size_t>{1}));  // para exocyclic C=CFCl
    EXPECT_EQ(stereo_atoms("CC1CCC(=C(C)C)CC1"), (std::set<std::size_t>{1}));   // identical methyls still support
    EXPECT_EQ(stereo_atoms("CC1CCC(=CC)CC1"), (std::set<std::size_t>{1}));      // exocyclic =CHCH3
    EXPECT_EQ(stereo_atoms("CC1CC(=C(F)Cl)C1"), (std::set<std::size_t>{1}));    // 4-ring, exocyclic bond acyclic
    EXPECT_EQ(stereo_atoms("C1CC(=C(F)Cl)CCC1C"), (std::set<std::size_t>{8}));  // support with methyl at far end
    EXPECT_EQ(stereo_atoms("CC1CC(=C(F)Cl)CC(=C(F)Cl)C1"),
              (std::set<std::size_t>{1}));                    // divisor-3 bond branch (two exocyclic bonds)
    EXPECT_TRUE(stereo_atoms("CC1CCC(=O)CC1").empty());        // terminal =O: far end degree 1 fails gate
    EXPECT_TRUE(stereo_atoms("CC1CCC(=S)CC1").empty());        // terminal =S: far end degree 1 fails gate
    EXPECT_TRUE(stereo_atoms("C1CCC(=C(F)Cl)CC1").empty());    // no ring substituent: across atom not a candidate
    EXPECT_TRUE(stereo_atoms("C1CC2(CCC1)CCC(=C(F)Cl)CC2").empty());  // spiro: across atom not a candidate
    EXPECT_TRUE(stereo_atoms("CC1CCCCC1").empty());            // methylcyclohexane regression guard (no support)
}
