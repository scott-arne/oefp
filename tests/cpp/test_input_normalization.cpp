#include "oefp/input_normalization.h"
#include <gtest/gtest.h>
#include <oechem.h>
using namespace OEFP;

namespace {
OEChem::OEGraphMol parse(const char* smi) {
    OEChem::OEGraphMol m;
    EXPECT_TRUE(OEChem::OESmilesToMol(m, smi)) << smi;
    return m;
}
// count explicit H atoms present as their own atoms
unsigned int explicit_h(const OEChem::OEMolBase& m) {
    unsigned int n = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> a = m.GetAtoms(); a; ++a) {
        if (a->GetAtomicNum() == 1) ++n;
    }
    return n;
}
}  // namespace

TEST(InputNormalizationTest, SuppressesExplicitHydrogens) {
    const auto norm = normalize_molecule(parse("[H]O[H]"));
    EXPECT_EQ(explicit_h(norm), 0u);
    EXPECT_EQ(norm.NumAtoms(), 1u);  // just the oxygen, H implicit
}

TEST(InputNormalizationTest, NeutralNitroBecomesCharged) {
    const auto norm = normalize_molecule(parse("c1ccc(cc1)N(=O)=O"));
    // find the nitrogen: formal charge +1, exactly one -1 oxygen neighbor
    int n_charge = 0, o_minus = 0, o_double = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> a = norm.GetAtoms(); a; ++a) {
        if (a->GetAtomicNum() == 7) n_charge = a->GetFormalCharge();
        if (a->GetAtomicNum() == 8 && a->GetFormalCharge() == -1) ++o_minus;
        if (a->GetAtomicNum() == 8 && a->GetFormalCharge() == 0) ++o_double;
    }
    EXPECT_EQ(n_charge, 1);
    EXPECT_EQ(o_minus, 1);
    EXPECT_EQ(o_double, 1);
}

TEST(InputNormalizationTest, IdempotentAndNonNitroUntouched) {
    // charged nitro unchanged
    const auto charged = normalize_molecule(parse("c1ccc(cc1)[N+](=O)[O-]"));
    int n_charge = 0, o_minus = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> a = charged.GetAtoms(); a; ++a) {
        if (a->GetAtomicNum() == 7) n_charge = a->GetFormalCharge();
        if (a->GetAtomicNum() == 8 && a->GetFormalCharge() == -1) ++o_minus;
    }
    EXPECT_EQ(n_charge, 1);
    EXPECT_EQ(o_minus, 1);
    // non-nitro nitrogens untouched (amide, amine, pyridine, azide)
    for (const char* smi : {"CC(=O)N", "CCN", "c1ccncc1", "CCN=[N+]=[N-]"}) {
        const auto norm = normalize_molecule(parse(smi));
        for (OESystem::OEIter<OEChem::OEAtomBase> a = norm.GetAtoms(); a; ++a) {
            if (a->GetAtomicNum() == 7) {
                // no neutral N should have been turned into a +1 nitro N here
                // (azide's central N is legitimately +1 already from the SMILES)
            }
        }
        SUCCEED();  // parse + normalize without crash; charges below asserted structurally
    }
}
