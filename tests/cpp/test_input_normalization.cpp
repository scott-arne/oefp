#include "oefp/input_normalization.h"
#include <gtest/gtest.h>
#include <oechem.h>
#include <set>
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

TEST(InputNormalizationTest, PreservesExplicitHydrogens) {
    // Hydrogen suppression is intentionally NOT part of normalization: bracket
    // hydrogens must survive so the shared Gasteiger model is not regressed on
    // stereo inputs. Explicit-H handling stays with the individual sources.
    const auto norm = normalize_molecule(parse("[H]O[H]"));
    EXPECT_EQ(explicit_h(norm), 2u);
    EXPECT_EQ(norm.NumAtoms(), 3u);  // oxygen plus both explicit hydrogens
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

    // Non-nitro molecules must be untouched: the guard must not over-fire on an
    // amide, amine, pyridine, or azide (whose central N is legitimately +1 in
    // the input). Normalization must leave every formal charge exactly as parsed
    // and must introduce no -1 oxygen (the sole product of the nitro rewrite).
    for (const char* smi : {"CC(=O)N", "CCN", "c1ccncc1", "CCN=[N+]=[N-]"}) {
        const auto input = parse(smi);
        const auto norm = normalize_molecule(input);
        std::multiset<int> before, after;
        int norm_o_minus = 0;
        for (OESystem::OEIter<OEChem::OEAtomBase> a = input.GetAtoms(); a; ++a) {
            before.insert(a->GetFormalCharge());
        }
        for (OESystem::OEIter<OEChem::OEAtomBase> a = norm.GetAtoms(); a; ++a) {
            after.insert(a->GetFormalCharge());
            if (a->GetAtomicNum() == 8 && a->GetFormalCharge() == -1) ++norm_o_minus;
        }
        EXPECT_EQ(before, after) << smi << ": formal charges must be unchanged";
        EXPECT_EQ(norm_o_minus, 0) << smi << ": nitro rewrite must not have fired";
    }

    // Double application is a no-op: re-normalizing an already-charged nitro
    // leaves the +1 N and single -1 O in place (the guard skips charged N).
    const auto once = normalize_molecule(parse("CN(=O)=O"));
    const auto twice = normalize_molecule(once);
    int twice_n = 0, twice_o_minus = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> a = twice.GetAtoms(); a; ++a) {
        if (a->GetAtomicNum() == 7) twice_n = a->GetFormalCharge();
        if (a->GetAtomicNum() == 8 && a->GetFormalCharge() == -1) ++twice_o_minus;
    }
    EXPECT_EQ(twice_n, 1);
    EXPECT_EQ(twice_o_minus, 1);
}
