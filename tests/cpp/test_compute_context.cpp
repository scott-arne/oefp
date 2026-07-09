#include "oefp/compute_context.h"

#include <gtest/gtest.h>

#include <oechem.h>

#include <cmath>

using namespace OEFP;

TEST(ComputeContextTest, AccessorsCacheAndReturnStableReferences) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));  // aspirin
    ComputeContext ctx(mol);

    const auto& graph_a = ctx.HeavyAtomGraph();
    const auto& graph_b = ctx.HeavyAtomGraph();
    EXPECT_EQ(&graph_a, &graph_b);  // same cached object, computed once

    const auto& dist_a = ctx.HeavyAtomDistances();
    const auto& dist_b = ctx.HeavyAtomDistances();
    EXPECT_EQ(&dist_a, &dist_b);

    EXPECT_EQ(&ctx.RingPerceivedMol(), &ctx.RingPerceivedMol());
    EXPECT_EQ(&ctx.GasteigerAtomCharges(), &ctx.GasteigerAtomCharges());
    EXPECT_EQ(&ctx.CrippenContributions(), &ctx.CrippenContributions());
    EXPECT_EQ(&ctx.EStateIndices(), &ctx.EStateIndices());
    EXPECT_EQ(&ctx.LabuteAtomContributions(), &ctx.LabuteAtomContributions());
    EXPECT_EQ(&ctx.BCUTEigenvalues(), &ctx.BCUTEigenvalues());
}

TEST(ComputeContextTest, EachIntermediateComputedAtMostOnce) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));
    ComputeContext ctx(mol);
    // Touch every accessor twice; only the nine first-touches compute.
    // BCUTEigenvalues() reuses the heavy-atom graph, Gasteiger charges, and Crippen
    // contributions internally, so it adds exactly one computation of its own.
    for (int pass = 0; pass < 2; ++pass) {
        ctx.RingPerceivedMol();
        ctx.HeavyAtomGraph();
        ctx.HeavyAtomDistances();   // reuses HeavyAtomGraph internally
        ctx.GasteigerAtomCharges();
        ctx.RDKitGasteigerAtomCharges();
        ctx.CrippenContributions();
        ctx.EStateIndices();
        ctx.LabuteAtomContributions();
        ctx.BCUTEigenvalues();      // reuses graph/Gasteiger/Crippen internally
    }
    EXPECT_EQ(ctx.ComputeCount(), 9u);  // one per intermediate, regardless of repeated access
}

// Each new accessor (EState indices, Labute per-atom contributions, BCUT2D
// eigenvalues) computes once, returns a stable cached reference on repeated call,
// and bumps the compute count by exactly one — mirroring the existing accessors.
TEST(ComputeContextTest, NewAccessorsCacheAndBumpCountByOne) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));  // aspirin
    ComputeContext ctx(mol);

    const auto before_estate = ctx.ComputeCount();
    const auto& estate_a = ctx.EStateIndices();
    const auto after_first_estate = ctx.ComputeCount();
    const auto& estate_b = ctx.EStateIndices();
    EXPECT_EQ(&estate_a, &estate_b);                       // same cached object
    EXPECT_EQ(after_first_estate, before_estate + 1u);      // computed exactly once
    EXPECT_EQ(ctx.ComputeCount(), after_first_estate);      // repeat call does not recompute

    const auto before_labute = ctx.ComputeCount();
    const auto& labute_a = ctx.LabuteAtomContributions();
    const auto after_first_labute = ctx.ComputeCount();
    const auto& labute_b = ctx.LabuteAtomContributions();
    EXPECT_EQ(&labute_a, &labute_b);
    EXPECT_EQ(after_first_labute, before_labute + 1u);
    EXPECT_EQ(ctx.ComputeCount(), after_first_labute);

    // BCUTEigenvalues() reuses the heavy-atom graph, Gasteiger charges, and Crippen
    // contributions internally, so pre-warm those three; then its own first compute
    // must bump the count by exactly one and cache a stable reference thereafter.
    ctx.HeavyAtomGraph();
    ctx.GasteigerAtomCharges();
    ctx.CrippenContributions();
    const auto before_bcut = ctx.ComputeCount();
    const auto& bcut_a = ctx.BCUTEigenvalues();
    const auto after_first_bcut = ctx.ComputeCount();
    const auto& bcut_b = ctx.BCUTEigenvalues();
    EXPECT_EQ(&bcut_a, &bcut_b);
    EXPECT_EQ(after_first_bcut, before_bcut + 1u);
    EXPECT_EQ(ctx.ComputeCount(), after_first_bcut);
    EXPECT_TRUE(bcut_a.defined);  // aspirin has RDKit Gasteiger parameters
}

// The two Gasteiger accessors intentionally diverge on cumulenes: the RDKit
// accessor types the central sp atom "sp" (RDKit 2026.03.3), the Mordred
// accessor keeps "sp2" (Mordred 1.2.0 fidelity). CO2 has no H so the charges
// are exact/H-independent.
TEST(ComputeContextTest, RDKitGasteigerTypesCumuleneSpWhileMordredStaysSp2) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "O=C=O"));
    OEFP::ComputeContext ctx(mol);
    const auto& rdkit = ctx.RDKitGasteigerAtomCharges().charges;   // sp
    const auto& mordred = ctx.GasteigerAtomCharges().charges;       // sp2
    ASSERT_EQ(rdkit.size(), 3u);
    ASSERT_EQ(mordred.size(), 3u);
    // atom order O, C, O — carbon is the middle heavy atom.
    EXPECT_NEAR(rdkit[1], 0.3729, 1e-3);      // RDKit "C sp"
    EXPECT_NEAR(rdkit[0], -0.1865, 1e-3);
    EXPECT_GT(std::abs(mordred[1] - rdkit[1]), 1e-2);  // Mordred stays sp2 (differs)
}
