#include "oefp/compute_context.h"

#include <gtest/gtest.h>

#include <oechem.h>

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
}

TEST(ComputeContextTest, EachIntermediateComputedAtMostOnce) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));
    ComputeContext ctx(mol);
    // Touch every accessor twice; only the seven first-touches compute.
    for (int pass = 0; pass < 2; ++pass) {
        ctx.RingPerceivedMol();
        ctx.HeavyAtomGraph();
        ctx.HeavyAtomDistances();   // reuses HeavyAtomGraph internally
        ctx.GasteigerAtomCharges();
        ctx.CrippenContributions();
        ctx.EStateIndices();
        ctx.LabuteAtomContributions();
    }
    EXPECT_EQ(ctx.ComputeCount(), 7u);  // one per intermediate, regardless of repeated access
}

// Each new accessor (EState indices, Labute per-atom contributions) computes
// once, returns a stable cached reference on repeated call, and bumps the
// compute count by exactly one — mirroring the existing five accessors.
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
}
