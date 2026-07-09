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
    // Touch every accessor twice; only the eight first-touches compute.
    // BCUTEigenvalues() reuses the heavy-atom graph, Gasteiger charges, and Crippen
    // contributions internally, so it adds exactly one computation of its own.
    for (int pass = 0; pass < 2; ++pass) {
        ctx.RingPerceivedMol();
        ctx.HeavyAtomGraph();
        ctx.HeavyAtomDistances();   // reuses HeavyAtomGraph internally
        ctx.GasteigerAtomCharges();
        ctx.CrippenContributions();
        ctx.EStateIndices();
        ctx.LabuteAtomContributions();
        ctx.BCUTEigenvalues();      // reuses graph/Gasteiger/Crippen internally
    }
    EXPECT_EQ(ctx.ComputeCount(), 8u);  // one per intermediate, regardless of repeated access
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

    // BCUTEigenvalues() reuses the heavy-atom graph, Gasteiger charges, and
    // Crippen contributions internally, so pre-warm those three; then its own first
    // compute must bump the count by exactly one and cache a stable reference thereafter.
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

// The RDKit-faithful Gasteiger accessor types cumulene centres "sp" (RDKit's
// ComputeGasteigerCharges behavior, which Mordred 1.2.0 delegates to). CO2 has
// no H so the charges are exact/H-independent.
TEST(ComputeContextTest, GasteigerTypesCumuleneSp) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "O=C=O"));
    OEFP::ComputeContext ctx(mol);
    const auto& charges = ctx.GasteigerAtomCharges().charges;
    ASSERT_EQ(charges.size(), 3u);
    // atom order O, C, O — carbon is the middle heavy atom, typed "sp"
    EXPECT_NEAR(charges[1], 0.3729, 1e-3);
    EXPECT_NEAR(charges[0], -0.1865, 1e-3);
    EXPECT_NEAR(charges[2], -0.1865, 1e-3);
}

// RDKit NaNs only the connected component containing a parameterless atom; a
// mixture's well-formed component stays finite. C[Na].CCO -> the C-Na component
// (atoms 0,1) is NaN, the ethanol component (atoms 2,3,4) is finite AND equals
// ethanol computed alone.
TEST(ComputeContextTest, GasteigerNaNsOnlyTheOffendingComponent) {
    OEChem::OEGraphMol mix;
    ASSERT_TRUE(OEChem::OESmilesToMol(mix, "C[Na].CCO"));
    OEFP::ComputeContext ctx_mix(mix);
    const auto& mixed = ctx_mix.GasteigerAtomCharges().charges;
    ASSERT_EQ(mixed.size(), 5u);
    EXPECT_TRUE(std::isnan(mixed[0]));   // methyl C on the C-Na component
    EXPECT_TRUE(std::isnan(mixed[1]));   // Na
    EXPECT_FALSE(std::isnan(mixed[2]));  // ethanol C
    EXPECT_FALSE(std::isnan(mixed[3]));
    EXPECT_FALSE(std::isnan(mixed[4]));

    OEChem::OEGraphMol etoh;
    ASSERT_TRUE(OEChem::OESmilesToMol(etoh, "CCO"));
    OEFP::ComputeContext ctx_etoh(etoh);
    const auto& alone = ctx_etoh.GasteigerAtomCharges().charges;
    ASSERT_EQ(alone.size(), 3u);
    for (std::size_t i = 0u; i < 3u; ++i) {
        EXPECT_NEAR(mixed[i + 2u], alone[i], 1e-9);  // component-independent
    }
}

// The RDKit-faithful accessor reproduces RDKit's molecule-wide NaN: when any
// charge-flowing atom lacks a Gasteiger parameter — a no-parameter element
// (Na, Se) or a hypervalent main-group atom ([SiH5-]) — the ENTIRE charge vector
// is NaN, so downstream PEOE_VSA binning routes it to the open tail bin. Normal
// organics, including hypervalent-looking but parameterized sulfones, stay finite.
TEST(ComputeContextTest, GasteigerNaNsMoleculesLackingParameters) {
    for (const auto* smiles : {"C[Na]", "O=[Se]=O", "[SiH5-]"}) {
        OEChem::OEGraphMol mol;
        ASSERT_TRUE(OEChem::OESmilesToMol(mol, smiles)) << smiles;
        OEFP::ComputeContext ctx(mol);
        const auto& charges = ctx.GasteigerAtomCharges().charges;
        ASSERT_FALSE(charges.empty()) << smiles;
        for (const double q : charges) {
            EXPECT_TRUE(std::isnan(q)) << smiles << " expected all-NaN, got " << q;
        }
    }
    // No over-fire: normal organics and a parameterized sulfone (hypervalent-looking
    // but S has RDKit "so2" parameters, connectivity == 4) stay finite.
    for (const auto* smiles : {"CCO", "O=C=O", "CS(=O)(=O)C"}) {
        OEChem::OEGraphMol mol;
        ASSERT_TRUE(OEChem::OESmilesToMol(mol, smiles)) << smiles;
        OEFP::ComputeContext ctx(mol);
        const auto& charges = ctx.GasteigerAtomCharges().charges;
        ASSERT_FALSE(charges.empty()) << smiles;
        for (const double q : charges) {
            EXPECT_TRUE(std::isfinite(q)) << smiles << " expected finite, got " << q;
        }
    }
}
