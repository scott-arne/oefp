#include "oefp/column_request.h"
#include "oefp/compute_context.h"
#include "oefp/descriptor_schema.h"
#include "oefp/mordred.h"

#include <gtest/gtest.h>
#include <oechem.h>

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

using namespace OEFP;

namespace {

std::vector<std::size_t> indices_for(const std::vector<std::string>& names) {
    const auto schema = MordredDescriptorSchema();
    std::vector<std::size_t> out;
    out.reserve(names.size());
    for (const auto& n : names) {
        out.push_back(schema->IndexOf(n));
    }
    return out;
}

// Assert that computing only the requested columns yields byte-identical values
// for every requested column compared with computing the full schema, and that
// pruning is subtractive: a spread of non-requested columns stays missing.
void expect_subset_matches_all(const char* smiles, const std::vector<std::string>& names) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, smiles)) << smiles;

    ComputeContext ctx_all(mol);
    const auto full = MakeMordredDescriptors(mol, ctx_all, ColumnRequest::All());

    ComputeContext ctx_sub(mol);
    const auto sub = MakeMordredDescriptors(mol, ctx_sub, ColumnRequest::Subset(indices_for(names)));

    const auto schema = MordredDescriptorSchema();
    std::unordered_set<std::size_t> requested;
    for (const auto& n : names) {
        const auto i = schema->IndexOf(n);
        requested.insert(i);
        ASSERT_EQ(sub.Has(i), full.Has(i)) << n;
        if (full.Has(i)) {
            EXPECT_EQ(sub.Value(i), full.Value(i)) << n;  // byte-identical
        }
    }

    // Unrequested columns must be MISSING in the pruned row (subtractive
    // pruning): sample a spread of non-requested indices and assert none is set.
    for (std::size_t i = 0u; i < schema->Size(); i += 37u) {
        if (requested.count(i) != 0u) {
            continue;
        }
        EXPECT_FALSE(sub.Has(i)) << "unrequested column " << i << " should be missing";
    }
}

}  // namespace

TEST(MordredPruningEquivalenceTest, SingleColumn) {
    expect_subset_matches_all("CC(=O)OC1=CC=CC=C1C(=O)O", {"MW"});
}

TEST(MordredPruningEquivalenceTest, WholeGroup) {
    expect_subset_matches_all("c1ccccc1", {"ATS0dv", "ATS1dv"});  // autocorrelation group
}

TEST(MordredPruningEquivalenceTest, CrossGroupDependency) {
    expect_subset_matches_all("C1CCCCC1", {"WPath"});  // depends on distance matrix
}

TEST(MordredPruningEquivalenceTest, MultiGroupMix) {
    expect_subset_matches_all("CCN", {"MW", "nAtom", "ATS0dv", "WPath", "SLogP"});
}

namespace {
// A representative panel spanning small, ring, hetero, and branched molecules.
constexpr const char* kPanel[] = {
    "CC(=O)OC1=CC=CC=C1C(=O)O",  // aspirin
    "c1ccccc1",                  // benzene
    "C1CCCCC1",                  // cyclohexane
    "CCN",                       // ethylamine
    "OC(=O)Cc1ccc(O)cc1",        // 4-hydroxyphenylacetic acid
    "C",                         // methane (degenerate rings/paths)
};

// Assert Subset==All for one requested column across the whole panel. This is
// the per-group guard: a wrong emitted_columns list or a missing dependency
// edge surfaces as a Subset!=All divergence on at least one panel molecule.
void expect_column_matches_across_panel(const std::vector<std::string>& names) {
    for (const auto* smiles : kPanel) {
        expect_subset_matches_all(smiles, names);
    }
}
}  // namespace

// One scenario per emitting group: request a single representative column from
// the group and prove the pruned value equals the full-schema value.
TEST(MordredPruningEquivalenceTest, PerGroupRepresentativeColumn) {
    expect_column_matches_across_panel({"nAcid"});       // FirstBatch
    expect_column_matches_across_panel({"MW"});          // Weight
    expect_column_matches_across_panel({"MPC2"});        // PathCount
    expect_column_matches_across_panel({"Kier1"});       // KappaShape (dep FirstBatch,PathCount)
    expect_column_matches_across_panel({"Xp-0d"});       // ChiPath
    expect_column_matches_across_panel({"Xch-3d"});      // ChiNonPath
    expect_column_matches_across_panel({"Zagreb1"});     // Zagreb
    expect_column_matches_across_panel({"VAdjMat"});     // VertexAdjacency
    expect_column_matches_across_panel({"BalabanJ"});    // BalabanJ
    expect_column_matches_across_panel({"BertzCT"});     // BertzCT
    expect_column_matches_across_panel({"GGI1"});        // TopologicalCharge
    expect_column_matches_across_panel({"MID"});         // MolecularId
    expect_column_matches_across_panel({"SpAbs_A"});     // AdjacencyMatrix
    expect_column_matches_across_panel({"SpAbs_D"});     // DistanceMatrix
    expect_column_matches_across_panel({"SpAbs_Dt"});    // DetourMatrix
    expect_column_matches_across_panel({"SpAbs_Dzv"});   // BaryszMatrix
    expect_column_matches_across_panel({"BCUTdv-1h"});   // BCUT
    expect_column_matches_across_panel({"MDEC-11"});     // MolecularDistanceEdge
    expect_column_matches_across_panel({"ABC"});         // ABCIndex
    expect_column_matches_across_panel({"RNCG"});        // CPSACharge
    expect_column_matches_across_panel({"ATS0dv"});      // Autocorrelation
    expect_column_matches_across_panel({"IC0"});         // InformationContent
    expect_column_matches_across_panel({"WPath"});       // Wiener
    expect_column_matches_across_panel({"Diameter"});    // TopologicalIndex
    expect_column_matches_across_panel({"ECIndex"});     // EccentricConnectivity
    expect_column_matches_across_panel({"nRing"});       // RingCount
    expect_column_matches_across_panel({"ETA_alpha"});   // ETA (dep RingCount)
    expect_column_matches_across_panel({"NsLi"});        // EState
    expect_column_matches_across_panel({"FilterItLogS"});// FilterItLogS
    expect_column_matches_across_panel({"fMF"});         // Framework
    expect_column_matches_across_panel({"MWC01"});       // WalkCount
    expect_column_matches_across_panel({"SZ"});          // Additive
    expect_column_matches_across_panel({"LabuteASA"});   // LabuteASA
    expect_column_matches_across_panel({"PEOE_VSA1"});   // PEOE_VSA
    expect_column_matches_across_panel({"SMR_VSA1"});    // SMR_VSA (dep LabuteASA)
    expect_column_matches_across_panel({"SlogP_VSA1"});  // SlogP_VSA (dep LabuteASA)
    expect_column_matches_across_panel({"VSA_EState1"}); // VSA_EState (dep LabuteASA)
    expect_column_matches_across_panel({"EState_VSA1"}); // EState_VSA (dep LabuteASA)
    expect_column_matches_across_panel({"GeomDiameter"});// LowCount3D (dep ThreeDContext)
    expect_column_matches_across_panel({"Mor01"});       // MoRSE (dep ThreeDContext)
    expect_column_matches_across_panel({"PNSA1"});       // CPSASurface (dep ThreeDContext)
}

// Dependency-without-dependent scenarios: request only the DEPENDENT group's
// column and prove the value still matches All(). This proves the dependency
// group runs to populate its artifact even when none of its own columns is
// requested (its artifact is populated request-independently).
TEST(MordredPruningEquivalenceTest, DependencyClosureRunsWithoutDependentColumns) {
    // KappaShape depends on FirstBatch (heavy_atoms) and PathCount (path counts).
    // Requesting only Kier1 must NOT emit any FirstBatch or PathCount column, yet
    // Kier1 must equal its full-schema value.
    expect_column_matches_across_panel({"Kier1"});
    // ETA depends on RingCount's ring total.
    expect_column_matches_across_panel({"ETA_alpha"});
    // The VSA families depend on LabuteASA's artifact.
    expect_column_matches_across_panel({"SMR_VSA1"});
    expect_column_matches_across_panel({"SlogP_VSA1"});
    expect_column_matches_across_panel({"VSA_EState1"});
    expect_column_matches_across_panel({"EState_VSA1"});
    // The 3D groups depend on the (possibly absent) 3D context artifact. On the
    // 2D panel these are missing in both All() and Subset; the equivalence still
    // holds (Has()==false on both sides) and the producer runs in the closure.
    expect_column_matches_across_panel({"GeomDiameter"});
    expect_column_matches_across_panel({"Mor01"});
    expect_column_matches_across_panel({"PNSA1"});
}

// A wide multi-group request mixing many groups and their dependencies.
TEST(MordredPruningEquivalenceTest, WideMultiGroupMix) {
    expect_column_matches_across_panel(
        {"MW", "nAtom", "ATS0dv", "WPath", "SLogP", "Kier1", "ETA_alpha", "SMR_VSA1",
         "BCUTdv-1h", "SpAbs_Dzv", "GGI1", "nRing", "MWC01", "IC0", "ABC"});
}
