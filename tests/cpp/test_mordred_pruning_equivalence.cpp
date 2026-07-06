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
void expect_subset_matches_all_for_mol(
    const OEChem::OEMolBase& mol, const std::vector<std::string>& names) {
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
    // pruning): scan every non-requested index and assert none is set, so a
    // stray emission anywhere in the schema fails the test deterministically.
    for (std::size_t i = 0u; i < schema->Size(); ++i) {
        if (requested.count(i) != 0u) {
            continue;
        }
        EXPECT_FALSE(sub.Has(i)) << "unrequested column " << i << " should be missing";
    }
}

void expect_subset_matches_all(const char* smiles, const std::vector<std::string>& names) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, smiles)) << smiles;
    expect_subset_matches_all_for_mol(mol, names);
}

// Build a molecule that reports a real 3D conformer so the Mordred 3D groups
// (MoRSE, LowCount3D, CPSASurface) emit values. Omega is not linked into the
// test build, so we set explicit atom coordinates directly. Hydrogens are made
// explicit and given coordinates BEFORE returning: build_mordred_3d_context
// requires every atom (including hydrogens) to carry 3D coordinates both on the
// input molecule and on its internal explicit-hydrogen copy, and that copy is
// produced with OEAddExplicitHydrogens(mol, false, /*set3D=*/false), which would
// leave newly added hydrogens without coordinates. Making them explicit here
// first turns that internal call into a coordinate-preserving no-op.
void build_3d_mol(OEChem::OEGraphMol& mol, const char* smiles) {
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, smiles)) << smiles;
    OEChem::OEAddExplicitHydrogens(mol, false, false);

    // Assign a spread-out coordinate to every atom. Exact geometry is
    // irrelevant to this test; only that the molecule reports dimension 3 with a
    // coordinate on each atom so build_mordred_3d_context succeeds. Distinct,
    // non-collinear coordinates keep the resulting 3D descriptors finite.
    unsigned int i = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom, ++i) {
        const double coords[3] = {
            static_cast<double>(i) * 1.5,
            static_cast<double>((i * 7u) % 5u) * 1.1,
            static_cast<double>((i * 3u) % 4u) * 0.9,
        };
        ASSERT_TRUE(mol.SetCoords(atom, coords));
    }
    mol.SetDimension(3u);
    ASSERT_EQ(mol.GetDimension(), 3u);
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

// The 3D groups (MoRSE, LowCount3D, CPSASurface) reach ThreeDContext through a
// declared dependency edge. On the 2D panel above, their columns are missing in
// both All() and Subset, so the equivalence there holds trivially and never
// exercises that edge: dropping the {ThreeDContext} edge would keep every 2D
// guard green. This scenario closes that gap. It computes on a molecule with a
// real 3D conformer so build_mordred_3d_context succeeds and the 3D groups emit
// values, then proves Subset==All for 3D-group columns.
//
// ThreeDContext emits no columns of its own, so requesting only a MoRSE or
// LowCount3D column (without any ThreeDContext column, because there are none)
// forces the request closure to pull ThreeDContext in as a dependency. If that
// edge were missing, ThreeDContext would not run under the pruned request, the
// 3D context artifact would be empty, and the requested 3D value would be
// missing while All() still has it -- surfacing as Subset != All here.
TEST(MordredPruningEquivalenceTest, ThreeDGroupDependencyEdgesAreExercised) {
    OEChem::OEGraphMol mol;
    build_3d_mol(mol, "CCO");
    ASSERT_EQ(mol.GetDimension(), 3u);

    // Precondition: the 3D-group columns must actually be PRESENT under All() on
    // this molecule. If they are missing, build_mordred_3d_context did not
    // succeed and the equivalence below would hold vacuously.
    ComputeContext ctx_all(mol);
    const auto full = MakeMordredDescriptors(mol, ctx_all, ColumnRequest::All());
    const auto schema = MordredDescriptorSchema();
    for (const auto* name : {"GeomDiameter", "Mor01", "PNSA1"}) {
        EXPECT_TRUE(full.Has(schema->IndexOf(name)))
            << name << " must be present under All() on a 3D molecule; if missing, "
                       "build_mordred_3d_context failed and the 3D edge is untested";
    }

    // Each request targets a single 3D group's column without requesting any
    // ThreeDContext column (it has none), so the value can only be correct if
    // the ThreeDContext dependency ran in the pruned closure.
    expect_subset_matches_all_for_mol(mol, {"GeomDiameter"});  // LowCount3D
    expect_subset_matches_all_for_mol(mol, {"Mor01"});         // MoRSE
    expect_subset_matches_all_for_mol(mol, {"PNSA1"});         // CPSASurface

    // A combined request across all three 3D groups plus a 2D anchor column.
    expect_subset_matches_all_for_mol(mol, {"GeomDiameter", "Mor01", "PNSA1", "MW"});
}
