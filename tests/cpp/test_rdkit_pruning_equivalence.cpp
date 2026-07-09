#include "oefp/column_request.h"
#include "oefp/compute_context.h"
#include "oefp/descriptor_schema.h"
#include "oefp/rdkit_descriptors.h"

#include <gtest/gtest.h>
#include <oechem.h>

#include <cstddef>
#include <string>
#include <unordered_set>
#include <vector>

using namespace OEFP;

namespace {

std::vector<std::size_t> indices_for(const std::vector<std::string>& names) {
    const auto schema = RDKitDescriptorSchema();
    std::vector<std::size_t> out;
    out.reserve(names.size());
    for (const auto& n : names) {
        out.push_back(schema->IndexOf(n));
    }
    return out;
}

// Assert that computing only the requested columns yields byte-identical values
// for every requested column compared with computing the full schema, and that
// pruning is subtractive: every non-requested column stays missing.
void expect_subset_matches_all_for_mol(
    const OEChem::OEMolBase& mol, const std::vector<std::string>& names) {
    ComputeContext ctx_all(mol);
    const auto full = MakeRDKitDescriptors(mol, ctx_all, ColumnRequest::All());

    ComputeContext ctx_sub(mol);
    const auto sub = MakeRDKitDescriptors(mol, ctx_sub, ColumnRequest::Subset(indices_for(names)));

    const auto schema = RDKitDescriptorSchema();
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
    // pruning): scan every non-requested index across all 213 and assert none is
    // set, so a stray emission anywhere in the schema fails deterministically.
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

// A representative panel spanning small, ring, hetero, and branched molecules.
constexpr const char* kPanel[] = {
    "CC(=O)OC1=CC=CC=C1C(=O)O",  // aspirin
    "c1ccccc1",                  // benzene
    "C1CCCCC1",                  // cyclohexane
    "CCN",                       // ethylamine
    "OC(=O)Cc1ccc(O)cc1",        // 4-hydroxyphenylacetic acid
    "C",                         // methane (degenerate rings/paths)
};

// Assert Subset==All for one requested column across the whole panel.
void expect_column_matches_across_panel(const std::vector<std::string>& names) {
    for (const auto* smiles : kPanel) {
        expect_subset_matches_all(smiles, names);
    }
}

}  // namespace

TEST(RDKitPruningEquivalenceTest, SingleColumn) {
    expect_subset_matches_all("CC(=O)OC1=CC=CC=C1C(=O)O", {"ExactMolWt"});
}

TEST(RDKitPruningEquivalenceTest, WholeCountsWeightsGroup) {
    expect_subset_matches_all("c1ccccc1O",
        {"HeavyAtomCount", "ExactMolWt", "MolWt", "FractionCSP3"});
}

// One representative column from the only emitting group, verified across the
// panel: a wrong emitted_columns list surfaces as a Subset!=All divergence on at
// least one panel molecule.
TEST(RDKitPruningEquivalenceTest, PerGroupRepresentativeColumn) {
    expect_column_matches_across_panel({"HeavyAtomCount"});  // CountsWeights
}

// A wider mix of CountsWeights columns spanning integer counts and floats.
TEST(RDKitPruningEquivalenceTest, WideCountsWeightsMix) {
    expect_column_matches_across_panel(
        {"MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons", "NumHeteroatoms",
         "FractionCSP3", "FpDensityMorgan1", "FpDensityMorgan2", "FpDensityMorgan3",
         "NumHDonors", "NumHAcceptors", "NumRotatableBonds"});
}

// RingCounts representative single column: requesting only RingCount must match
// the full-schema value and leave the other 10 group members missing on every
// panel molecule, proving the group's emitted_columns list is correct.
TEST(RDKitPruningEquivalenceTest, RingCountsSingleColumn) {
    expect_column_matches_across_panel({"RingCount"});
}

// The whole RingCounts group requested together, across the panel: a divergence
// on any fused/ring molecule surfaces a pruning error in the group.
TEST(RDKitPruningEquivalenceTest, WholeRingCountsGroup) {
    expect_column_matches_across_panel(
        {"RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
         "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAliphaticCarbocycles",
         "NumAliphaticHeterocycles", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
         "NumHeterocycles"});
}

// Connectivity representative single column across the panel: requesting only
// Chi0 must match the full-schema value and leave the other 19 Connectivity
// members (and every other column) missing, proving the group's emitted_columns
// list is correct.
TEST(RDKitPruningEquivalenceTest, ConnectivitySingleColumn) {
    expect_column_matches_across_panel({"Chi0"});
}

// The whole Connectivity group requested together across the panel: a divergence
// on any ring/hetero/branched molecule surfaces a pruning error in the group
// (e.g. a Chi path or Kappa shape index computed differently under pruning).
TEST(RDKitPruningEquivalenceTest, WholeConnectivityGroup) {
    expect_column_matches_across_panel(
        {"Chi0", "Chi1", "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
         "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v", "HallKierAlpha",
         "Kappa1", "Kappa2", "Kappa3", "BertzCT", "BalabanJ", "Ipc", "AvgIpc"});
}

// `Phi` (= Kappa1*Kappa2/heavy-count) is emitted by the Connectivity group,
// co-located with the Kappa indices it derives from (it was moved out of
// CountsWeights so that cheap CountsWeights requests no longer force this
// group's O(n^4) computation). Requesting ONLY {"Phi"} must (a) reproduce the
// full-schema Phi value exactly, and (b) emit NO other Connectivity column
// (Chi*, Kappa*, BertzCT, ...) — the group runs and emits only the requested
// Phi, its 20 siblings staying missing. expect_subset_matches_all already
// asserts every non-requested column (all 212 others) is missing in the row.
TEST(RDKitPruningEquivalenceTest, PhiSingleColumnAcrossPanel) {
    expect_column_matches_across_panel({"Phi"});
}

// The perf motivation for emitting Phi from the Connectivity group: a request
// for a cheap CountsWeights column must NOT drag in the Connectivity group's
// heavy computation (its O(n^4) Ipc and O(n^3) BertzCT). CountsWeights reads
// only ctx.RingPerceivedMol(); Connectivity additionally builds
// ctx.HeavyAtomGraph(). So adding a Connectivity column (Chi0) to a MolWt
// request must STRICTLY increase the shared-context intermediates computed —
// i.e. MolWt alone does not already run Connectivity. If a
// CountsWeights->Connectivity dependency is ever reintroduced (as when Phi
// lived in CountsWeights), MolWt alone would already run Connectivity and the
// two counts would converge, failing this test.
TEST(RDKitPruningEquivalenceTest, CheapCountsWeightsRequestSkipsConnectivity) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));  // aspirin

    ComputeContext ctx_molwt(mol);
    MakeRDKitDescriptors(mol, ctx_molwt, ColumnRequest::Subset(indices_for({"MolWt"})));

    ComputeContext ctx_molwt_chi(mol);
    MakeRDKitDescriptors(
        mol, ctx_molwt_chi, ColumnRequest::Subset(indices_for({"MolWt", "Chi0"})));

    EXPECT_LT(ctx_molwt.ComputeCount(), ctx_molwt_chi.ComputeCount());
}

// One representative column per Task-7 family across the panel: requesting only
// that column must reproduce the full-schema value and leave every other column
// (including its own group siblings) missing, proving each new group's
// emitted_columns list is correct under subtractive pruning.
TEST(RDKitPruningEquivalenceTest, CrippenSingleColumn) {
    expect_column_matches_across_panel({"MolLogP"});
}

TEST(RDKitPruningEquivalenceTest, SurfacePolaritySingleColumn) {
    expect_column_matches_across_panel({"LabuteASA"});
}

TEST(RDKitPruningEquivalenceTest, EStateSingleColumn) {
    expect_column_matches_across_panel({"MaxEStateIndex"});
}

// VSA per-sub-family single-column scenarios across the panel: requesting only
// one bin of each sub-family must reproduce the full-schema value and leave every
// other column (including its own sub-family siblings) missing, proving the VSA
// group's emitted_columns list is correct under subtractive pruning.
TEST(RDKitPruningEquivalenceTest, SlogpVsaSingleColumn) {
    expect_column_matches_across_panel({"SlogP_VSA2"});
}

TEST(RDKitPruningEquivalenceTest, SmrVsaSingleColumn) {
    expect_column_matches_across_panel({"SMR_VSA1"});
}

TEST(RDKitPruningEquivalenceTest, PeoeVsaSingleColumnAcrossPanel) {
    expect_column_matches_across_panel({"PEOE_VSA7"});
}

TEST(RDKitPruningEquivalenceTest, EstateVsaSingleColumn) {
    expect_column_matches_across_panel({"EState_VSA1"});
}

TEST(RDKitPruningEquivalenceTest, VsaEstateSingleColumn) {
    expect_column_matches_across_panel({"VSA_EState1"});
}

// The mandatory dependency-closure pruning scenario (Spec §7 item 3): the VSA
// group declares a dependency on the EState group. Requesting ONLY
// {"EState_VSA1"} must (a) reproduce the full-schema EState_VSA1 value exactly,
// and (b) emit NO *EStateIndex scalar column (MaxEStateIndex, MinEStateIndex,
// MaxAbsEStateIndex, MinAbsEStateIndex all stay missing) — the EState group runs
// only as a dependency, warming the shared per-atom EState vector on the context,
// without any of its own columns being requested. This differs from the Phi case:
// the shared datum lives on ComputeContext, not a group artifact, so the edge is
// about run ordering/warming rather than an artifact hand-off. Because
// EState_VSA1 alone must equal its All() value, the scenario genuinely exercises
// the dependency (a missing/empty EState vector would zero the bin and diverge).
// expect_subset_matches_all already asserts every non-requested column (all 212
// others, including the four *EStateIndex columns) is missing in the pruned row.
// Fragments representative single column across the panel: requesting only one
// fr_* SMARTS count must reproduce the full-schema value and leave every other
// column (including the other 79 fragment members) missing, proving the group's
// emitted_columns list is correct under subtractive pruning.
TEST(RDKitPruningEquivalenceTest, FragmentsSingleColumn) {
    expect_column_matches_across_panel({"fr_benzene"});
}

// Single R<n> ring-membership fragment across the panel: fr_bicyclic is computed
// by relaxed match + SSSR-membership post-filter, and requesting it alone must
// reproduce the full-schema value and leave every other column missing — this
// also proves the lazy SSSR-membership map fires when only a ring fragment is
// wanted.
TEST(RDKitPruningEquivalenceTest, RingConstrainedFragmentSingleColumn) {
    expect_column_matches_across_panel({"fr_bicyclic"});
}

// The whole Fragments group requested together across the panel: a divergence on
// any molecule surfaces a pruning error in the group. Lists all 85 emitted fr_*,
// including the five R<n> ring-membership fragments (fr_bicyclic, fr_lactone,
// fr_benzodiazepine, fr_HOCCN, fr_Ndealkylation2) computed via relaxed match +
// SSSR post-filter.
TEST(RDKitPruningEquivalenceTest, WholeFragmentsGroup) {
    expect_column_matches_across_panel(
        {"fr_C_O", "fr_C_O_noCOO", "fr_Al_OH", "fr_Ar_OH", "fr_methoxy",
         "fr_oxime", "fr_ester", "fr_Al_COO", "fr_Ar_COO", "fr_COO", "fr_COO2",
         "fr_ketone", "fr_ether", "fr_phenol", "fr_aldehyde", "fr_quatN",
         "fr_NH2", "fr_NH1", "fr_NH0", "fr_Ar_N", "fr_Ar_NH", "fr_aniline",
         "fr_Imine", "fr_nitrile", "fr_hdrzine", "fr_hdrzone", "fr_nitroso",
         "fr_N_O", "fr_nitro", "fr_azo", "fr_diazo", "fr_azide", "fr_amide",
         "fr_priamide", "fr_amidine", "fr_guanido", "fr_Nhpyrrole", "fr_imide",
         "fr_isocyan", "fr_isothiocyan", "fr_thiocyan", "fr_halogen",
         "fr_alkyl_halide", "fr_sulfide", "fr_SH", "fr_C_S", "fr_sulfone",
         "fr_sulfonamd", "fr_prisulfonamd", "fr_barbitur", "fr_urea",
         "fr_term_acetylene", "fr_imidazole", "fr_furan", "fr_thiophene",
         "fr_thiazole", "fr_oxazole", "fr_pyridine", "fr_piperdine",
         "fr_piperzine", "fr_morpholine", "fr_lactam", "fr_tetrazole",
         "fr_epoxide", "fr_unbrch_alkane", "fr_benzene", "fr_phos_acid",
         "fr_phos_ester", "fr_nitro_arom", "fr_nitro_arom_nonortho",
         "fr_dihydropyridine", "fr_phenol_noOrthoHbond", "fr_Al_OH_noTert",
         "fr_para_hydroxylation", "fr_allylic_oxid", "fr_aryl_methyl",
         "fr_Ndealkylation1", "fr_alkyl_carbamate", "fr_ketone_Topliss",
         "fr_ArN", "fr_bicyclic", "fr_lactone", "fr_benzodiazepine", "fr_HOCCN",
         "fr_Ndealkylation2"});
}

TEST(RDKitPruningEquivalenceTest, EStateVsaDependencyClosureAcrossPanel) {
    expect_column_matches_across_panel({"EState_VSA1"});

    // Belt-and-suspenders: assert directly that requesting only EState_VSA1 leaves
    // every *EStateIndex column missing while still producing EState_VSA1, so a
    // future regression that leaked the dependency's own columns fails loudly.
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));
    const auto schema = RDKitDescriptorSchema();
    ComputeContext ctx(mol);
    const auto row = MakeRDKitDescriptors(
        mol, ctx, ColumnRequest::Subset(indices_for({"EState_VSA1"})));
    EXPECT_TRUE(row.Has(schema->IndexOf("EState_VSA1")));
    for (const auto* name : {"MaxEStateIndex", "MinEStateIndex",
                             "MaxAbsEStateIndex", "MinAbsEStateIndex"}) {
        EXPECT_FALSE(row.Has(schema->IndexOf(name)))
            << name << " must stay missing when only EState_VSA1 is requested";
    }
}

// BCUT2D representative single column across the panel: requesting only
// BCUT2D_MWHI must reproduce the full-schema value and leave every other column
// (including the other seven emitted BCUT2D members) missing, proving the group's
// emitted_columns list is correct under subtractive pruning. Every panel molecule
// is Gasteiger-parametrized, so the eigenvalues are finite here.
TEST(RDKitPruningEquivalenceTest, BCUT2DSingleColumn) {
    expect_column_matches_across_panel({"BCUT2D_MWHI"});
}

// BCUT2D_CHGHI single column across the panel: the charge-weighted eigenvalues
// (re-enabled after the RDKit sp-cumulene Gasteiger port) must prune correctly
// like the other BCUT2D pairs.
TEST(RDKitPruningEquivalenceTest, Bcut2dChgSingleColumnAcrossPanel) {
    expect_column_matches_across_panel({"BCUT2D_CHGHI"});
}

// The whole emitted BCUT2D group requested together across the panel: a
// divergence on any molecule surfaces a pruning error in the group.
TEST(RDKitPruningEquivalenceTest, WholeBCUT2DGroup) {
    expect_column_matches_across_panel(
        {"BCUT2D_MWHI", "BCUT2D_MWLOW", "BCUT2D_CHGHI", "BCUT2D_CHGLO",
         "BCUT2D_LOGPHI", "BCUT2D_LOGPLOW", "BCUT2D_MRHI", "BCUT2D_MRLOW"});
}

// PartialCharge representative single column across the panel: requesting only
// MaxPartialCharge must reproduce the full-schema value and leave every other
// column missing, proving the group's emitted_columns list is correct under
// subtractive pruning.
TEST(RDKitPruningEquivalenceTest, PartialChargeSingleColumnAcrossPanel) {
    expect_column_matches_across_panel({"MaxPartialCharge"});
}

// Composite representative single column across the panel: requesting only qed
// must reproduce the full-schema value and leave every other column missing,
// proving the group's emitted_columns list is correct under subtractive pruning.
TEST(RDKitPruningEquivalenceTest, CompositeSingleColumn) {
    expect_column_matches_across_panel({"qed"});
}
