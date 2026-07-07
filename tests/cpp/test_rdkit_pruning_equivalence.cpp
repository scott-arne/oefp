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
    // pruning): scan every non-requested index across all 214 and assert none is
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
