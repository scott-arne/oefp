#include "oefp/descriptor_source.h"
#include "oefp/rdkit_descriptors.h"

#include <gtest/gtest.h>
#include <oechem.h>

using namespace OEFP;

namespace {
OEChem::OEGraphMol mol_from(const char* smiles) {
    OEChem::OEGraphMol mol;
    OEChem::OESmilesToMol(mol, smiles);
    return mol;
}
}  // namespace

TEST(RDKitDescriptorSourceTest, SchemaMatchesFreeFunction) {
    RDKitDescriptorSource source;
    EXPECT_EQ(source.Schema(), RDKitDescriptorSchema());
}

TEST(RDKitDescriptorSourceTest, ComputeReturnsFullWidthRow) {
    RDKitDescriptorSource source;
    const auto mol = mol_from("CCO");
    const auto row = source.Compute(mol);
    EXPECT_EQ(row.Schema().Size(), 213u);
}

TEST(RDKitDescriptorSourceTest, ComputeMolMatchesComputeWithContextAll) {
    RDKitDescriptorSource source;
    const auto mol = mol_from("c1ccccc1O");
    ComputeContext ctx(mol);
    const auto a = source.Compute(mol);
    const auto b = source.Compute(mol, ctx, ColumnRequest::All());
    EXPECT_EQ(a, b);
}

// TPSA must count each nitrogen/oxygen hydrogen exactly once. OpenEye's
// GetTotalHCount() already includes bonded explicit hydrogens, so explicit-H
// input (e.g. after OEAddExplicitHydrogens) must yield the SAME TPSA as the
// implicit-H form; RDKit's Descriptors.TPSA strips explicit H and reports the
// implicit-H value. This regression guards the earlier double-count where
// explicit-H neighbors were re-added on top of GetTotalHCount(): water inflated
// to 34.5 (correct 31.5), ammonia to 39.5 (correct 35.0), CCO+H to 22.9
// (correct 20.23). It fails against that buggy code and passes once total
// hydrogens are counted once.
TEST(RDKitDescriptorSourceTest, TpsaCountsExplicitHydrogensOnce) {
    RDKitDescriptorSource source;
    struct Case {
        const char* smiles;
        double expected_tpsa;  // RDKit implicit-H Descriptors.TPSA value
    };
    for (const auto& c : {Case{"O", 31.5},                    // water
                          Case{"N", 35.0},                    // ammonia
                          Case{"CCO", 20.23},                 // ethanol
                          Case{"O=S(=O)(N)c1ccccc1", 60.16}}) {  // benzenesulfonamide
        OEChem::OEGraphMol mol;
        ASSERT_TRUE(OEChem::OESmilesToMol(mol, c.smiles)) << c.smiles;
        OEChem::OEAddExplicitHydrogens(mol);  // exercise the explicit-H path
        const auto row = source.Compute(mol);
        ASSERT_TRUE(row.Has("TPSA")) << c.smiles;
        EXPECT_NEAR(row.Float("TPSA"), c.expected_tpsa, 1e-6) << c.smiles;
    }
}
