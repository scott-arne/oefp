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
    EXPECT_EQ(row.Schema().Size(), 214u);
}

TEST(RDKitDescriptorSourceTest, ComputeMolMatchesComputeWithContextAll) {
    RDKitDescriptorSource source;
    const auto mol = mol_from("c1ccccc1O");
    ComputeContext ctx(mol);
    const auto a = source.Compute(mol);
    const auto b = source.Compute(mol, ctx, ColumnRequest::All());
    EXPECT_EQ(a, b);
}
