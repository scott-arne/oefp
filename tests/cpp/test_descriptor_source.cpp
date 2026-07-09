#include "oefp/descriptor_source.h"
#include "oefp/column_request.h"
#include "oefp/compute_context.h"
#include "oefp/molecular_properties.h"
#include "oefp/mordred.h"

#include <gtest/gtest.h>

#include <cstdint>

#include <oechem.h>
#include <oemolprop.h>

using namespace OEFP;

TEST(MordredDescriptorSourceTest, SchemaMatchesMordredSchema) {
    MordredDescriptorSource source;
    EXPECT_EQ(source.Schema()->SchemaId(), MordredDescriptorSchema()->SchemaId());
}

TEST(MordredDescriptorSourceTest, ComputeMatchesFreeFunction) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CCO"));
    MordredDescriptorSource source;
    const auto row = source.Compute(mol);
    EXPECT_DOUBLE_EQ(row.Float("MW"), MakeMordredDescriptors(mol).Float("MW"));
}

TEST(OpenEyePropertyDescriptorSourceTest, TaggedColumnsMatchMordredByConstruction) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));  // aspirin
    OpenEyePropertyDescriptorSource oe;
    MordredDescriptorSource mordred;
    const auto oe_row = oe.Compute(mol);
    const auto md_row = mordred.Compute(mol);
    EXPECT_DOUBLE_EQ(oe_row.Float("MolecularWeight"), md_row.Float("MW"));
    EXPECT_DOUBLE_EQ(oe_row.Float("TopologicalPSA"), md_row.Float("TopoPSA"));
    EXPECT_EQ(oe_row.Int("HeavyAtomCount"), md_row.Int("nHeavyAtom"));
    EXPECT_EQ(oe_row.Int("LipinskiHBD"), md_row.Int("nHBDon"));
    EXPECT_EQ(oe_row.Int("HBA"), md_row.Int("nHBAcc"));
}

TEST(OpenEyePropertyDescriptorSourceTest, SchemaTagsMatchMordred) {
    OpenEyePropertyDescriptorSource oe;
    const auto& schema = *oe.Schema();
    EXPECT_EQ(schema.Definition(schema.IndexOf("MolecularWeight")).canonical_id,
              "quantity:exact_molecular_weight");
    // OpenEye-unique columns are intentionally untagged (never deduplicated).
    EXPECT_TRUE(schema.Definition(schema.IndexOf("XLogP")).canonical_id.empty());
    EXPECT_TRUE(schema.Definition(schema.IndexOf("RotatableBondCount")).canonical_id.empty());
}

TEST(OpenEyePropertyDescriptorSourceTest, UntaggedColumnsMatchOpenEyeToolkit) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));  // aspirin
    OpenEyePropertyDescriptorSource oe;
    const auto row = oe.Compute(mol);
    float xlogp = 0.0f;
    ASSERT_TRUE(OEMolProp::OEGetXLogP(mol, xlogp));
    EXPECT_DOUBLE_EQ(row.Float("XLogP"), static_cast<double>(xlogp));
    EXPECT_EQ(row.Int("RotatableBondCount"),
              static_cast<std::int64_t>(OEMolProp::OEGetRotatableBondCount(mol)));
    EXPECT_EQ(row.Int("AromaticRingCount"),
              static_cast<std::int64_t>(OEMolProp::OEGetAromaticRingCount(mol)));
}

TEST(DescriptorSourceTest, OpenEyePrunesToRequestedColumns) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));
    OpenEyePropertyDescriptorSource oe;
    const auto schema = oe.Schema();
    ComputeContext ctx(mol);
    const auto full = oe.Compute(mol, ctx, ColumnRequest::All());
    ComputeContext ctx2(mol);
    const auto sub = oe.Compute(mol, ctx2, ColumnRequest::Subset({schema->IndexOf("XLogP")}));
    EXPECT_TRUE(sub.Has(schema->IndexOf("XLogP")));
    EXPECT_FALSE(sub.Has(schema->IndexOf("MolecularWeight")));
    EXPECT_FALSE(sub.Has(schema->IndexOf("TopologicalPSA")));
    EXPECT_DOUBLE_EQ(sub.Float("XLogP"), full.Float("XLogP"));
}

namespace {
// A minimal source that only overrides the pure Compute(mol) — proves the
// base context/request overload falls back correctly.
class LegacyOnlySource : public OEFP::DescriptorSource {
public:
    std::shared_ptr<const OEFP::DescriptorSchema> Schema() const override {
        OEFP::DescriptorSchemaBuilder b;
        b.Add(OEFP::DescriptorDefinition{"X", OEFP::DescriptorValueKind::Float});
        return b.Build();
    }
    OEFP::DescriptorSet Compute(const OEChem::OEMolBase&) const override {
        return OEFP::DescriptorSetBuilder(Schema()).Build();
    }
};
}  // namespace

TEST(DescriptorSourceTest, ContextRequestFallsBackToLegacyCompute) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CCO"));
    LegacyOnlySource source;
    // Call through a DescriptorSource& — this is how the calculator holds and
    // invokes sources (shared_ptr<const DescriptorSource>). A concrete subclass
    // that declares only Compute(mol) HIDES the base's other Compute overloads
    // by C++ name-hiding, so the context/request overload must be reached via
    // the base type, not the concrete type.
    const OEFP::DescriptorSource& base = source;
    OEFP::ComputeContext ctx(mol);
    const auto row = base.Compute(mol, ctx, OEFP::ColumnRequest::All());
    EXPECT_EQ(row.Schema().SchemaId(), source.Schema()->SchemaId());  // fell back to Compute(mol)
}

TEST(DescriptorSourceTest, OpenEyeComputesRingPerceptionThroughContextThenReuses) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));
    OEFP::OpenEyePropertyDescriptorSource openeye;
    const OEFP::DescriptorSource& oe = openeye;   // base-ref: avoid name-hiding

    OEFP::ComputeContext ctx(mol);
    ASSERT_EQ(ctx.ComputeCount(), 0u);            // nothing computed yet

    // POSITIVE proof: OpenEye must build its shared intermediate THROUGH ctx.
    // A non-sharing fallback that rebuilds ring perception on a private inline
    // OEGraphMol never touches ctx and would leave the count at 0 -> FAILS here.
    const auto oe_row = oe.Compute(mol, ctx, OEFP::ColumnRequest::All());
    EXPECT_EQ(ctx.ComputeCount(), 2u);            // ring perception + H-suppression computed via ctx (0 -> 2)

    // REUSE proof: a second OpenEye call on the same ctx adds nothing.
    const auto oe_row2 = oe.Compute(mol, ctx, OEFP::ColumnRequest::All());
    EXPECT_EQ(ctx.ComputeCount(), 2u);            // cache hit (2 -> 2)

    // Value-equivalence regression guard vs the unshared (fresh-context) path.
    const auto oe_plain = openeye.Compute(mol);   // convenience overload, fresh ctx
    EXPECT_DOUBLE_EQ(oe_row.Float("TopologicalPSA"), oe_plain.Float("TopologicalPSA"));
    EXPECT_EQ(oe_row.Int("HBA"), oe_plain.Int("HBA"));
    EXPECT_EQ(oe_row.Int("LipinskiHBD"), oe_plain.Int("LipinskiHBD"));
}

// Cross-source sharing of the ring-perceived working molecule. Task 10 routed
// Mordred's first-batch preparation through ctx.RingPerceivedMol() (its inline
// prep was byte-identical: copy + OEFindRingAtomsAndBonds + OEAssignHybridization),
// so Mordred now warms that shared intermediate. OpenEye's only shared
// intermediate is the ring-perceived molecule, so its later pull is a pure cache
// hit and adds nothing to the compute count.
TEST(DescriptorSourceTest, SecondSourceReusesSharedIntermediate) {
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CC(=O)OC1=CC=CC=C1C(=O)O"));
    OEFP::MordredDescriptorSource mordred;
    OEFP::OpenEyePropertyDescriptorSource openeye;
    const OEFP::DescriptorSource& md = mordred;
    const OEFP::DescriptorSource& oe = openeye;

    OEFP::ComputeContext ctx(mol);
    md.Compute(mol, ctx, OEFP::ColumnRequest::All());
    const auto after_mordred = ctx.ComputeCount();   // Mordred touched its shared intermediates
    oe.Compute(mol, ctx, OEFP::ColumnRequest::All());
    // Ring perception was already computed by Mordred; OpenEye reuses it and adds
    // nothing (its only shared intermediate is ring perception, which Mordred
    // touched). Combined with the 0->1 proof above, this shows genuine sharing.
    EXPECT_EQ(ctx.ComputeCount(), after_mordred);
}
