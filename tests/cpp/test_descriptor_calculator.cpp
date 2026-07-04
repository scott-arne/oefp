#include "oefp/descriptor_calculator.h"
#include "oefp/descriptor_batch.h"
#include "oefp/descriptor_source.h"

#include <gtest/gtest.h>
#include <oechem.h>

#include <memory>
#include <stdexcept>
#include <vector>

using namespace OEFP;

TEST(DescriptorCalculatorTest, FirstWinsDropsLaterCanonicalDuplicate) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<MordredDescriptorSource>());
    entries.emplace_back(std::make_shared<OpenEyePropertyDescriptorSource>());
    DescriptorCalculator calc(std::move(entries));
    const auto& schema = calc.Schema();
    // Mordred registered first keeps "MW"; OpenEye "MolecularWeight" (same canonical_id) is dropped.
    EXPECT_TRUE(schema.Contains("MW"));
    EXPECT_FALSE(schema.Contains("MolecularWeight"));
    // Every tagged OpenEye column is dropped (Mordred owns all shared canonical ids); e.g. HBA.
    EXPECT_FALSE(schema.Contains("HBA"));
    // Mordred-unique columns survive; OpenEye-unique untagged columns survive.
    EXPECT_TRUE(schema.Contains("ABC"));
    EXPECT_TRUE(schema.Contains("XLogP"));
}

TEST(DescriptorCalculatorTest, SelectionNarrowsSourceBeforeDedup) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<OpenEyePropertyDescriptorSource>(),
                         DescriptorSelection::Names({"HBA"}));
    DescriptorCalculator calc(std::move(entries));
    EXPECT_EQ(calc.Schema().Size(), 1u);
    EXPECT_TRUE(calc.Schema().Contains("HBA"));
}

namespace {
class FixedSource : public DescriptorSource {
public:
    explicit FixedSource(std::shared_ptr<const DescriptorSchema> schema) : schema_(std::move(schema)) {}
    std::shared_ptr<const DescriptorSchema> Schema() const override { return schema_; }
    DescriptorSet Compute(const OEChem::OEMolBase&) const override {
        return DescriptorSetBuilder(schema_).Build();
    }
private:
    std::shared_ptr<const DescriptorSchema> schema_;
};
// A source with a non-empty schema whose Compute always throws; used to prove
// that an empty-plan source is never computed.
class ThrowingSource : public DescriptorSource {
public:
    explicit ThrowingSource(std::shared_ptr<const DescriptorSchema> schema) : schema_(std::move(schema)) {}
    std::shared_ptr<const DescriptorSchema> Schema() const override { return schema_; }
    DescriptorSet Compute(const OEChem::OEMolBase&) const override {
        throw std::runtime_error("should not be computed");
    }
private:
    std::shared_ptr<const DescriptorSchema> schema_;
};
std::shared_ptr<const DescriptorSchema> single_column(const char* name) {
    DescriptorSchemaBuilder b;
    b.Add(DescriptorDefinition{name, DescriptorValueKind::Float});
    return b.Build();
}
} // namespace

TEST(DescriptorCalculatorTest, UnresolvedNameCollisionThrows) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<FixedSource>(single_column("X")));
    entries.emplace_back(std::make_shared<FixedSource>(single_column("X")));
    EXPECT_THROW(DescriptorCalculator(std::move(entries)), std::invalid_argument);
}

TEST(DescriptorCalculatorTest, NullSourceThrows) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::shared_ptr<const DescriptorSource>{});
    EXPECT_THROW(DescriptorCalculator(std::move(entries)), std::invalid_argument);
}

TEST(DescriptorCalculatorTest, NullSchemaThrows) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<FixedSource>(nullptr));
    EXPECT_THROW(DescriptorCalculator(std::move(entries)), std::invalid_argument);
}

TEST(DescriptorCalculatorTest, EmptyCalculatorHasEmptyValidSchema) {
    DescriptorCalculator calc(std::vector<DescriptorSourceEntry>{});
    EXPECT_EQ(calc.Schema().Size(), 0u);
}

TEST(DescriptorCalculatorTest, FullySelectedEmptyIsValid) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<OpenEyePropertyDescriptorSource>(),
                         DescriptorSelection::Names({}));  // selects nothing
    DescriptorCalculator calc(std::move(entries));
    EXPECT_EQ(calc.Schema().Size(), 0u);
}

TEST(DescriptorCalculatorTest, ComputeMergesKeptColumns) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<MordredDescriptorSource>());
    entries.emplace_back(std::make_shared<OpenEyePropertyDescriptorSource>());
    DescriptorCalculator calc(std::move(entries));
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CCO"));
    const auto row = calc.Compute(mol);
    EXPECT_EQ(row.Schema().SchemaId(), calc.Schema().SchemaId());
    EXPECT_TRUE(row.Has("MW"));       // Mordred-first keeps its tagged column
    EXPECT_TRUE(row.Has("XLogP"));    // OpenEye-unique untagged column survives dedup
}

TEST(DescriptorCalculatorTest, CalculateBatchEqualsPerRowComputeInOrder) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<MordredDescriptorSource>());
    DescriptorCalculator calc(std::move(entries));
    OEChem::OEGraphMol a, b;
    ASSERT_TRUE(OEChem::OESmilesToMol(a, "CCO"));
    ASSERT_TRUE(OEChem::OESmilesToMol(b, "c1ccccc1"));
    const OEChem::OEMolBase& base_a = a;
    const OEChem::OEMolBase& base_b = b;
    std::vector<const OEChem::OEMolBase*> mols{&base_a, &base_b};
    const auto batch = calc.CalculateBatch(mols);
    ASSERT_EQ(batch.Size(), 2u);
    EXPECT_DOUBLE_EQ(batch.FloatColumn("MW")[0], calc.Compute(a).Float("MW"));
    EXPECT_DOUBLE_EQ(batch.FloatColumn("MW")[1], calc.Compute(b).Float("MW"));
}

TEST(DescriptorCalculatorTest, CalculateBatchRejectsNullMoleculePointer) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<MordredDescriptorSource>());
    DescriptorCalculator calc(std::move(entries));
    std::vector<const OEChem::OEMolBase*> mols{nullptr};
    EXPECT_THROW(calc.CalculateBatch(mols), std::invalid_argument);
}

TEST(DescriptorCalculatorTest, EmptySelectedSourceIsNotComputed) {
    std::vector<DescriptorSourceEntry> entries;
    // Throwing source is selected away to an empty plan, so its Compute must be skipped.
    entries.emplace_back(std::make_shared<ThrowingSource>(single_column("Boom")),
                         DescriptorSelection::Names({}));
    // A real source gives the calculator a non-empty schema to merge into.
    entries.emplace_back(std::make_shared<MordredDescriptorSource>());
    DescriptorCalculator calc(std::move(entries));
    EXPECT_FALSE(calc.Schema().Contains("Boom"));
    OEChem::OEGraphMol mol;
    ASSERT_TRUE(OEChem::OESmilesToMol(mol, "CCO"));
    EXPECT_NO_THROW(calc.Compute(mol));
}

TEST(DescriptorCalculatorTest, CalculateBatchAllowsDuplicateMoleculePointer) {
    std::vector<DescriptorSourceEntry> entries;
    entries.emplace_back(std::make_shared<MordredDescriptorSource>());
    DescriptorCalculator calc(std::move(entries));
    OEChem::OEGraphMol a;
    ASSERT_TRUE(OEChem::OESmilesToMol(a, "CCO"));
    const OEChem::OEMolBase& base_a = a;
    std::vector<const OEChem::OEMolBase*> mols{&base_a, &base_a};
    const auto batch = calc.CalculateBatch(mols);
    ASSERT_EQ(batch.Size(), 2u);
    EXPECT_DOUBLE_EQ(batch.FloatColumn("MW")[0], calc.Compute(a).Float("MW"));
    EXPECT_DOUBLE_EQ(batch.FloatColumn("MW")[1], calc.Compute(a).Float("MW"));
}
