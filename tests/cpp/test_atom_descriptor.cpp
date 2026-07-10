#include <gtest/gtest.h>

#include "oefp/atom_descriptor.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {
namespace {

std::shared_ptr<const DescriptorSchema> test_schema() {
    DescriptorSchemaBuilder builder;
    DescriptorDefinition col_a;
    col_a.name = "col_a";
    col_a.value_kind = DescriptorValueKind::Float;
    builder.Add(col_a);

    DescriptorDefinition col_b;
    col_b.name = "col_b";
    col_b.value_kind = DescriptorValueKind::Float;
    builder.Add(col_b);

    return builder.Build();
}

} // namespace

TEST(AtomDescriptorSetTest, StoresAtomIndicesAndValuesWithMissing) {
    auto schema = test_schema();
    std::vector<std::uint32_t> atom_indices = {0u, 1u, 2u};
    std::vector<std::vector<std::optional<double>>> columns = {
        {1.0, 2.0, std::nullopt},  // col_a
        {3.5, std::nullopt, 4.5}   // col_b
    };

    AtomDescriptorSet set(schema, atom_indices, columns);

    EXPECT_EQ(set.AtomCount(), 3u);
    EXPECT_EQ(&set.Schema(), schema.get());
    EXPECT_EQ(set.AtomIndices(), atom_indices);

    // col_a present values
    EXPECT_EQ(set.Value(0, "col_a"), std::optional<double>(1.0));
    EXPECT_EQ(set.Value(1, "col_a"), std::optional<double>(2.0));
    EXPECT_EQ(set.Value(2, "col_a"), std::nullopt);

    // col_b present and missing
    EXPECT_EQ(set.Value(0, "col_b"), std::optional<double>(3.5));
    EXPECT_EQ(set.Value(1, "col_b"), std::nullopt);
    EXPECT_EQ(set.Value(2, "col_b"), std::optional<double>(4.5));

    // By column index
    EXPECT_EQ(set.Value(0, 0), std::optional<double>(1.0));
    EXPECT_EQ(set.Value(1, 1), std::nullopt);
}

TEST(AtomDescriptorSetTest, EmptySetHasZeroAtoms) {
    auto schema = test_schema();
    auto empty = AtomDescriptorSet::Empty(schema);

    EXPECT_EQ(empty.AtomCount(), 0u);
    EXPECT_EQ(&empty.Schema(), schema.get());
    EXPECT_TRUE(empty.AtomIndices().empty());
}

TEST(AtomDescriptorSetTest, RejectsColumnCountMismatch) {
    auto schema = test_schema();  // expects 2 columns
    std::vector<std::uint32_t> atom_indices = {0u};
    std::vector<std::vector<std::optional<double>>> wrong_columns = {
        {1.0}  // only 1 column
    };

    EXPECT_THROW(
        AtomDescriptorSet(schema, atom_indices, wrong_columns),
        std::invalid_argument
    );
}

TEST(AtomDescriptorSetTest, RejectsShorterColumn) {
    auto schema = test_schema();
    std::vector<std::uint32_t> atom_indices = {0u, 1u, 2u};
    std::vector<std::vector<std::optional<double>>> ragged_columns = {
        {1.0, 2.0},       // too short: only 2 values for 3 atoms
        {3.0, 4.0, 5.0}   // correct length
    };

    EXPECT_THROW(
        AtomDescriptorSet(schema, atom_indices, ragged_columns),
        std::invalid_argument
    );
}

TEST(AtomDescriptorSetTest, RejectsLongerColumn) {
    auto schema = test_schema();
    std::vector<std::uint32_t> atom_indices = {0u, 1u};
    std::vector<std::vector<std::optional<double>>> ragged_columns = {
        {1.0, 2.0},             // correct length
        {3.0, 4.0, 5.0, 6.0}    // too long: 4 values for 2 atoms
    };

    EXPECT_THROW(
        AtomDescriptorSet(schema, atom_indices, ragged_columns),
        std::invalid_argument
    );
}

TEST(BondDescriptorSetTest, StoresBondEndpointsAndValues) {
    auto schema = test_schema();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bond_endpoints = {
        {0u, 1u}, {1u, 2u}
    };
    std::vector<std::vector<std::optional<double>>> columns = {
        {0.5, 1.5},   // col_a
        {2.0, 3.0}    // col_b
    };

    BondDescriptorSet set(schema, bond_endpoints, columns);

    EXPECT_EQ(set.BondCount(), 2u);
    EXPECT_EQ(&set.Schema(), schema.get());
    EXPECT_EQ(set.BondEndpoints(), bond_endpoints);

    EXPECT_EQ(set.Value(0, "col_a"), std::optional<double>(0.5));
    EXPECT_EQ(set.Value(1, "col_b"), std::optional<double>(3.0));
    EXPECT_EQ(set.Value(0, 1), std::optional<double>(2.0));
}

TEST(BondDescriptorSetTest, EmptySetHasZeroBonds) {
    auto schema = test_schema();
    auto empty = BondDescriptorSet::Empty(schema);

    EXPECT_EQ(empty.BondCount(), 0u);
    EXPECT_EQ(&empty.Schema(), schema.get());
    EXPECT_TRUE(empty.BondEndpoints().empty());
}

TEST(BondDescriptorSetTest, RejectsShorterColumn) {
    auto schema = test_schema();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bond_endpoints = {
        {0u, 1u}, {1u, 2u}, {2u, 3u}
    };
    std::vector<std::vector<std::optional<double>>> ragged_columns = {
        {1.0, 2.0},       // too short: only 2 values for 3 bonds
        {3.0, 4.0, 5.0}   // correct length
    };

    EXPECT_THROW(
        BondDescriptorSet(schema, bond_endpoints, ragged_columns),
        std::invalid_argument
    );
}

TEST(BondDescriptorSetTest, RejectsLongerColumn) {
    auto schema = test_schema();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bond_endpoints = {
        {0u, 1u}, {1u, 2u}
    };
    std::vector<std::vector<std::optional<double>>> ragged_columns = {
        {1.0, 2.0},             // correct length
        {3.0, 4.0, 5.0, 6.0}    // too long: 4 values for 2 bonds
    };

    EXPECT_THROW(
        BondDescriptorSet(schema, bond_endpoints, ragged_columns),
        std::invalid_argument
    );
}

TEST(AtomDescriptorBatchTest, BatchesMultipleSetsIncludingEmpty) {
    auto schema = test_schema();

    // First set: 2 atoms
    std::vector<std::uint32_t> atoms1 = {0u, 1u};
    std::vector<std::vector<std::optional<double>>> cols1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    AtomDescriptorSet set1(schema, atoms1, cols1);

    // Second set: Empty (skipped molecule)
    AtomDescriptorSet set2 = AtomDescriptorSet::Empty(schema);

    // Third set: 1 atom
    std::vector<std::uint32_t> atoms3 = {0u};
    std::vector<std::vector<std::optional<double>>> cols3 = {
        {5.0},
        {6.0}
    };
    AtomDescriptorSet set3(schema, atoms3, cols3);

    AtomDescriptorBatch batch = AtomDescriptorBatch::Empty(schema);
    batch.Append(set1);
    batch.Append(set2);
    batch.Append(set3);

    // Three molecule segments
    EXPECT_EQ(batch.Size(), 3u);
    EXPECT_EQ(batch.AtomCount(), 3u);  // 2 + 0 + 1

    // Segment atom counts
    EXPECT_EQ(batch.SegmentAtomCount(0), 2u);
    EXPECT_EQ(batch.SegmentAtomCount(1), 0u);  // empty segment preserved
    EXPECT_EQ(batch.SegmentAtomCount(2), 1u);

    // CSR offsets
    EXPECT_NE(batch.RowOffsetDataAddress(), 0u);
}

TEST(BondDescriptorBatchTest, BatchesMultipleSetsIncludingEmpty) {
    auto schema = test_schema();

    // First set: 2 bonds
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bonds1 = {{0u, 1u}, {1u, 2u}};
    std::vector<std::vector<std::optional<double>>> cols1 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    BondDescriptorSet set1(schema, bonds1, cols1);

    // Second set: Empty
    BondDescriptorSet set2 = BondDescriptorSet::Empty(schema);

    // Third set: 1 bond
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bonds3 = {{0u, 1u}};
    std::vector<std::vector<std::optional<double>>> cols3 = {
        {5.0},
        {6.0}
    };
    BondDescriptorSet set3(schema, bonds3, cols3);

    BondDescriptorBatch batch = BondDescriptorBatch::Empty(schema);
    batch.Append(set1);
    batch.Append(set2);
    batch.Append(set3);

    EXPECT_EQ(batch.Size(), 3u);
    EXPECT_EQ(batch.BondCount(), 3u);

    EXPECT_EQ(batch.SegmentBondCount(0), 2u);
    EXPECT_EQ(batch.SegmentBondCount(1), 0u);
    EXPECT_EQ(batch.SegmentBondCount(2), 1u);
}

TEST(AtomDescriptorBatchTest, RejectsDifferentColumnCount) {
    auto schema = test_schema();  // 2 columns

    // Build a different schema with 3 columns
    DescriptorSchemaBuilder builder;
    DescriptorDefinition col_a;
    col_a.name = "col_a";
    builder.Add(col_a);
    DescriptorDefinition col_b;
    col_b.name = "col_b";
    builder.Add(col_b);
    DescriptorDefinition col_c;
    col_c.name = "col_c";
    builder.Add(col_c);
    auto different_schema = builder.Build();

    std::vector<std::uint32_t> atoms = {0u};
    std::vector<std::vector<std::optional<double>>> cols = {
        {1.0}, {2.0}, {3.0}
    };
    AtomDescriptorSet set(different_schema, atoms, cols);

    AtomDescriptorBatch batch = AtomDescriptorBatch::Empty(schema);
    EXPECT_THROW(batch.Append(set), std::invalid_argument);
}

TEST(AtomDescriptorBatchTest, RejectsReorderedSchema) {
    // Build schema with col_a, col_b
    auto schema = test_schema();

    // Build reversed schema with col_b, col_a
    DescriptorSchemaBuilder builder;
    DescriptorDefinition col_b;
    col_b.name = "col_b";
    builder.Add(col_b);
    DescriptorDefinition col_a;
    col_a.name = "col_a";
    builder.Add(col_a);
    auto reordered_schema = builder.Build();

    std::vector<std::uint32_t> atoms = {0u};
    std::vector<std::vector<std::optional<double>>> cols = {
        {1.0}, {2.0}
    };
    AtomDescriptorSet set(reordered_schema, atoms, cols);

    AtomDescriptorBatch batch = AtomDescriptorBatch::Empty(schema);
    EXPECT_THROW(batch.Append(set), std::invalid_argument);
}

TEST(AtomDescriptorBatchTest, RejectedAppendLeavesStateUnchanged) {
    auto schema = test_schema();

    // Create a valid set and append it
    std::vector<std::uint32_t> atoms1 = {0u, 1u};
    std::vector<std::vector<std::optional<double>>> cols1 = {
        {1.0, 2.0}, {3.0, 4.0}
    };
    AtomDescriptorSet set1(schema, atoms1, cols1);

    AtomDescriptorBatch batch = AtomDescriptorBatch::Empty(schema);
    batch.Append(set1);

    const auto initial_size = batch.Size();
    const auto initial_atom_count = batch.AtomCount();

    // Build a mismatched schema set
    DescriptorSchemaBuilder builder;
    DescriptorDefinition col_x;
    col_x.name = "col_x";
    builder.Add(col_x);
    DescriptorDefinition col_y;
    col_y.name = "col_y";
    builder.Add(col_y);
    auto bad_schema = builder.Build();

    std::vector<std::uint32_t> atoms2 = {0u};
    std::vector<std::vector<std::optional<double>>> cols2 = {
        {5.0}, {6.0}
    };
    AtomDescriptorSet bad_set(bad_schema, atoms2, cols2);

    // Append should fail
    EXPECT_THROW(batch.Append(bad_set), std::invalid_argument);

    // Batch state should be unchanged
    EXPECT_EQ(batch.Size(), initial_size);
    EXPECT_EQ(batch.AtomCount(), initial_atom_count);
    EXPECT_EQ(batch.SegmentAtomCount(0), 2u);
}

TEST(BondDescriptorBatchTest, RejectsDifferentColumnCount) {
    auto schema = test_schema();  // 2 columns

    DescriptorSchemaBuilder builder;
    DescriptorDefinition col_a;
    col_a.name = "col_a";
    builder.Add(col_a);
    DescriptorDefinition col_b;
    col_b.name = "col_b";
    builder.Add(col_b);
    DescriptorDefinition col_c;
    col_c.name = "col_c";
    builder.Add(col_c);
    auto different_schema = builder.Build();

    std::vector<std::pair<std::uint32_t, std::uint32_t>> bonds = {{0u, 1u}};
    std::vector<std::vector<std::optional<double>>> cols = {
        {1.0}, {2.0}, {3.0}
    };
    BondDescriptorSet set(different_schema, bonds, cols);

    BondDescriptorBatch batch = BondDescriptorBatch::Empty(schema);
    EXPECT_THROW(batch.Append(set), std::invalid_argument);
}

TEST(BondDescriptorBatchTest, RejectsReorderedSchema) {
    auto schema = test_schema();

    DescriptorSchemaBuilder builder;
    DescriptorDefinition col_b;
    col_b.name = "col_b";
    builder.Add(col_b);
    DescriptorDefinition col_a;
    col_a.name = "col_a";
    builder.Add(col_a);
    auto reordered_schema = builder.Build();

    std::vector<std::pair<std::uint32_t, std::uint32_t>> bonds = {{0u, 1u}};
    std::vector<std::vector<std::optional<double>>> cols = {
        {1.0}, {2.0}
    };
    BondDescriptorSet set(reordered_schema, bonds, cols);

    BondDescriptorBatch batch = BondDescriptorBatch::Empty(schema);
    EXPECT_THROW(batch.Append(set), std::invalid_argument);
}

TEST(BondDescriptorBatchTest, RejectedAppendLeavesStateUnchanged) {
    auto schema = test_schema();

    std::vector<std::pair<std::uint32_t, std::uint32_t>> bonds1 = {{0u, 1u}, {1u, 2u}};
    std::vector<std::vector<std::optional<double>>> cols1 = {
        {1.0, 2.0}, {3.0, 4.0}
    };
    BondDescriptorSet set1(schema, bonds1, cols1);

    BondDescriptorBatch batch = BondDescriptorBatch::Empty(schema);
    batch.Append(set1);

    const auto initial_size = batch.Size();
    const auto initial_bond_count = batch.BondCount();

    DescriptorSchemaBuilder builder;
    DescriptorDefinition col_x;
    col_x.name = "col_x";
    builder.Add(col_x);
    DescriptorDefinition col_y;
    col_y.name = "col_y";
    builder.Add(col_y);
    auto bad_schema = builder.Build();

    std::vector<std::pair<std::uint32_t, std::uint32_t>> bonds2 = {{0u, 1u}};
    std::vector<std::vector<std::optional<double>>> cols2 = {
        {5.0}, {6.0}
    };
    BondDescriptorSet bad_set(bad_schema, bonds2, cols2);

    EXPECT_THROW(batch.Append(bad_set), std::invalid_argument);

    EXPECT_EQ(batch.Size(), initial_size);
    EXPECT_EQ(batch.BondCount(), initial_bond_count);
    EXPECT_EQ(batch.SegmentBondCount(0), 2u);
}

} // namespace test
} // namespace OEFP
