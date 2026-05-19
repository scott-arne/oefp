#include "oefp/descriptor_arrow.h"
#include "oefp/descriptor_batch.h"

#include <arrow/api.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace OEFP;

namespace {

std::shared_ptr<const DescriptorSchema> scalar_schema() {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float, "mordred:constitutional"});
    builder.Add(DescriptorDefinition{"nAtom", DescriptorValueKind::Int, "mordred:atom_count"});
    builder.Add(DescriptorDefinition{"Lipinski", DescriptorValueKind::Bool, "mordred:filter"});
    builder.Add(DescriptorDefinition{"Class", DescriptorValueKind::String, "manual:category"});
    return builder.Build();
}

DescriptorBatch scalar_batch_with_missing_values() {
    const auto schema = scalar_schema();

    DescriptorSetBuilder complete(schema);
    complete.Set("MW", DescriptorValue::Float(46.069));
    complete.Set("nAtom", DescriptorValue::Int(9));
    complete.Set("Lipinski", DescriptorValue::Bool(true));
    complete.Set("Class", DescriptorValue::String("alcohol"));

    DescriptorSetBuilder partial(schema);
    partial.Set("MW", DescriptorValue::Float(78.114));
    partial.Set("Class", DescriptorValue::String("aromatic"));

    return DescriptorBatch::FromDescriptorSets({complete.Build("CCO"), partial.Build("benzene")});
}

} // namespace

TEST(DescriptorArrowTest, RoundTripsScalarBatchThroughRecordBatch) {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float, "mordred:constitutional"});
    builder.Add(DescriptorDefinition{"nAtom", DescriptorValueKind::Int, "mordred:atom_count"});
    const auto schema = builder.Build();

    DescriptorSetBuilder row(schema);
    row.Set("MW", DescriptorValue::Float(46.069));
    row.Set("nAtom", DescriptorValue::Int(9));
    const auto batch = DescriptorBatch::FromDescriptorSets({row.Build("CCO")});

    const auto record_batch = ToArrowRecordBatch(batch);
    const auto restored = FromArrowRecordBatch(record_batch);

    EXPECT_EQ(restored.Size(), 1u);
    EXPECT_EQ(restored.Schema().Definition(0).name, "MW");
    EXPECT_DOUBLE_EQ(restored.FloatColumn("MW")[0], 46.069);
    EXPECT_EQ(restored.IntColumn("nAtom")[0], 9);
}

TEST(DescriptorArrowTest, MapsBoolStringAndMetadataToRecordBatch) {
    const auto batch = scalar_batch_with_missing_values();

    const auto record_batch = ToArrowRecordBatch(batch);
    const auto metadata = record_batch->schema()->metadata();

    ASSERT_NE(metadata, nullptr);
    EXPECT_TRUE(metadata->Contains("oefp.schema_id"));
    EXPECT_TRUE(metadata->Contains("oefp.descriptor_schema_json"));
    EXPECT_TRUE(metadata->Contains("oefp.format_version"));
    EXPECT_TRUE(metadata->Contains("oefp.row_ids_json"));
    EXPECT_TRUE(record_batch->schema()->field(2)->type()->Equals(arrow::boolean()));
    EXPECT_TRUE(record_batch->schema()->field(3)->type()->Equals(arrow::utf8()));
}

TEST(DescriptorArrowTest, PreservesRowIdsThroughRecordBatchMetadata) {
    const auto batch = scalar_batch_with_missing_values();

    const auto restored = FromArrowRecordBatch(ToArrowRecordBatch(batch));

    EXPECT_EQ(restored.RowIds(), std::vector<std::string>({"CCO", "benzene"}));
}

TEST(DescriptorArrowTest, PreservesMissingValuesAsArrowNulls) {
    const auto batch = scalar_batch_with_missing_values();

    const auto record_batch = ToArrowRecordBatch(batch);
    const auto n_atom = record_batch->column(1);
    const auto lipinski = record_batch->column(2);

    EXPECT_FALSE(n_atom->IsNull(0));
    EXPECT_TRUE(n_atom->IsNull(1));
    EXPECT_FALSE(lipinski->IsNull(0));
    EXPECT_TRUE(lipinski->IsNull(1));

    const auto restored = FromArrowRecordBatch(record_batch);
    EXPECT_EQ(restored.ColumnValidity("nAtom"), std::vector<std::uint8_t>({1u, 0u}));
    EXPECT_EQ(restored.ColumnValidity("Lipinski"), std::vector<std::uint8_t>({1u, 0u}));
}

TEST(DescriptorArrowTest, RejectsAllNullColumnWithWrongArrowType) {
    const auto batch = scalar_batch_with_missing_values();
    const auto good = ToArrowRecordBatch(batch);

    arrow::DoubleBuilder mw_builder;
    ASSERT_TRUE(mw_builder.AppendValues(batch.FloatColumn("MW")).ok());
    std::shared_ptr<arrow::Array> mw_array;
    ASSERT_TRUE(mw_builder.Finish(&mw_array).ok());

    arrow::StringBuilder wrong_builder;
    ASSERT_TRUE(wrong_builder.AppendNulls(static_cast<int64_t>(batch.Size())).ok());
    std::shared_ptr<arrow::Array> wrong_array;
    ASSERT_TRUE(wrong_builder.Finish(&wrong_array).ok());

    arrow::BooleanBuilder lipinski_builder;
    ASSERT_TRUE(lipinski_builder.Append(true).ok());
    ASSERT_TRUE(lipinski_builder.AppendNull().ok());
    std::shared_ptr<arrow::Array> lipinski_array;
    ASSERT_TRUE(lipinski_builder.Finish(&lipinski_array).ok());

    arrow::StringBuilder class_builder;
    ASSERT_TRUE(class_builder.Append("alcohol").ok());
    ASSERT_TRUE(class_builder.Append("aromatic").ok());
    std::shared_ptr<arrow::Array> class_array;
    ASSERT_TRUE(class_builder.Finish(&class_array).ok());

    auto wrong_schema = arrow::schema(
        {
            arrow::field("MW", arrow::float64()),
            arrow::field("nAtom", arrow::utf8()),
            arrow::field("Lipinski", arrow::boolean()),
            arrow::field("Class", arrow::utf8()),
        },
        good->schema()->metadata());
    const auto wrong_batch = arrow::RecordBatch::Make(
        wrong_schema,
        static_cast<int64_t>(batch.Size()),
        {mw_array, wrong_array, lipinski_array, class_array});

    EXPECT_THROW(static_cast<void>(FromArrowRecordBatch(wrong_batch)), std::invalid_argument);
}

TEST(DescriptorArrowTest, PreservesSchemaForZeroRowBatches) {
    const auto empty = DescriptorBatch::Empty(scalar_schema());

    const auto record_batch = ToArrowRecordBatch(empty);
    const auto restored = FromArrowRecordBatch(record_batch);

    EXPECT_EQ(record_batch->num_rows(), 0);
    EXPECT_EQ(restored.Size(), 0u);
    EXPECT_EQ(restored.Schema().SchemaId(), empty.Schema().SchemaId());
    EXPECT_EQ(restored.Schema().Definition(3).name, "Class");
}
