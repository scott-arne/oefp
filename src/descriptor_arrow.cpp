#include "oefp/descriptor_arrow.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>
#include <arrow/table.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace OEFP {
namespace {

constexpr const char* kFormatVersion = "1";
constexpr const char* kSchemaIdKey = "oefp.schema_id";
constexpr const char* kDescriptorSchemaJsonKey = "oefp.descriptor_schema_json";
constexpr const char* kFormatVersionKey = "oefp.format_version";
constexpr const char* kRowIdsJsonKey = "oefp.row_ids_json";

bool is_scalar_kind(DescriptorValueKind kind) {
    switch (kind) {
    case DescriptorValueKind::Bool:
    case DescriptorValueKind::Int:
    case DescriptorValueKind::Float:
    case DescriptorValueKind::String:
        return true;
    case DescriptorValueKind::FloatVector:
    case DescriptorValueKind::IntVector:
    case DescriptorValueKind::FloatMatrix:
    case DescriptorValueKind::IntMatrix:
    case DescriptorValueKind::CountedStringKeys:
    case DescriptorValueKind::CountedIntegerKeys:
    case DescriptorValueKind::CountedFloatKeys:
    case DescriptorValueKind::DenseBinaryFingerprint:
    case DescriptorValueKind::SparseBinaryFingerprint:
    case DescriptorValueKind::DenseCountFingerprint:
    case DescriptorValueKind::SparseCountFingerprint:
        return false;
    }
    throw std::invalid_argument("Descriptor value kind is invalid.");
}

const char* kind_token(DescriptorValueKind kind) {
    switch (kind) {
    case DescriptorValueKind::Bool:
        return "bool";
    case DescriptorValueKind::Int:
        return "int";
    case DescriptorValueKind::Float:
        return "float";
    case DescriptorValueKind::String:
        return "string";
    case DescriptorValueKind::FloatVector:
        return "float_vector";
    case DescriptorValueKind::IntVector:
        return "int_vector";
    case DescriptorValueKind::FloatMatrix:
        return "float_matrix";
    case DescriptorValueKind::IntMatrix:
        return "int_matrix";
    case DescriptorValueKind::CountedStringKeys:
        return "counted_string_keys";
    case DescriptorValueKind::CountedIntegerKeys:
        return "counted_integer_keys";
    case DescriptorValueKind::CountedFloatKeys:
        return "counted_float_keys";
    case DescriptorValueKind::DenseBinaryFingerprint:
        return "dense_binary_fingerprint";
    case DescriptorValueKind::SparseBinaryFingerprint:
        return "sparse_binary_fingerprint";
    case DescriptorValueKind::DenseCountFingerprint:
        return "dense_count_fingerprint";
    case DescriptorValueKind::SparseCountFingerprint:
        return "sparse_count_fingerprint";
    }
    throw std::invalid_argument("Descriptor value kind is invalid.");
}

DescriptorValueKind kind_from_token(const std::string& token) {
    if (token == "bool") {
        return DescriptorValueKind::Bool;
    }
    if (token == "int") {
        return DescriptorValueKind::Int;
    }
    if (token == "float") {
        return DescriptorValueKind::Float;
    }
    if (token == "string") {
        return DescriptorValueKind::String;
    }
    if (token == "float_vector") {
        return DescriptorValueKind::FloatVector;
    }
    if (token == "int_vector") {
        return DescriptorValueKind::IntVector;
    }
    if (token == "float_matrix") {
        return DescriptorValueKind::FloatMatrix;
    }
    if (token == "int_matrix") {
        return DescriptorValueKind::IntMatrix;
    }
    if (token == "counted_string_keys") {
        return DescriptorValueKind::CountedStringKeys;
    }
    if (token == "counted_integer_keys") {
        return DescriptorValueKind::CountedIntegerKeys;
    }
    if (token == "counted_float_keys") {
        return DescriptorValueKind::CountedFloatKeys;
    }
    if (token == "dense_binary_fingerprint") {
        return DescriptorValueKind::DenseBinaryFingerprint;
    }
    if (token == "sparse_binary_fingerprint") {
        return DescriptorValueKind::SparseBinaryFingerprint;
    }
    if (token == "dense_count_fingerprint") {
        return DescriptorValueKind::DenseCountFingerprint;
    }
    if (token == "sparse_count_fingerprint") {
        return DescriptorValueKind::SparseCountFingerprint;
    }
    throw std::invalid_argument("Descriptor schema metadata contains an invalid value kind.");
}

std::shared_ptr<arrow::DataType> arrow_type_for(DescriptorValueKind kind) {
    switch (kind) {
    case DescriptorValueKind::Bool:
        return arrow::boolean();
    case DescriptorValueKind::Int:
        return arrow::int64();
    case DescriptorValueKind::Float:
        return arrow::float64();
    case DescriptorValueKind::String:
        return arrow::utf8();
    case DescriptorValueKind::FloatVector:
    case DescriptorValueKind::IntVector:
    case DescriptorValueKind::FloatMatrix:
    case DescriptorValueKind::IntMatrix:
    case DescriptorValueKind::CountedStringKeys:
    case DescriptorValueKind::CountedIntegerKeys:
    case DescriptorValueKind::CountedFloatKeys:
    case DescriptorValueKind::DenseBinaryFingerprint:
    case DescriptorValueKind::SparseBinaryFingerprint:
    case DescriptorValueKind::DenseCountFingerprint:
    case DescriptorValueKind::SparseCountFingerprint:
        break;
    }
    throw std::invalid_argument("Descriptor kind cannot be represented as a scalar Arrow type.");
}

void check_arrow_status(const arrow::Status& status) {
    if (!status.ok()) {
        throw std::runtime_error(status.ToString());
    }
}

template <typename T>
T unwrap_arrow_result(arrow::Result<T> result) {
    check_arrow_status(result.status());
    return std::move(result).ValueOrDie();
}

std::string escape_json(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const auto ch : text) {
        switch (ch) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            escaped.push_back(ch);
            break;
        }
    }
    return escaped;
}

void append_json_string_field(std::string& json, const char* name, const std::string& value) {
    json += '"';
    json += name;
    json += "\":\"";
    json += escape_json(value);
    json += '"';
}

std::string schema_to_json(const DescriptorSchema& schema) {
    std::string json = "{\"definitions\":[";
    for (std::size_t index = 0; index < schema.Size(); ++index) {
        if (index != 0u) {
            json += ',';
        }
        const auto& definition = schema.Definition(index);
        json += '{';
        append_json_string_field(json, "name", definition.name);
        json += ',';
        append_json_string_field(json, "value_kind", kind_token(definition.value_kind));
        json += ',';
        append_json_string_field(json, "group", definition.group);
        json += ',';
        append_json_string_field(json, "source_name", definition.source_name);
        json += ',';
        append_json_string_field(json, "source_type", definition.source_type);
        json += ',';
        append_json_string_field(json, "source_version", definition.source_version);
        json += ',';
        append_json_string_field(json, "parameters", definition.parameters);
        json += ',';
        append_json_string_field(json, "units", definition.units);
        json += ',';
        append_json_string_field(json, "description", definition.description);
        json += ",\"shape\":";
        if (definition.shape.has_value()) {
            json += '[';
            for (std::size_t dim = 0; dim < definition.shape->dimensions.size(); ++dim) {
                if (dim != 0u) {
                    json += ',';
                }
                json += std::to_string(definition.shape->dimensions[dim]);
            }
            json += ']';
        } else {
            json += "null";
        }
        json += '}';
    }
    json += "]}";
    return json;
}

std::string string_vector_to_json(const std::vector<std::string>& values) {
    std::string json = "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0u) {
            json += ',';
        }
        json += '"';
        json += escape_json(values[index]);
        json += '"';
    }
    json += ']';
    return json;
}

class SchemaJsonParser {
public:
    explicit SchemaJsonParser(std::string json)
        : json_(std::move(json)) {
    }

    std::shared_ptr<const DescriptorSchema> Parse() {
        DescriptorSchemaBuilder builder;
        Expect('{');
        ExpectString("definitions");
        Expect(':');
        Expect('[');
        if (!Consume(']')) {
            do {
                builder.Add(ParseDefinition());
            } while (Consume(','));
            Expect(']');
        }
        Expect('}');
        ExpectEnd();
        return builder.Build();
    }

private:
    std::string json_;
    std::size_t offset_ = 0u;

    DescriptorDefinition ParseDefinition() {
        DescriptorDefinition definition;
        Expect('{');
        definition.name = ParseNamedString("name");
        Expect(',');
        definition.value_kind = kind_from_token(ParseNamedString("value_kind"));
        Expect(',');
        definition.group = ParseNamedString("group");
        Expect(',');
        definition.source_name = ParseNamedString("source_name");
        Expect(',');
        definition.source_type = ParseNamedString("source_type");
        Expect(',');
        definition.source_version = ParseNamedString("source_version");
        Expect(',');
        definition.parameters = ParseNamedString("parameters");
        Expect(',');
        definition.units = ParseNamedString("units");
        Expect(',');
        definition.description = ParseNamedString("description");
        Expect(',');
        ExpectString("shape");
        Expect(':');
        definition.shape = ParseShape();
        Expect('}');
        return definition;
    }

    std::optional<DescriptorShape> ParseShape() {
        if (ConsumeNull()) {
            return std::nullopt;
        }
        DescriptorShape shape;
        Expect('[');
        if (!Consume(']')) {
            do {
                shape.dimensions.push_back(ParseUnsigned());
            } while (Consume(','));
            Expect(']');
        }
        return shape;
    }

    std::uint64_t ParseUnsigned() {
        SkipWhitespace();
        if (offset_ >= json_.size() || json_[offset_] < '0' || json_[offset_] > '9') {
            throw std::invalid_argument("Descriptor schema metadata contains invalid JSON.");
        }
        std::uint64_t value = 0u;
        while (offset_ < json_.size() && json_[offset_] >= '0' && json_[offset_] <= '9') {
            value = (value * 10u) + static_cast<std::uint64_t>(json_[offset_] - '0');
            ++offset_;
        }
        return value;
    }

    std::string ParseNamedString(const char* name) {
        ExpectString(name);
        Expect(':');
        return ParseString();
    }

    std::string ParseString() {
        SkipWhitespace();
        Expect('"');
        std::string value;
        while (offset_ < json_.size()) {
            const auto ch = json_[offset_++];
            if (ch == '"') {
                return value;
            }
            if (ch != '\\') {
                value.push_back(ch);
                continue;
            }
            if (offset_ >= json_.size()) {
                break;
            }
            const auto escaped = json_[offset_++];
            switch (escaped) {
            case '"':
            case '\\':
            case '/':
                value.push_back(escaped);
                break;
            case 'b':
                value.push_back('\b');
                break;
            case 'f':
                value.push_back('\f');
                break;
            case 'n':
                value.push_back('\n');
                break;
            case 'r':
                value.push_back('\r');
                break;
            case 't':
                value.push_back('\t');
                break;
            default:
                throw std::invalid_argument("Descriptor schema metadata contains invalid JSON.");
            }
        }
        throw std::invalid_argument("Descriptor schema metadata contains invalid JSON.");
    }

    void ExpectString(const std::string& expected) {
        const auto actual = ParseString();
        if (actual != expected) {
            throw std::invalid_argument("Descriptor schema metadata contains unexpected JSON.");
        }
    }

    bool ConsumeNull() {
        SkipWhitespace();
        if (json_.compare(offset_, 4, "null") == 0) {
            offset_ += 4u;
            return true;
        }
        return false;
    }

    bool Consume(char expected) {
        SkipWhitespace();
        if (offset_ < json_.size() && json_[offset_] == expected) {
            ++offset_;
            return true;
        }
        return false;
    }

    void Expect(char expected) {
        if (!Consume(expected)) {
            throw std::invalid_argument("Descriptor schema metadata contains invalid JSON.");
        }
    }

    void ExpectEnd() {
        SkipWhitespace();
        if (offset_ != json_.size()) {
            throw std::invalid_argument("Descriptor schema metadata contains trailing JSON.");
        }
    }

    void SkipWhitespace() {
        while (offset_ < json_.size()) {
            const auto ch = json_[offset_];
            if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
                break;
            }
            ++offset_;
        }
    }
};

class StringArrayJsonParser {
public:
    explicit StringArrayJsonParser(std::string json)
        : json_(std::move(json)) {
    }

    std::vector<std::string> Parse() {
        std::vector<std::string> values;
        Expect('[');
        if (!Consume(']')) {
            do {
                values.push_back(ParseString());
            } while (Consume(','));
            Expect(']');
        }
        ExpectEnd();
        return values;
    }

private:
    std::string json_;
    std::size_t offset_ = 0u;

    std::string ParseString() {
        SkipWhitespace();
        Expect('"');
        std::string value;
        while (offset_ < json_.size()) {
            const auto ch = json_[offset_++];
            if (ch == '"') {
                return value;
            }
            if (ch != '\\') {
                value.push_back(ch);
                continue;
            }
            if (offset_ >= json_.size()) {
                break;
            }
            const auto escaped = json_[offset_++];
            switch (escaped) {
            case '"':
            case '\\':
            case '/':
                value.push_back(escaped);
                break;
            case 'b':
                value.push_back('\b');
                break;
            case 'f':
                value.push_back('\f');
                break;
            case 'n':
                value.push_back('\n');
                break;
            case 'r':
                value.push_back('\r');
                break;
            case 't':
                value.push_back('\t');
                break;
            default:
                throw std::invalid_argument("Descriptor row id metadata contains invalid JSON.");
            }
        }
        throw std::invalid_argument("Descriptor row id metadata contains invalid JSON.");
    }

    bool Consume(char expected) {
        SkipWhitespace();
        if (offset_ < json_.size() && json_[offset_] == expected) {
            ++offset_;
            return true;
        }
        return false;
    }

    void Expect(char expected) {
        if (!Consume(expected)) {
            throw std::invalid_argument("Descriptor row id metadata contains invalid JSON.");
        }
    }

    void ExpectEnd() {
        SkipWhitespace();
        if (offset_ != json_.size()) {
            throw std::invalid_argument("Descriptor row id metadata contains trailing JSON.");
        }
    }

    void SkipWhitespace() {
        while (offset_ < json_.size()) {
            const auto ch = json_[offset_];
            if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
                break;
            }
            ++offset_;
        }
    }
};

std::shared_ptr<const DescriptorSchema> schema_from_metadata(
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
    if (metadata == nullptr) {
        throw std::invalid_argument("Arrow schema is missing OEFP descriptor metadata.");
    }
    if (!metadata->Contains(kFormatVersionKey)) {
        throw std::invalid_argument("Arrow schema is missing OEFP descriptor format metadata.");
    }
    const auto version = metadata->Get(kFormatVersionKey).ValueOrDie();
    if (version != kFormatVersion) {
        throw std::invalid_argument("Arrow schema has an unsupported OEFP descriptor format version.");
    }
    if (!metadata->Contains(kDescriptorSchemaJsonKey)) {
        throw std::invalid_argument("Arrow schema is missing OEFP descriptor schema metadata.");
    }
    auto schema = SchemaJsonParser(metadata->Get(kDescriptorSchemaJsonKey).ValueOrDie()).Parse();
    if (!metadata->Contains(kSchemaIdKey)) {
        throw std::invalid_argument("Arrow schema is missing OEFP descriptor schema id metadata.");
    }
    const auto schema_id = metadata->Get(kSchemaIdKey).ValueOrDie();
    if (schema_id != schema->SchemaId()) {
        throw std::invalid_argument("Arrow schema OEFP descriptor schema id does not match metadata.");
    }
    return schema;
}

std::vector<std::string> row_ids_from_metadata(
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata,
    std::int64_t row_count) {
    if (metadata == nullptr || !metadata->Contains(kRowIdsJsonKey)) {
        return std::vector<std::string>(static_cast<std::size_t>(row_count));
    }
    auto row_ids = StringArrayJsonParser(metadata->Get(kRowIdsJsonKey).ValueOrDie()).Parse();
    if (row_ids.size() != static_cast<std::size_t>(row_count)) {
        throw std::invalid_argument("Arrow schema row id metadata does not match row count.");
    }
    return row_ids;
}

std::shared_ptr<arrow::Array> finish_array(arrow::ArrayBuilder& builder) {
    std::shared_ptr<arrow::Array> array;
    check_arrow_status(builder.Finish(&array));
    return array;
}

std::shared_ptr<arrow::Array> build_bool_array(const DescriptorBatch& batch, const std::string& name) {
    arrow::BooleanBuilder builder;
    const auto values = batch.BoolColumn(name);
    const auto validity = batch.ColumnValidity(name);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (validity[index] == 0u) {
            check_arrow_status(builder.AppendNull());
            continue;
        }
        const auto value = values[index];
        check_arrow_status(builder.Append(value != 0u));
    }
    return finish_array(builder);
}

std::shared_ptr<arrow::Array> build_int_array(const DescriptorBatch& batch, const std::string& name) {
    arrow::Int64Builder builder;
    const auto values = batch.IntColumn(name);
    const auto validity = batch.ColumnValidity(name);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (validity[index] == 0u) {
            check_arrow_status(builder.AppendNull());
            continue;
        }
        check_arrow_status(builder.Append(values[index]));
    }
    return finish_array(builder);
}

std::shared_ptr<arrow::Array> build_float_array(const DescriptorBatch& batch, const std::string& name) {
    arrow::DoubleBuilder builder;
    const auto values = batch.FloatColumn(name);
    const auto validity = batch.ColumnValidity(name);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (validity[index] == 0u) {
            check_arrow_status(builder.AppendNull());
            continue;
        }
        check_arrow_status(builder.Append(values[index]));
    }
    return finish_array(builder);
}

std::shared_ptr<arrow::Array> build_string_array(const DescriptorBatch& batch, const std::string& name) {
    arrow::StringBuilder builder;
    const auto values = batch.StringColumn(name);
    const auto validity = batch.ColumnValidity(name);
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (validity[index] == 0u) {
            check_arrow_status(builder.AppendNull());
            continue;
        }
        check_arrow_status(builder.Append(values[index]));
    }
    return finish_array(builder);
}

void validate_arrow_batch_schema(
    const std::shared_ptr<arrow::RecordBatch>& batch,
    const DescriptorSchema& descriptor_schema) {
    check_arrow_status(batch->ValidateFull());
    if (static_cast<std::size_t>(batch->num_columns()) != descriptor_schema.Size()) {
        throw std::invalid_argument("Arrow record batch column count does not match descriptor schema.");
    }
    for (std::size_t column_index = 0; column_index < descriptor_schema.Size(); ++column_index) {
        const auto& definition = descriptor_schema.Definition(column_index);
        const auto field = batch->schema()->field(static_cast<int>(column_index));
        if (field->name() != definition.name) {
            throw std::invalid_argument("Arrow record batch columns do not match descriptor schema.");
        }
        if (!field->type()->Equals(arrow_type_for(definition.value_kind))) {
            throw std::invalid_argument("Arrow field type does not match descriptor schema.");
        }
    }
}

DescriptorValue value_from_array(
    const std::shared_ptr<arrow::Array>& array,
    DescriptorValueKind kind,
    std::int64_t row,
    const std::string& name) {
    switch (kind) {
    case DescriptorValueKind::Bool: {
        const auto typed = std::dynamic_pointer_cast<arrow::BooleanArray>(array);
        if (typed == nullptr) {
            throw std::invalid_argument("Arrow column type does not match descriptor bool column.");
        }
        return DescriptorValue::Bool(typed->Value(row));
    }
    case DescriptorValueKind::Int: {
        const auto typed = std::dynamic_pointer_cast<arrow::Int64Array>(array);
        if (typed == nullptr) {
            throw std::invalid_argument("Arrow column type does not match descriptor int column.");
        }
        return DescriptorValue::Int(typed->Value(row));
    }
    case DescriptorValueKind::Float: {
        const auto typed = std::dynamic_pointer_cast<arrow::DoubleArray>(array);
        if (typed == nullptr) {
            throw std::invalid_argument("Arrow column type does not match descriptor float column.");
        }
        return DescriptorValue::Float(typed->Value(row));
    }
    case DescriptorValueKind::String: {
        const auto typed = std::dynamic_pointer_cast<arrow::StringArray>(array);
        if (typed == nullptr) {
            throw std::invalid_argument("Arrow column type does not match descriptor string column.");
        }
        return DescriptorValue::String(typed->GetString(row));
    }
    case DescriptorValueKind::FloatVector:
    case DescriptorValueKind::IntVector:
    case DescriptorValueKind::FloatMatrix:
    case DescriptorValueKind::IntMatrix:
    case DescriptorValueKind::CountedStringKeys:
    case DescriptorValueKind::CountedIntegerKeys:
    case DescriptorValueKind::CountedFloatKeys:
    case DescriptorValueKind::DenseBinaryFingerprint:
    case DescriptorValueKind::SparseBinaryFingerprint:
    case DescriptorValueKind::DenseCountFingerprint:
    case DescriptorValueKind::SparseCountFingerprint:
        break;
    }
    throw std::invalid_argument("Arrow column '" + name + "' is not a scalar descriptor column.");
}

} // namespace

std::shared_ptr<arrow::RecordBatch> ToArrowRecordBatch(const DescriptorBatch& batch) {
    const auto& descriptor_schema = batch.Schema();
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    fields.reserve(descriptor_schema.Size());
    arrays.reserve(descriptor_schema.Size());

    for (const auto& definition : descriptor_schema.Definitions()) {
        if (!is_scalar_kind(definition.value_kind)) {
            throw std::invalid_argument("Arrow descriptor conversion supports scalar columns only.");
        }
        fields.push_back(arrow::field(definition.name, arrow_type_for(definition.value_kind)));
        switch (definition.value_kind) {
        case DescriptorValueKind::Bool:
            arrays.push_back(build_bool_array(batch, definition.name));
            break;
        case DescriptorValueKind::Int:
            arrays.push_back(build_int_array(batch, definition.name));
            break;
        case DescriptorValueKind::Float:
            arrays.push_back(build_float_array(batch, definition.name));
            break;
        case DescriptorValueKind::String:
            arrays.push_back(build_string_array(batch, definition.name));
            break;
        case DescriptorValueKind::FloatVector:
        case DescriptorValueKind::IntVector:
        case DescriptorValueKind::FloatMatrix:
        case DescriptorValueKind::IntMatrix:
        case DescriptorValueKind::CountedStringKeys:
        case DescriptorValueKind::CountedIntegerKeys:
        case DescriptorValueKind::CountedFloatKeys:
        case DescriptorValueKind::DenseBinaryFingerprint:
        case DescriptorValueKind::SparseBinaryFingerprint:
        case DescriptorValueKind::DenseCountFingerprint:
        case DescriptorValueKind::SparseCountFingerprint:
            throw std::invalid_argument("Arrow descriptor conversion supports scalar columns only.");
        }
    }

    // Row IDs live in schema metadata instead of a hidden column so Arrow fields
    // remain a direct descriptor-definition projection.
    auto metadata = arrow::KeyValueMetadata::Make(
        {kSchemaIdKey, kDescriptorSchemaJsonKey, kFormatVersionKey, kRowIdsJsonKey},
        {
            descriptor_schema.SchemaId(),
            schema_to_json(descriptor_schema),
            kFormatVersion,
            string_vector_to_json(batch.RowIds()),
        });
    auto arrow_schema = arrow::schema(std::move(fields), std::move(metadata));
    return arrow::RecordBatch::Make(std::move(arrow_schema), static_cast<int64_t>(batch.Size()), arrays);
}

DescriptorBatch FromArrowRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch) {
    if (batch == nullptr) {
        throw std::invalid_argument("Arrow record batch must not be null.");
    }
    const auto descriptor_schema = schema_from_metadata(batch->schema()->metadata());
    validate_arrow_batch_schema(batch, *descriptor_schema);
    const auto row_ids = row_ids_from_metadata(batch->schema()->metadata(), batch->num_rows());
    if (batch->num_rows() == 0) {
        return DescriptorBatch::Empty(descriptor_schema);
    }

    std::vector<DescriptorSet> rows;
    rows.reserve(static_cast<std::size_t>(batch->num_rows()));
    for (std::int64_t row_index = 0; row_index < batch->num_rows(); ++row_index) {
        DescriptorSetBuilder builder(descriptor_schema);
        for (std::size_t column_index = 0; column_index < descriptor_schema->Size(); ++column_index) {
            const auto& definition = descriptor_schema->Definition(column_index);
            const auto column = batch->column(static_cast<int>(column_index));
            if (column->IsNull(row_index)) {
                continue;
            }
            builder.Set(
                definition.name,
                value_from_array(column, definition.value_kind, row_index, definition.name));
        }
        rows.push_back(builder.Build(row_ids[static_cast<std::size_t>(row_index)]));
    }
    return DescriptorBatch::FromDescriptorSets(rows);
}

void WriteDescriptorIpc(const DescriptorBatch& batch, const std::string& path) {
    const auto record_batch = ToArrowRecordBatch(batch);
    auto output = unwrap_arrow_result(arrow::io::FileOutputStream::Open(path));
    auto writer = unwrap_arrow_result(arrow::ipc::MakeFileWriter(output, record_batch->schema()));

    check_arrow_status(writer->WriteRecordBatch(*record_batch));
    check_arrow_status(writer->Close());
    check_arrow_status(output->Close());
}

DescriptorBatch ReadDescriptorIpc(const std::string& path) {
    auto input = unwrap_arrow_result(arrow::io::ReadableFile::Open(path));
    auto reader = unwrap_arrow_result(arrow::ipc::RecordBatchFileReader::Open(input));
    if (reader->num_record_batches() != 1) {
        throw std::invalid_argument("Arrow IPC descriptor file must contain exactly one record batch.");
    }
    auto record_batch = unwrap_arrow_result(reader->ReadRecordBatch(0));
    check_arrow_status(input->Close());
    return FromArrowRecordBatch(record_batch);
}

void WriteDescriptorParquet(const DescriptorBatch& batch, const std::string& path) {
    const auto record_batch = ToArrowRecordBatch(batch);
    auto table = unwrap_arrow_result(arrow::Table::FromRecordBatches({record_batch}));
    auto output = unwrap_arrow_result(arrow::io::FileOutputStream::Open(path));
    parquet::ArrowWriterProperties::Builder properties_builder;
    const auto arrow_properties = properties_builder.store_schema()->build();

    check_arrow_status(parquet::arrow::WriteTable(
        *table,
        arrow::default_memory_pool(),
        output,
        parquet::DEFAULT_MAX_ROW_GROUP_LENGTH,
        parquet::default_writer_properties(),
        arrow_properties));
    check_arrow_status(output->Close());
}

DescriptorBatch ReadDescriptorParquet(const std::string& path) {
    auto input = unwrap_arrow_result(arrow::io::ReadableFile::Open(path));
    parquet::arrow::FileReaderBuilder builder;
    check_arrow_status(builder.Open(input));
    auto reader = unwrap_arrow_result(builder.Build());
    auto table = unwrap_arrow_result(reader->ReadTable());
    auto record_batch = unwrap_arrow_result(table->CombineChunksToBatch(arrow::default_memory_pool()));

    check_arrow_status(input->Close());
    return FromArrowRecordBatch(record_batch);
}

} // namespace OEFP
