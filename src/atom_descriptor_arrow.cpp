#include "oefp/atom_descriptor_arrow.h"

#include <arrow/api.h>
#include <arrow/io/file.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/writer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace OEFP {
namespace {

constexpr const char* kFormatVersion = "1";
constexpr const char* kDescriptorSchemaJsonKey = "oefp.descriptor_schema_json";
constexpr const char* kFormatVersionKey = "oefp.format_version";
constexpr const char* kMoleculeCountKey = "oefp.kallisto.molecule_count";

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
        append_json_string_field(json, "units", definition.units);
        json += '}';
    }
    json += "]}";
    return json;
}

class SchemaJsonParser {
public:
    explicit SchemaJsonParser(std::string json)
        : json_(std::move(json)) {
    }

    std::vector<std::string> Parse() {
        std::vector<std::string> names;
        Expect('{');
        ExpectString("definitions");
        Expect(':');
        Expect('[');
        if (!Consume(']')) {
            do {
                names.push_back(ParseDefinition());
            } while (Consume(','));
            Expect(']');
        }
        Expect('}');
        ExpectEnd();
        return names;
    }

private:
    std::string json_;
    std::size_t offset_ = 0u;

    std::string ParseDefinition() {
        Expect('{');
        const auto name = ParseNamedString("name");
        Expect(',');
        // Skip units field
        ExpectString("units");
        Expect(':');
        ParseString();
        Expect('}');
        return name;
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
                throw std::invalid_argument("Schema metadata contains invalid JSON.");
            }
        }
        throw std::invalid_argument("Schema metadata contains invalid JSON.");
    }

    void ExpectString(const std::string& expected) {
        const auto actual = ParseString();
        if (actual != expected) {
            throw std::invalid_argument("Schema metadata contains unexpected JSON.");
        }
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
            throw std::invalid_argument("Schema metadata contains invalid JSON.");
        }
    }

    void ExpectEnd() {
        SkipWhitespace();
        if (offset_ != json_.size()) {
            throw std::invalid_argument("Schema metadata contains trailing JSON.");
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

std::size_t parse_molecule_count(const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) {
    if (metadata == nullptr || !metadata->Contains(kMoleculeCountKey)) {
        throw std::invalid_argument("Arrow schema is missing molecule count metadata.");
    }
    const auto count_str = metadata->Get(kMoleculeCountKey).ValueOrDie();
    try {
        return static_cast<std::size_t>(std::stoull(count_str));
    } catch (...) {
        throw std::invalid_argument("Arrow molecule count metadata is invalid.");
    }
}

std::shared_ptr<arrow::Array> finish_array(arrow::ArrayBuilder& builder) {
    std::shared_ptr<arrow::Array> array;
    check_arrow_status(builder.Finish(&array));
    return array;
}

} // namespace

std::shared_ptr<arrow::RecordBatch> AtomDescriptorBatchToArrow(
    const AtomDescriptorBatch& batch) {
    const auto& schema = batch.Schema();
    const auto molecule_count = batch.Size();
    const auto atom_count = batch.AtomCount();

    // Build molecule_id column (segment index for each atom)
    arrow::UInt32Builder molecule_id_builder;
    for (std::size_t mol = 0; mol < molecule_count; ++mol) {
        const auto seg_atom_count = batch.SegmentAtomCount(mol);
        for (std::size_t i = 0; i < seg_atom_count; ++i) {
            check_arrow_status(molecule_id_builder.Append(static_cast<std::uint32_t>(mol)));
        }
    }
    auto molecule_id_array = finish_array(molecule_id_builder);

    // Build atom_index column
    arrow::UInt32Builder atom_index_builder;
    const auto* atom_indices = reinterpret_cast<const std::uint32_t*>(
        batch.AtomIndexDataAddress());
    for (std::size_t i = 0; i < atom_count; ++i) {
        check_arrow_status(atom_index_builder.Append(atom_indices[i]));
    }
    auto atom_index_array = finish_array(atom_index_builder);

    // Build descriptor columns
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    fields.reserve(2u + schema.Size());
    arrays.reserve(2u + schema.Size());

    fields.push_back(arrow::field("molecule_id", arrow::uint32()));
    arrays.push_back(std::move(molecule_id_array));
    fields.push_back(arrow::field("atom_index", arrow::uint32()));
    arrays.push_back(std::move(atom_index_array));

    for (std::size_t col = 0; col < schema.Size(); ++col) {
        const auto& definition = schema.Definition(col);
        arrow::DoubleBuilder double_builder;

        const auto* values = reinterpret_cast<const double*>(
            batch.ColumnDataAddress(col));
        const auto* validity = reinterpret_cast<const std::uint8_t*>(
            batch.ColumnValidityAddress(col));

        for (std::size_t i = 0; i < atom_count; ++i) {
            if (validity[i] == 0u) {
                check_arrow_status(double_builder.AppendNull());
            } else {
                check_arrow_status(double_builder.Append(values[i]));
            }
        }

        fields.push_back(arrow::field(definition.name, arrow::float64()));
        arrays.push_back(finish_array(double_builder));
    }

    auto metadata = arrow::KeyValueMetadata::Make(
        {kDescriptorSchemaJsonKey, kFormatVersionKey, kMoleculeCountKey},
        {
            schema_to_json(schema),
            kFormatVersion,
            std::to_string(molecule_count),
        });

    auto arrow_schema = arrow::schema(std::move(fields), std::move(metadata));
    return arrow::RecordBatch::Make(
        std::move(arrow_schema),
        static_cast<std::int64_t>(atom_count),
        arrays);
}

AtomDescriptorBatch AtomDescriptorBatchFromArrow(
    const std::shared_ptr<arrow::RecordBatch>& rb) {
    if (rb == nullptr) {
        throw std::invalid_argument("Arrow record batch must not be null.");
    }

    const auto metadata = rb->schema()->metadata();
    if (metadata == nullptr) {
        throw std::invalid_argument("Arrow schema is missing OEFP metadata.");
    }

    const auto molecule_count = parse_molecule_count(metadata);
    const auto column_names = SchemaJsonParser(
        metadata->Get(kDescriptorSchemaJsonKey).ValueOrDie()).Parse();

    if (rb->num_columns() < 2) {
        throw std::invalid_argument("Arrow record batch must have at least molecule_id and atom_index.");
    }

    const auto molecule_id_column = std::dynamic_pointer_cast<arrow::UInt32Array>(rb->column(0));
    const auto atom_index_column = std::dynamic_pointer_cast<arrow::UInt32Array>(rb->column(1));

    if (molecule_id_column == nullptr || atom_index_column == nullptr) {
        throw std::invalid_argument("Arrow molecule_id or atom_index column has wrong type.");
    }

    if (static_cast<std::size_t>(rb->num_columns() - 2) != column_names.size()) {
        throw std::invalid_argument("Arrow column count does not match schema.");
    }

    // Group rows by molecule_id
    std::vector<std::vector<std::size_t>> mol_row_indices(molecule_count);
    for (std::int64_t row = 0; row < rb->num_rows(); ++row) {
        const auto mol_id = molecule_id_column->Value(row);
        if (mol_id >= molecule_count) {
            throw std::invalid_argument("Arrow molecule_id out of range.");
        }
        mol_row_indices[mol_id].push_back(static_cast<std::size_t>(row));
    }

    // Build batch segment by segment
    // We need the schema - reconstruct it from the first segment's data or use a minimal schema
    // For now, we'll build a minimal schema from column names
    DescriptorSchemaBuilder schema_builder;
    for (const auto& name : column_names) {
        DescriptorDefinition def;
        def.name = name;
        def.value_kind = DescriptorValueKind::Float;
        def.group = "kallisto";
        def.source_name = "kallisto";
        def.source_type = "geometric";
        def.source_version = "kallisto-1.0.10";
        def.units = "Bohr";
        def.prerequisites = kDescriptorPrerequisiteCoordinates3D;
        schema_builder.Add(std::move(def));
    }
    auto schema = schema_builder.Build();

    auto batch = AtomDescriptorBatch::Empty(schema);

    for (std::size_t mol = 0; mol < molecule_count; ++mol) {
        const auto& row_indices = mol_row_indices[mol];
        if (row_indices.empty()) {
            // Empty segment
            batch.Append(AtomDescriptorSet::Empty(schema));
            continue;
        }

        // Extract atom_indices
        std::vector<std::uint32_t> atom_indices;
        atom_indices.reserve(row_indices.size());
        for (const auto row : row_indices) {
            atom_indices.push_back(atom_index_column->Value(static_cast<std::int64_t>(row)));
        }

        // Extract column values
        std::vector<std::vector<std::optional<double>>> columns(schema->Size());
        for (std::size_t col = 0; col < schema->Size(); ++col) {
            columns[col].reserve(row_indices.size());
            const auto arrow_col = std::dynamic_pointer_cast<arrow::DoubleArray>(
                rb->column(static_cast<int>(col + 2)));
            if (arrow_col == nullptr) {
                throw std::invalid_argument("Arrow descriptor column has wrong type.");
            }
            for (const auto row : row_indices) {
                if (arrow_col->IsNull(static_cast<std::int64_t>(row))) {
                    columns[col].push_back(std::nullopt);
                } else {
                    columns[col].push_back(arrow_col->Value(static_cast<std::int64_t>(row)));
                }
            }
        }

        batch.Append(AtomDescriptorSet(schema, std::move(atom_indices), std::move(columns)));
    }

    return batch;
}

std::shared_ptr<arrow::RecordBatch> BondDescriptorBatchToArrow(
    const BondDescriptorBatch& batch) {
    const auto& schema = batch.Schema();
    const auto molecule_count = batch.Size();
    const auto bond_count = batch.BondCount();

    // Build molecule_id column
    arrow::UInt32Builder molecule_id_builder;
    for (std::size_t mol = 0; mol < molecule_count; ++mol) {
        const auto seg_bond_count = batch.SegmentBondCount(mol);
        for (std::size_t i = 0; i < seg_bond_count; ++i) {
            check_arrow_status(molecule_id_builder.Append(static_cast<std::uint32_t>(mol)));
        }
    }
    auto molecule_id_array = finish_array(molecule_id_builder);

    // Build begin/end columns
    arrow::UInt32Builder begin_builder;
    arrow::UInt32Builder end_builder;
    const auto* bond_begin = reinterpret_cast<const std::uint32_t*>(
        batch.BondBeginDataAddress());
    const auto* bond_end = reinterpret_cast<const std::uint32_t*>(
        batch.BondEndDataAddress());
    for (std::size_t i = 0; i < bond_count; ++i) {
        check_arrow_status(begin_builder.Append(bond_begin[i]));
        check_arrow_status(end_builder.Append(bond_end[i]));
    }
    auto begin_array = finish_array(begin_builder);
    auto end_array = finish_array(end_builder);

    // Build descriptor columns
    std::vector<std::shared_ptr<arrow::Field>> fields;
    std::vector<std::shared_ptr<arrow::Array>> arrays;
    fields.reserve(3u + schema.Size());
    arrays.reserve(3u + schema.Size());

    fields.push_back(arrow::field("molecule_id", arrow::uint32()));
    arrays.push_back(std::move(molecule_id_array));
    fields.push_back(arrow::field("begin", arrow::uint32()));
    arrays.push_back(std::move(begin_array));
    fields.push_back(arrow::field("end", arrow::uint32()));
    arrays.push_back(std::move(end_array));

    for (std::size_t col = 0; col < schema.Size(); ++col) {
        const auto& definition = schema.Definition(col);
        arrow::DoubleBuilder double_builder;

        const auto* values = reinterpret_cast<const double*>(
            batch.ColumnDataAddress(col));
        const auto* validity = reinterpret_cast<const std::uint8_t*>(
            batch.ColumnValidityAddress(col));

        for (std::size_t i = 0; i < bond_count; ++i) {
            if (validity[i] == 0u) {
                check_arrow_status(double_builder.AppendNull());
            } else {
                check_arrow_status(double_builder.Append(values[i]));
            }
        }

        fields.push_back(arrow::field(definition.name, arrow::float64()));
        arrays.push_back(finish_array(double_builder));
    }

    auto metadata = arrow::KeyValueMetadata::Make(
        {kDescriptorSchemaJsonKey, kFormatVersionKey, kMoleculeCountKey},
        {
            schema_to_json(schema),
            kFormatVersion,
            std::to_string(molecule_count),
        });

    auto arrow_schema = arrow::schema(std::move(fields), std::move(metadata));
    return arrow::RecordBatch::Make(
        std::move(arrow_schema),
        static_cast<std::int64_t>(bond_count),
        arrays);
}

BondDescriptorBatch BondDescriptorBatchFromArrow(
    const std::shared_ptr<arrow::RecordBatch>& rb) {
    if (rb == nullptr) {
        throw std::invalid_argument("Arrow record batch must not be null.");
    }

    const auto metadata = rb->schema()->metadata();
    if (metadata == nullptr) {
        throw std::invalid_argument("Arrow schema is missing OEFP metadata.");
    }

    const auto molecule_count = parse_molecule_count(metadata);
    const auto column_names = SchemaJsonParser(
        metadata->Get(kDescriptorSchemaJsonKey).ValueOrDie()).Parse();

    if (rb->num_columns() < 3) {
        throw std::invalid_argument("Arrow record batch must have molecule_id, begin, and end.");
    }

    const auto molecule_id_column = std::dynamic_pointer_cast<arrow::UInt32Array>(rb->column(0));
    const auto begin_column = std::dynamic_pointer_cast<arrow::UInt32Array>(rb->column(1));
    const auto end_column = std::dynamic_pointer_cast<arrow::UInt32Array>(rb->column(2));

    if (molecule_id_column == nullptr || begin_column == nullptr || end_column == nullptr) {
        throw std::invalid_argument("Arrow bond identifier columns have wrong type.");
    }

    if (static_cast<std::size_t>(rb->num_columns() - 3) != column_names.size()) {
        throw std::invalid_argument("Arrow column count does not match schema.");
    }

    // Group rows by molecule_id
    std::vector<std::vector<std::size_t>> mol_row_indices(molecule_count);
    for (std::int64_t row = 0; row < rb->num_rows(); ++row) {
        const auto mol_id = molecule_id_column->Value(row);
        if (mol_id >= molecule_count) {
            throw std::invalid_argument("Arrow molecule_id out of range.");
        }
        mol_row_indices[mol_id].push_back(static_cast<std::size_t>(row));
    }

    // Build schema
    DescriptorSchemaBuilder schema_builder;
    for (const auto& name : column_names) {
        DescriptorDefinition def;
        def.name = name;
        def.value_kind = DescriptorValueKind::Float;
        def.group = "kallisto";
        def.source_name = "kallisto";
        def.source_type = "geometric";
        def.source_version = "kallisto-1.0.10";
        def.units = "Bohr";
        def.prerequisites = kDescriptorPrerequisiteCoordinates3D;
        schema_builder.Add(std::move(def));
    }
    auto schema = schema_builder.Build();

    auto batch = BondDescriptorBatch::Empty(schema);

    for (std::size_t mol = 0; mol < molecule_count; ++mol) {
        const auto& row_indices = mol_row_indices[mol];
        if (row_indices.empty()) {
            batch.Append(BondDescriptorSet::Empty(schema));
            continue;
        }

        // Extract bond endpoints
        std::vector<std::pair<std::uint32_t, std::uint32_t>> bond_endpoints;
        bond_endpoints.reserve(row_indices.size());
        for (const auto row : row_indices) {
            bond_endpoints.emplace_back(
                begin_column->Value(static_cast<std::int64_t>(row)),
                end_column->Value(static_cast<std::int64_t>(row)));
        }

        // Extract column values
        std::vector<std::vector<std::optional<double>>> columns(schema->Size());
        for (std::size_t col = 0; col < schema->Size(); ++col) {
            columns[col].reserve(row_indices.size());
            const auto arrow_col = std::dynamic_pointer_cast<arrow::DoubleArray>(
                rb->column(static_cast<int>(col + 3)));
            if (arrow_col == nullptr) {
                throw std::invalid_argument("Arrow descriptor column has wrong type.");
            }
            for (const auto row : row_indices) {
                if (arrow_col->IsNull(static_cast<std::int64_t>(row))) {
                    columns[col].push_back(std::nullopt);
                } else {
                    columns[col].push_back(arrow_col->Value(static_cast<std::int64_t>(row)));
                }
            }
        }

        batch.Append(BondDescriptorSet(schema, std::move(bond_endpoints), std::move(columns)));
    }

    return batch;
}

void WriteKallistoAtomIpc(const AtomDescriptorBatch& batch, const std::string& path) {
    const auto record_batch = AtomDescriptorBatchToArrow(batch);
    auto output = unwrap_arrow_result(arrow::io::FileOutputStream::Open(path));
    auto writer = unwrap_arrow_result(arrow::ipc::MakeFileWriter(output, record_batch->schema()));

    check_arrow_status(writer->WriteRecordBatch(*record_batch));
    check_arrow_status(writer->Close());
    check_arrow_status(output->Close());
}

AtomDescriptorBatch ReadKallistoAtomIpc(const std::string& path) {
    auto input = unwrap_arrow_result(arrow::io::ReadableFile::Open(path));
    auto reader = unwrap_arrow_result(arrow::ipc::RecordBatchFileReader::Open(input));
    if (reader->num_record_batches() != 1) {
        throw std::invalid_argument("Arrow IPC file must contain exactly one record batch.");
    }
    auto record_batch = unwrap_arrow_result(reader->ReadRecordBatch(0));
    check_arrow_status(input->Close());
    return AtomDescriptorBatchFromArrow(record_batch);
}

void WriteKallistoBondIpc(const BondDescriptorBatch& batch, const std::string& path) {
    const auto record_batch = BondDescriptorBatchToArrow(batch);
    auto output = unwrap_arrow_result(arrow::io::FileOutputStream::Open(path));
    auto writer = unwrap_arrow_result(arrow::ipc::MakeFileWriter(output, record_batch->schema()));

    check_arrow_status(writer->WriteRecordBatch(*record_batch));
    check_arrow_status(writer->Close());
    check_arrow_status(output->Close());
}

BondDescriptorBatch ReadKallistoBondIpc(const std::string& path) {
    auto input = unwrap_arrow_result(arrow::io::ReadableFile::Open(path));
    auto reader = unwrap_arrow_result(arrow::ipc::RecordBatchFileReader::Open(input));
    if (reader->num_record_batches() != 1) {
        throw std::invalid_argument("Arrow IPC file must contain exactly one record batch.");
    }
    auto record_batch = unwrap_arrow_result(reader->ReadRecordBatch(0));
    check_arrow_status(input->Close());
    return BondDescriptorBatchFromArrow(record_batch);
}

} // namespace OEFP
