#include "oefp/descriptor_batch.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

std::size_t total_entry_count(const std::vector<::OEFP::DescriptorSet>& descriptors) {
    std::size_t entry_count = 0u;
    for (const auto& descriptor_set : descriptors) {
        entry_count += descriptor_set.Size();
    }
    return entry_count;
}

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

bool exposes_legacy_descriptor_storage(const DescriptorSet& descriptors) {
    try {
        static_cast<void>(descriptors.Spec());
        static_cast<void>(descriptors.ValueType());
        static_cast<void>(descriptors.Counts());
    } catch (const std::invalid_argument&) {
        return false;
    }
    return true;
}

} // namespace

DescriptorBatch::DescriptorBatch(DescriptorSpec spec)
    : spec_(std::move(spec)),
      has_spec_(true) {
}

DescriptorBatch DescriptorBatch::FromDescriptorSets(
    const std::vector<::OEFP::DescriptorSet>& descriptors) {
    if (descriptors.empty()) {
        return DescriptorBatch();
    }

    if (descriptors.front().SchemaPtr() != nullptr
        && !exposes_legacy_descriptor_storage(descriptors.front())) {
        DescriptorBatch batch;
        batch.InitializeColumns(descriptors.front().SchemaPtr());
        for (const auto& descriptor_set : descriptors) {
            batch.Append(descriptor_set);
        }
        return batch;
    }

    DescriptorBatch batch(descriptors.front().Spec());
    batch.ReserveRowsAndEntries(descriptors.size(), total_entry_count(descriptors));
    for (const auto& descriptor_set : descriptors) {
        batch.Append(descriptor_set);
    }
    return batch;
}

DescriptorBatch DescriptorBatch::Empty(std::shared_ptr<const DescriptorSchema> schema) {
    DescriptorBatch batch;
    batch.InitializeColumns(std::move(schema));
    return batch;
}

void DescriptorBatch::Append(const DescriptorSet& descriptors) {
    if (descriptors.SchemaPtr() != nullptr && !exposes_legacy_descriptor_storage(descriptors)) {
        AppendTypedRow(descriptors);
        return;
    }

    if (schema_ != nullptr) {
        throw std::invalid_argument("Cannot append legacy descriptor rows to a schema-backed batch.");
    }
    ValidateDescriptorSet(descriptors);

    if (!has_spec_) {
        spec_ = descriptors.Spec();
        has_spec_ = true;
    }

    AppendActiveKeys(descriptors);
    const auto& descriptor_counts = descriptors.Counts();
    counts_.insert(counts_.end(), descriptor_counts.begin(), descriptor_counts.end());
    row_offsets_.push_back(static_cast<std::uint64_t>(EntryCount()));
}

void DescriptorBatch::ReserveRowsAndEntries(std::size_t row_count, std::size_t entry_count) {
    row_offsets_.reserve(row_count + 1u);
    counts_.reserve(entry_count);

    switch (ValueType()) {
    case DescriptorValueType::Integer:
        integer_keys_.reserve(entry_count);
        break;
    case DescriptorValueType::Float:
        float_keys_.reserve(entry_count);
        break;
    case DescriptorValueType::String:
        string_keys_.reserve(entry_count);
        break;
    }
}

void DescriptorBatch::AppendActiveKeys(const DescriptorSet& descriptors) {
    switch (ValueType()) {
    case DescriptorValueType::Integer: {
        const auto& keys = descriptors.IntegerKeys();
        integer_keys_.insert(integer_keys_.end(), keys.begin(), keys.end());
        break;
    }
    case DescriptorValueType::Float: {
        const auto& keys = descriptors.FloatKeys();
        float_keys_.insert(float_keys_.end(), keys.begin(), keys.end());
        break;
    }
    case DescriptorValueType::String: {
        const auto& keys = descriptors.StringKeys();
        string_keys_.insert(string_keys_.end(), keys.begin(), keys.end());
        break;
    }
    }
}

void DescriptorBatch::AppendTypedRow(const DescriptorSet& descriptors) {
    if (has_spec_) {
        throw std::invalid_argument("Cannot append schema-backed descriptor rows to a legacy batch.");
    }
    if (schema_ == nullptr) {
        InitializeColumns(descriptors.SchemaPtr());
    }
    ValidateTypedDescriptorSet(descriptors);

    row_ids_.push_back(descriptors.RowId());
    const auto& values = descriptors.Values();
    for (std::size_t column_index = 0; column_index < columns_.size(); ++column_index) {
        auto& column = columns_[column_index];
        const auto has_value = values[column_index].has_value();
        column.validity.push_back(has_value ? 1u : 0u);
        switch (column.value_kind) {
        case DescriptorValueKind::Bool:
            column.bool_values.push_back(
                has_value && values[column_index]->AsBool() ? 1u : 0u);
            break;
        case DescriptorValueKind::Int:
            column.int_values.push_back(has_value ? values[column_index]->AsInt() : 0);
            break;
        case DescriptorValueKind::Float:
            column.float_values.push_back(has_value ? values[column_index]->AsFloat() : 0.0);
            break;
        case DescriptorValueKind::String:
            column.string_values.push_back(has_value ? values[column_index]->AsString() : "");
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
            throw std::invalid_argument("Descriptor batch column kind is not scalar.");
        }
    }
}

void DescriptorBatch::InitializeColumns(std::shared_ptr<const DescriptorSchema> schema) {
    if (schema == nullptr) {
        throw std::invalid_argument("Descriptor batch schema must not be null.");
    }
    schema_ = std::move(schema);
    columns_.clear();
    columns_.reserve(schema_->Size());
    for (const auto& definition : schema_->Definitions()) {
        if (!is_scalar_kind(definition.value_kind)) {
            throw std::invalid_argument("Descriptor batches currently support scalar columns only.");
        }
        DescriptorColumnBlock column;
        column.value_kind = definition.value_kind;
        columns_.push_back(std::move(column));
    }
}

const DescriptorSpec& DescriptorBatch::Spec() const {
    RequireLegacyStorage();
    return spec_;
}

DescriptorValueType DescriptorBatch::ValueType() const {
    RequireLegacyStorage();
    return spec_.value_type;
}

const DescriptorSchema& DescriptorBatch::Schema() const {
    if (schema_ == nullptr) {
        throw std::invalid_argument("Descriptor batch does not have a schema.");
    }
    return *schema_;
}

std::size_t DescriptorBatch::Size() const {
    if (schema_ != nullptr) {
        return row_ids_.size();
    }
    return row_offsets_.empty() ? 0u : row_offsets_.size() - 1u;
}

const std::vector<std::string>& DescriptorBatch::RowIds() const {
    if (schema_ == nullptr) {
        throw std::invalid_argument("Legacy descriptor batches do not expose row identifiers.");
    }
    return row_ids_;
}

std::vector<double> DescriptorBatch::FloatColumn(const std::string& name) const {
    return Column(name, DescriptorValueKind::Float).float_values;
}

std::vector<std::int64_t> DescriptorBatch::IntColumn(const std::string& name) const {
    return Column(name, DescriptorValueKind::Int).int_values;
}

std::vector<std::uint8_t> DescriptorBatch::BoolColumn(const std::string& name) const {
    return Column(name, DescriptorValueKind::Bool).bool_values;
}

std::vector<std::string> DescriptorBatch::StringColumn(const std::string& name) const {
    return Column(name, DescriptorValueKind::String).string_values;
}

std::vector<std::uint8_t> DescriptorBatch::ColumnValidity(const std::string& name) const {
    const auto index = Schema().IndexOf(name);
    return columns_[index].validity;
}

DescriptorBatch DescriptorBatch::Subset(const DescriptorSelection& selection) const {
    const auto indices = selection.Resolve(Schema());
    std::vector<std::string> names;
    names.reserve(indices.size());
    for (const auto index : indices) {
        names.push_back(schema_->Definition(index).name);
    }

    DescriptorBatch subset;
    subset.InitializeColumns(schema_->Project(names));
    subset.row_ids_ = row_ids_;
    for (std::size_t projected_index = 0; projected_index < indices.size(); ++projected_index) {
        subset.columns_[projected_index] = columns_[indices[projected_index]];
    }
    return subset;
}

DescriptorNumericMatrix DescriptorBatch::ToNumericMatrix(
    const DescriptorSelection& selection) const {
    const auto& schema = Schema();  // throws for legacy CSR batches
    const auto indices = selection.Resolve(schema);

    DescriptorNumericMatrix matrix;
    matrix.rows = Size();
    matrix.columns = indices.size();
    matrix.names.reserve(indices.size());

    // Guard against overflow in the matrix size product before allocating.
    const std::size_t entry_count =
        (matrix.columns != 0u
         && matrix.rows > std::numeric_limits<std::size_t>::max() / matrix.columns)
            ? throw std::invalid_argument("Matrix dimension product is too large to index.")
            : matrix.rows * matrix.columns;

    matrix.values.assign(entry_count, 0.0);
    matrix.validity.assign(entry_count, 0u);

    for (std::size_t column = 0u; column < indices.size(); ++column) {
        const auto& block = columns_[indices[column]];
        const auto& definition = schema.Definition(indices[column]);
        matrix.names.push_back(definition.name);

        // Reject non-numeric columns up front so an empty batch with a String column throws.
        if (block.value_kind != DescriptorValueKind::Float
            && block.value_kind != DescriptorValueKind::Int
            && block.value_kind != DescriptorValueKind::Bool) {
            throw std::invalid_argument(
                "Descriptor column '" + definition.name + "' is not a numeric scalar column.");
        }

        for (std::size_t row = 0u; row < matrix.rows; ++row) {
            const auto slot = row * matrix.columns + column;
            matrix.validity[slot] = block.validity[row];
            switch (block.value_kind) {
            case DescriptorValueKind::Float:
                matrix.values[slot] = block.float_values[row];
                break;
            case DescriptorValueKind::Int:
                matrix.values[slot] = static_cast<double>(block.int_values[row]);
                break;
            case DescriptorValueKind::Bool:
                matrix.values[slot] = block.bool_values[row] != 0u ? 1.0 : 0.0;
                break;
            default:
                // Unreachable: the allow-list check above caught it.
                throw std::invalid_argument(
                    "Descriptor column '" + definition.name + "' is not a numeric scalar column.");
            }
        }
    }

    return matrix;
}

std::size_t DescriptorBatch::EntryCount() const {
    RequireLegacyStorage();
    switch (ValueType()) {
    case DescriptorValueType::Integer:
        return integer_keys_.size();
    case DescriptorValueType::Float:
        return float_keys_.size();
    case DescriptorValueType::String:
        return string_keys_.size();
    }
    return 0u;
}

std::uint64_t DescriptorBatch::RowOffset(std::size_t row) const {
    RequireLegacyStorage();
    CheckOffsetIndex(row);
    return row_offsets_[row];
}

std::size_t DescriptorBatch::RowEntryCount(std::size_t row) const {
    RequireLegacyStorage();
    CheckRowIndex(row);
    return static_cast<std::size_t>(row_offsets_[row + 1u] - row_offsets_[row]);
}

const std::vector<std::string>& DescriptorBatch::StringKeys() const {
    RequireLegacyStorage();
    return string_keys_;
}

const std::vector<std::int64_t>& DescriptorBatch::IntegerKeys() const {
    RequireLegacyStorage();
    return integer_keys_;
}

const std::vector<double>& DescriptorBatch::FloatKeys() const {
    RequireLegacyStorage();
    return float_keys_;
}

const std::vector<std::uint32_t>& DescriptorBatch::Counts() const {
    RequireLegacyStorage();
    return counts_;
}

const std::vector<std::uint64_t>& DescriptorBatch::RowOffsets() const {
    RequireLegacyStorage();
    return row_offsets_;
}

const std::uint32_t* DescriptorBatch::CountData() const {
    RequireLegacyStorage();
    if (counts_.empty()) {
        return nullptr;
    }
    return counts_.data();
}

std::uint64_t DescriptorBatch::CountDataAddress() const {
    const auto* data = CountData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const std::uint64_t* DescriptorBatch::RowOffsetData() const {
    RequireLegacyStorage();
    if (row_offsets_.empty()) {
        return nullptr;
    }
    return row_offsets_.data();
}

std::uint64_t DescriptorBatch::RowOffsetDataAddress() const {
    const auto* data = RowOffsetData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const std::int64_t* DescriptorBatch::IntegerKeyData() const {
    RequireLegacyStorage();
    if (integer_keys_.empty()) {
        return nullptr;
    }
    return integer_keys_.data();
}

std::uint64_t DescriptorBatch::IntegerKeyDataAddress() const {
    const auto* data = IntegerKeyData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const double* DescriptorBatch::FloatKeyData() const {
    RequireLegacyStorage();
    if (float_keys_.empty()) {
        return nullptr;
    }
    return float_keys_.data();
}

std::uint64_t DescriptorBatch::FloatKeyDataAddress() const {
    const auto* data = FloatKeyData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

void DescriptorBatch::ValidateDescriptorSet(const DescriptorSet& descriptors) const {
    if (has_spec_ && descriptors.Spec() != spec_) {
        throw std::invalid_argument("Descriptor set spec does not match batch spec.");
    }
}

void DescriptorBatch::ValidateTypedDescriptorSet(const DescriptorSet& descriptors) const {
    const auto row_schema = descriptors.SchemaPtr();
    if (row_schema == nullptr) {
        throw std::invalid_argument("Descriptor set is not schema-backed.");
    }
    if (schema_ == nullptr || row_schema->SchemaId() != schema_->SchemaId()) {
        throw std::invalid_argument("Descriptor set schema does not match batch schema.");
    }
}

void DescriptorBatch::CheckRowIndex(std::size_t row) const {
    if (row >= Size()) {
        throw std::out_of_range("Descriptor batch row index is out of range.");
    }
}

void DescriptorBatch::CheckOffsetIndex(std::size_t row) const {
    if (row > Size()) {
        throw std::out_of_range("Descriptor batch row offset index is out of range.");
    }
}

void DescriptorBatch::RequireLegacyStorage() const {
    if (schema_ != nullptr) {
        throw std::invalid_argument("Schema-backed descriptor batches do not expose legacy CSR storage.");
    }
}

const DescriptorBatch::DescriptorColumnBlock& DescriptorBatch::Column(
    const std::string& name,
    DescriptorValueKind kind) const {
    const auto index = Schema().IndexOf(name);
    const auto& column = columns_[index];
    if (column.value_kind != kind) {
        throw std::invalid_argument("Descriptor column kind does not match requested type.");
    }
    return column;
}

bool operator==(const DescriptorBatch& lhs, const DescriptorBatch& rhs) {
    if (lhs.schema_ != nullptr || rhs.schema_ != nullptr) {
        if (lhs.schema_ == nullptr || rhs.schema_ == nullptr) {
            return false;
        }
        if (lhs.schema_->SchemaId() != rhs.schema_->SchemaId()
            || lhs.row_ids_ != rhs.row_ids_ || lhs.columns_.size() != rhs.columns_.size()) {
            return false;
        }
        for (std::size_t index = 0; index < lhs.columns_.size(); ++index) {
            const auto& lhs_column = lhs.columns_[index];
            const auto& rhs_column = rhs.columns_[index];
            if (lhs_column.value_kind != rhs_column.value_kind
                || lhs_column.validity != rhs_column.validity
                || lhs_column.int_values != rhs_column.int_values
                || lhs_column.float_values != rhs_column.float_values
                || lhs_column.bool_values != rhs_column.bool_values
                || lhs_column.string_values != rhs_column.string_values) {
                return false;
            }
        }
        return true;
    }
    return lhs.has_spec_ == rhs.has_spec_ && lhs.Spec() == rhs.Spec()
        && lhs.StringKeys() == rhs.StringKeys()
        && lhs.IntegerKeys() == rhs.IntegerKeys() && lhs.FloatKeys() == rhs.FloatKeys()
        && lhs.Counts() == rhs.Counts() && lhs.RowOffsets() == rhs.RowOffsets();
}

bool operator!=(const DescriptorBatch& lhs, const DescriptorBatch& rhs) {
    return !(lhs == rhs);
}

} // namespace OEFP
