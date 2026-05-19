#include "oefp/descriptor.h"

#include "oefp/descriptor_selection.h"

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

struct CountedDescriptorView {
    DescriptorValueType value_type = DescriptorValueType::String;
    const DescriptorDefinition* definition = nullptr;
    const std::vector<std::string>* string_keys = nullptr;
    const std::vector<std::int64_t>* integer_keys = nullptr;
    const std::vector<std::uint32_t>* counts = nullptr;
};

std::optional<CountedDescriptorView> counted_descriptor_view(
    const std::shared_ptr<const DescriptorSchema>& schema,
    const std::vector<std::optional<DescriptorValue>>& values) {
    if (schema == nullptr || schema->Size() != 1u || values.size() != 1u
        || !values[0].has_value()) {
        return std::nullopt;
    }

    const auto& definition = schema->Definition(0);
    const auto& value = *values[0];
    if (definition.value_kind != value.Kind()) {
        return std::nullopt;
    }

    CountedDescriptorView view;
    view.definition = &definition;
    switch (value.Kind()) {
    case DescriptorValueKind::CountedStringKeys: {
        const auto& counted = value.CountedStringKeys();
        view.value_type = DescriptorValueType::String;
        view.string_keys = &counted.keys;
        view.counts = &counted.counts;
        return view;
    }
    case DescriptorValueKind::CountedIntegerKeys: {
        const auto& counted = value.CountedIntegerKeys();
        view.value_type = DescriptorValueType::Integer;
        view.integer_keys = &counted.keys;
        view.counts = &counted.counts;
        return view;
    }
    case DescriptorValueKind::Bool:
    case DescriptorValueKind::Int:
    case DescriptorValueKind::Float:
    case DescriptorValueKind::String:
    case DescriptorValueKind::FloatVector:
    case DescriptorValueKind::IntVector:
    case DescriptorValueKind::FloatMatrix:
    case DescriptorValueKind::IntMatrix:
    case DescriptorValueKind::CountedFloatKeys:
    case DescriptorValueKind::DenseBinaryFingerprint:
    case DescriptorValueKind::SparseBinaryFingerprint:
    case DescriptorValueKind::DenseCountFingerprint:
    case DescriptorValueKind::SparseCountFingerprint:
        return std::nullopt;
    }
    throw std::invalid_argument("Descriptor value kind is invalid.");
}

DescriptorSpec spec_from_counted_view(const CountedDescriptorView& view) {
    DescriptorSpec spec;
    spec.value_type = view.value_type;
    spec.source_name = view.definition->source_name;
    spec.source_type = view.definition->source_type;
    spec.source_version = view.definition->source_version;
    spec.parameters = view.definition->parameters;
    return spec;
}

template <typename Key>
void validate_key_counts(
    const std::vector<Key>& keys,
    const std::vector<std::uint32_t>& counts,
    const char* key_name) {
    if (keys.size() != counts.size()) {
        throw std::invalid_argument("Descriptor keys and counts must have the same length.");
    }
    for (std::size_t row = 0; row < keys.size(); ++row) {
        if (counts[row] == 0u) {
            throw std::invalid_argument("Descriptor counts must be positive.");
        }
        if (row > 0 && !(keys[row - 1] < keys[row])) {
            throw std::invalid_argument(std::string("Descriptor ") + key_name
                + " keys must be strictly increasing.");
        }
    }
}

void validate_float_keys(const std::vector<double>& keys) {
    for (const auto key : keys) {
        if (!std::isfinite(key)) {
            throw std::invalid_argument("Descriptor float keys must be finite.");
        }
    }
}

void validate_value_type(
    DescriptorValueType actual,
    DescriptorValueType expected,
    const char* key_name) {
    if (actual != expected) {
        throw std::invalid_argument(std::string("Descriptor ") + key_name
            + " constructor requires matching value_type.");
    }
}

void validate_descriptor_value_kind(
    DescriptorValueKind actual,
    DescriptorValueKind expected,
    const std::string& name) {
    if (actual != expected) {
        throw std::invalid_argument("Descriptor value kind does not match schema for " + name + ".");
    }
}

void validate_descriptor_value_shape(
    const DescriptorValue& value,
    const DescriptorDefinition& definition) {
    if (!definition.shape.has_value()) {
        return;
    }
    if (value.Shape() != definition.shape->dimensions) {
        throw std::invalid_argument("Descriptor value shape does not match schema for "
            + definition.name + ".");
    }
}

template <typename Key>
std::pair<std::vector<Key>, std::vector<std::uint32_t>> aggregate_keys(
    const std::vector<Key>& keys) {
    // The ordered map gives duplicate aggregation and canonical key order together.
    std::map<Key, std::uint32_t> counts_by_key;
    for (const auto& key : keys) {
        auto& count = counts_by_key[key];
        if (count == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Descriptor key count exceeds uint32 storage.");
        }
        ++count;
    }

    std::vector<Key> canonical_keys;
    std::vector<std::uint32_t> counts;
    canonical_keys.reserve(counts_by_key.size());
    counts.reserve(counts_by_key.size());
    for (const auto& [key, count] : counts_by_key) {
        canonical_keys.push_back(key);
        counts.push_back(count);
    }
    return {std::move(canonical_keys), std::move(counts)};
}

} // namespace

bool operator==(const DescriptorSpec& lhs, const DescriptorSpec& rhs) {
    return lhs.value_type == rhs.value_type && lhs.source_name == rhs.source_name
        && lhs.source_type == rhs.source_type && lhs.source_version == rhs.source_version
        && lhs.parameters == rhs.parameters;
}

bool operator!=(const DescriptorSpec& lhs, const DescriptorSpec& rhs) {
    return !(lhs == rhs);
}

DescriptorSet::DescriptorSet(
    DescriptorSpec spec,
    std::vector<std::string> keys,
    std::vector<std::uint32_t> counts)
    : spec_(std::move(spec)),
      string_keys_(std::move(keys)),
      counts_(std::move(counts)) {
    validate_value_type(spec_.value_type, DescriptorValueType::String, "string");
    ValidateStorage();
}

DescriptorSet::DescriptorSet(
    DescriptorSpec spec,
    std::vector<std::int64_t> keys,
    std::vector<std::uint32_t> counts)
    : spec_(std::move(spec)),
      integer_keys_(std::move(keys)),
      counts_(std::move(counts)) {
    validate_value_type(spec_.value_type, DescriptorValueType::Integer, "integer");
    ValidateStorage();
}

DescriptorSet::DescriptorSet(
    DescriptorSpec spec,
    std::vector<double> keys,
    std::vector<std::uint32_t> counts)
    : spec_(std::move(spec)),
      float_keys_(std::move(keys)),
      counts_(std::move(counts)) {
    validate_value_type(spec_.value_type, DescriptorValueType::Float, "float");
    ValidateStorage();
}

DescriptorSet::DescriptorSet(
    std::shared_ptr<const DescriptorSchema> schema,
    std::vector<std::optional<DescriptorValue>> values,
    std::string row_id)
    : schema_(std::move(schema)),
      values_(std::move(values)),
      row_id_(std::move(row_id)) {
    ValidateTypedStorage();
    if (const auto view = counted_descriptor_view(schema_, values_)) {
        spec_ = spec_from_counted_view(*view);
    }
}

DescriptorSet DescriptorSet::FromStrings(
    DescriptorSpec spec,
    const std::vector<std::string>& keys) {
    auto [canonical_keys, counts] = aggregate_keys(keys);
    return DescriptorSet(std::move(spec), std::move(canonical_keys), std::move(counts));
}

DescriptorSet DescriptorSet::FromIntegers(
    DescriptorSpec spec,
    const std::vector<std::int64_t>& keys) {
    auto [canonical_keys, counts] = aggregate_keys(keys);
    return DescriptorSet(std::move(spec), std::move(canonical_keys), std::move(counts));
}

DescriptorSet DescriptorSet::FromFloats(
    DescriptorSpec spec,
    const std::vector<double>& keys) {
    validate_float_keys(keys);
    auto [canonical_keys, counts] = aggregate_keys(keys);
    return DescriptorSet(std::move(spec), std::move(canonical_keys), std::move(counts));
}

DescriptorSet DescriptorSet::FromStringCounts(
    DescriptorSpec spec,
    const std::vector<std::string>& keys,
    const std::vector<std::uint32_t>& counts) {
    return DescriptorSet(std::move(spec), keys, counts);
}

DescriptorSet DescriptorSet::FromIntegerCounts(
    DescriptorSpec spec,
    const std::vector<std::int64_t>& keys,
    const std::vector<std::uint32_t>& counts) {
    return DescriptorSet(std::move(spec), keys, counts);
}

DescriptorSet DescriptorSet::FromFloatCounts(
    DescriptorSpec spec,
    const std::vector<double>& keys,
    const std::vector<std::uint32_t>& counts) {
    return DescriptorSet(std::move(spec), keys, counts);
}

const DescriptorSpec& DescriptorSet::Spec() const {
    if (schema_ != nullptr) {
        if (counted_descriptor_view(schema_, values_)) {
            return spec_;
        }
        throw std::invalid_argument("Schema-backed descriptor rows do not expose DescriptorSpec.");
    }
    return spec_;
}

DescriptorValueType DescriptorSet::ValueType() const {
    if (schema_ != nullptr) {
        if (const auto view = counted_descriptor_view(schema_, values_)) {
            return view->value_type;
        }
        throw std::invalid_argument("Schema-backed descriptor rows do not expose DescriptorValueType.");
    }
    return spec_.value_type;
}

std::size_t DescriptorSet::Size() const {
    if (schema_ != nullptr) {
        if (const auto view = counted_descriptor_view(schema_, values_)) {
            return view->counts->size();
        }
        return values_.size();
    }
    switch (spec_.value_type) {
    case DescriptorValueType::Integer:
        return integer_keys_.size();
    case DescriptorValueType::Float:
        return float_keys_.size();
    case DescriptorValueType::String:
        return string_keys_.size();
    }
    return 0;
}

std::uint64_t DescriptorSet::TotalCount() const {
    if (schema_ != nullptr) {
        if (const auto view = counted_descriptor_view(schema_, values_)) {
            std::uint64_t total = 0;
            for (const auto count : *view->counts) {
                total += count;
            }
            return total;
        }
        throw std::invalid_argument("Schema-backed descriptor rows do not expose key counts.");
    }
    std::uint64_t total = 0;
    for (const auto count : counts_) {
        total += count;
    }
    return total;
}

const std::vector<std::string>& DescriptorSet::StringKeys() const {
    if (schema_ != nullptr) {
        if (const auto view = counted_descriptor_view(schema_, values_)) {
            if (view->string_keys != nullptr) {
                return *view->string_keys;
            }
        }
        throw std::invalid_argument("Schema-backed descriptor rows do not expose string keys.");
    }
    return string_keys_;
}

const std::vector<std::int64_t>& DescriptorSet::IntegerKeys() const {
    if (schema_ != nullptr) {
        if (const auto view = counted_descriptor_view(schema_, values_)) {
            if (view->integer_keys != nullptr) {
                return *view->integer_keys;
            }
        }
        throw std::invalid_argument("Schema-backed descriptor rows do not expose integer keys.");
    }
    return integer_keys_;
}

const std::vector<double>& DescriptorSet::FloatKeys() const {
    if (schema_ != nullptr) {
        throw std::invalid_argument("Schema-backed descriptor rows do not expose float keys.");
    }
    return float_keys_;
}

const std::vector<std::uint32_t>& DescriptorSet::Counts() const {
    if (schema_ != nullptr) {
        if (const auto view = counted_descriptor_view(schema_, values_)) {
            return *view->counts;
        }
        throw std::invalid_argument("Schema-backed descriptor rows do not expose key counts.");
    }
    return counts_;
}

const std::uint32_t* DescriptorSet::CountData() const {
    const auto& counts = Counts();
    if (counts.empty()) {
        return nullptr;
    }
    return counts.data();
}

std::uint64_t DescriptorSet::CountDataAddress() const {
    const auto* data = CountData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const std::int64_t* DescriptorSet::IntegerKeyData() const {
    const auto& keys = IntegerKeys();
    if (keys.empty()) {
        return nullptr;
    }
    return keys.data();
}

std::uint64_t DescriptorSet::IntegerKeyDataAddress() const {
    const auto* data = IntegerKeyData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const double* DescriptorSet::FloatKeyData() const {
    if (schema_ != nullptr) {
        throw std::invalid_argument("Schema-backed descriptor rows do not expose float keys.");
    }
    if (float_keys_.empty()) {
        return nullptr;
    }
    return float_keys_.data();
}

std::uint64_t DescriptorSet::FloatKeyDataAddress() const {
    const auto* data = FloatKeyData();
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

const DescriptorSchema& DescriptorSet::Schema() const {
    if (schema_ == nullptr) {
        throw std::invalid_argument("Descriptor set does not have a schema.");
    }
    return *schema_;
}

std::shared_ptr<const DescriptorSchema> DescriptorSet::SchemaPtr() const {
    return schema_;
}

const std::string& DescriptorSet::RowId() const {
    return row_id_;
}

bool DescriptorSet::Has(const std::string& name) const {
    if (schema_ == nullptr || !schema_->Contains(name)) {
        return false;
    }
    return Has(schema_->IndexOf(name));
}

bool DescriptorSet::Has(std::size_t index) const {
    if (index >= values_.size()) {
        return false;
    }
    return values_[index].has_value();
}

const DescriptorValue& DescriptorSet::Value(const std::string& name) const {
    return Value(Schema().IndexOf(name));
}

const DescriptorValue& DescriptorSet::Value(std::size_t index) const {
    if (index >= values_.size() || !values_[index].has_value()) {
        throw std::invalid_argument("Descriptor value is missing.");
    }
    return *values_[index];
}

const std::vector<std::optional<DescriptorValue>>& DescriptorSet::Values() const {
    return values_;
}

bool DescriptorSet::Bool(const std::string& name) const {
    return Value(name).AsBool();
}

std::int64_t DescriptorSet::Int(const std::string& name) const {
    return Value(name).AsInt();
}

double DescriptorSet::Float(const std::string& name) const {
    return Value(name).AsFloat();
}

const std::string& DescriptorSet::String(const std::string& name) const {
    return Value(name).AsString();
}

DescriptorSet DescriptorSet::Subset(const std::vector<std::string>& names) const {
    const auto projected_schema = Schema().Project(names);
    std::vector<std::optional<DescriptorValue>> projected_values;
    projected_values.reserve(names.size());
    for (const auto& name : names) {
        const auto index = schema_->IndexOf(name);
        projected_values.push_back(values_[index]);
    }
    return DescriptorSet(std::move(projected_schema), std::move(projected_values), row_id_);
}

DescriptorSet DescriptorSet::Subset(const DescriptorSelection& selection) const {
    const auto indices = selection.Resolve(Schema());
    std::vector<std::string> names;
    names.reserve(indices.size());
    for (const auto index : indices) {
        names.push_back(schema_->Definition(index).name);
    }
    return Subset(names);
}

void DescriptorSet::ValidateStorage() const {
    switch (spec_.value_type) {
    case DescriptorValueType::Integer:
        if (!string_keys_.empty() || !float_keys_.empty()) {
            throw std::invalid_argument("Integer descriptors cannot store non-integer keys.");
        }
        validate_key_counts(integer_keys_, counts_, "integer");
        break;
    case DescriptorValueType::Float:
        if (!string_keys_.empty() || !integer_keys_.empty()) {
            throw std::invalid_argument("Float descriptors cannot store non-float keys.");
        }
        validate_float_keys(float_keys_);
        validate_key_counts(float_keys_, counts_, "float");
        break;
    case DescriptorValueType::String:
        if (!integer_keys_.empty() || !float_keys_.empty()) {
            throw std::invalid_argument("String descriptors cannot store non-string keys.");
        }
        validate_key_counts(string_keys_, counts_, "string");
        break;
    }
}

void DescriptorSet::ValidateTypedStorage() const {
    if (schema_ == nullptr) {
        throw std::invalid_argument("Descriptor set schema must not be null.");
    }
    if (values_.size() != schema_->Size()) {
        throw std::invalid_argument("Descriptor set value count must match schema size.");
    }
    for (std::size_t index = 0; index < values_.size(); ++index) {
        if (!values_[index].has_value()) {
            continue;
        }
        const auto& definition = schema_->Definition(index);
        validate_descriptor_value_kind(values_[index]->Kind(), definition.value_kind, definition.name);
        validate_descriptor_value_shape(*values_[index], definition);
    }
}

bool operator==(const DescriptorSet& lhs, const DescriptorSet& rhs) {
    if (lhs.SchemaPtr() != nullptr || rhs.SchemaPtr() != nullptr) {
        const auto lhs_schema = lhs.SchemaPtr();
        const auto rhs_schema = rhs.SchemaPtr();
        if (lhs_schema == nullptr || rhs_schema == nullptr) {
            return false;
        }
        return lhs_schema->SchemaId() == rhs_schema->SchemaId()
            && lhs.RowId() == rhs.RowId()
            && lhs.Values() == rhs.Values();
    }
    return lhs.Spec() == rhs.Spec() && lhs.StringKeys() == rhs.StringKeys()
        && lhs.IntegerKeys() == rhs.IntegerKeys() && lhs.FloatKeys() == rhs.FloatKeys()
        && lhs.Counts() == rhs.Counts();
}

bool operator!=(const DescriptorSet& lhs, const DescriptorSet& rhs) {
    return !(lhs == rhs);
}

DescriptorSetBuilder::DescriptorSetBuilder(std::shared_ptr<const DescriptorSchema> schema)
    : schema_(std::move(schema)) {
    if (schema_ == nullptr) {
        throw std::invalid_argument("Descriptor set builder schema must not be null.");
    }
    values_.resize(schema_->Size());
}

void DescriptorSetBuilder::Set(const std::string& name, DescriptorValue value) {
    const auto index = schema_->IndexOf(name);
    const auto& definition = schema_->Definition(index);
    validate_descriptor_value_kind(value.Kind(), definition.value_kind, definition.name);
    validate_descriptor_value_shape(value, definition);
    values_[index] = std::move(value);
}

DescriptorSet DescriptorSetBuilder::Build(std::string row_id) const {
    return DescriptorSet(schema_, values_, std::move(row_id));
}

} // namespace OEFP
