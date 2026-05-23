#include "oefp/descriptor_schema.h"

#include <array>
#include <charconv>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace OEFP {
namespace {

std::uint64_t stable_hash(const std::string& text) {
    std::uint64_t hash = 14695981039346656037ull;
    for (const auto ch : text) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 1099511628211ull;
    }
    return hash;
}

void append_uint(std::string& text, std::uint64_t value) {
    std::array<char, 32> buffer{};
    const auto [ptr, ec] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (ec != std::errc()) {
        throw std::runtime_error("Could not serialize descriptor schema integer.");
    }
    text.append(buffer.data(), ptr);
}

void append_field(std::string& text, const std::string& value) {
    append_uint(text, value.size());
    text.push_back(':');
    text.append(value);
}

const char* value_kind_token(DescriptorValueKind value_kind) {
    switch (value_kind) {
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

void validate_shape(const std::optional<DescriptorShape>& shape) {
    if (!shape.has_value()) {
        return;
    }
    if (shape->dimensions.empty()) {
        throw std::invalid_argument("Descriptor schema shape must not be empty.");
    }
    std::uint64_t size = 1;
    for (const auto dimension : shape->dimensions) {
        if (dimension == 0u) {
            throw std::invalid_argument("Descriptor schema shape dimensions must be positive.");
        }
        if (size > std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw std::overflow_error("Descriptor schema shape is too large.");
        }
        size *= dimension;
    }
}

std::string schema_id_for(const std::vector<DescriptorDefinition>& definitions) {
    std::string serialized = "oefp-descriptor-schema-v1\n";
    for (const auto& definition : definitions) {
        append_field(serialized, definition.name);
        serialized.push_back('|');
        serialized.append(value_kind_token(definition.value_kind));
        serialized.push_back('|');
        append_field(serialized, definition.group);
        serialized.push_back('|');
        append_field(serialized, definition.source_name);
        serialized.push_back('|');
        append_field(serialized, definition.source_type);
        serialized.push_back('|');
        append_field(serialized, definition.source_version);
        serialized.push_back('|');
        append_field(serialized, definition.parameters);
        serialized.push_back('|');
        append_field(serialized, definition.units);
        serialized.push_back('|');
        append_field(serialized, definition.description);
        serialized.push_back('|');
        append_uint(serialized, definition.prerequisites);
        serialized.push_back('|');
        if (definition.shape.has_value()) {
            append_uint(serialized, definition.shape->dimensions.size());
            for (const auto dimension : definition.shape->dimensions) {
                serialized.push_back(':');
                append_uint(serialized, dimension);
            }
        } else {
            serialized.push_back('-');
        }
        serialized.push_back('\n');
    }

    std::string id(16, '0');
    std::uint64_t hash = stable_hash(serialized);
    constexpr char hex[] = "0123456789abcdef";
    for (auto iter = id.rbegin(); iter != id.rend(); ++iter) {
        *iter = hex[hash & 0xfu];
        hash >>= 4u;
    }
    return id;
}

} // namespace

DescriptorPrerequisites MissingDescriptorPrerequisites(
    DescriptorPrerequisites required,
    DescriptorPrerequisites available) {
    return required & ~available;
}

bool DescriptorPrerequisitesSatisfied(
    DescriptorPrerequisites required,
    DescriptorPrerequisites available) {
    return MissingDescriptorPrerequisites(required, available) == kDescriptorPrerequisiteNone;
}

DescriptorSchema::DescriptorSchema(std::vector<DescriptorDefinition> definitions)
    : definitions_(std::move(definitions)) {
    index_by_name_.reserve(definitions_.size());
    for (std::size_t index = 0; index < definitions_.size(); ++index) {
        const auto& definition = definitions_[index];
        if (definition.name.empty()) {
            throw std::invalid_argument("Descriptor schema names must be non-empty.");
        }
        static_cast<void>(value_kind_token(definition.value_kind));
        validate_shape(definition.shape);
        const auto [_, inserted] = index_by_name_.emplace(definition.name, index);
        if (!inserted) {
            throw std::invalid_argument("Descriptor schema names must be unique.");
        }
        if (!definition.group.empty()) {
            indices_by_group_[definition.group].push_back(index);
        }
    }
    schema_id_ = schema_id_for(definitions_);
}

std::size_t DescriptorSchema::Size() const {
    return definitions_.size();
}

const std::string& DescriptorSchema::SchemaId() const {
    return schema_id_;
}

const DescriptorDefinition& DescriptorSchema::Definition(std::size_t index) const {
    if (index >= definitions_.size()) {
        throw std::out_of_range("Descriptor schema index is out of range.");
    }
    return definitions_[index];
}

std::size_t DescriptorSchema::IndexOf(const std::string& name) const {
    const auto iter = index_by_name_.find(name);
    if (iter == index_by_name_.end()) {
        throw std::out_of_range("Descriptor schema name was not found.");
    }
    return iter->second;
}

bool DescriptorSchema::Contains(const std::string& name) const {
    return index_by_name_.find(name) != index_by_name_.end();
}

std::vector<std::size_t> DescriptorSchema::IndicesForGroup(const std::string& group) const {
    const auto iter = indices_by_group_.find(group);
    if (iter == indices_by_group_.end()) {
        return {};
    }
    return iter->second;
}

std::shared_ptr<const DescriptorSchema> DescriptorSchema::Project(
    const std::vector<std::string>& names) const {
    std::vector<DescriptorDefinition> projected;
    projected.reserve(names.size());
    for (const auto& name : names) {
        projected.push_back(Definition(IndexOf(name)));
    }
    return std::make_shared<const DescriptorSchema>(std::move(projected));
}

const std::vector<DescriptorDefinition>& DescriptorSchema::Definitions() const {
    return definitions_;
}

void DescriptorSchemaBuilder::Add(DescriptorDefinition definition) {
    definitions_.push_back(std::move(definition));
}

std::shared_ptr<const DescriptorSchema> DescriptorSchemaBuilder::Build() const {
    return std::make_shared<const DescriptorSchema>(definitions_);
}

} // namespace OEFP
