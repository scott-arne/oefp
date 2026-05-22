#ifndef OEFP_DESCRIPTOR_SCHEMA_H
#define OEFP_DESCRIPTOR_SCHEMA_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace OEFP {

enum class DescriptorValueKind {
    Bool = 0,
    Int = 1,
    Float = 2,
    String = 3,
    FloatVector = 4,
    IntVector = 5,
    FloatMatrix = 6,
    IntMatrix = 7,
    CountedStringKeys = 8,
    CountedIntegerKeys = 9,
    CountedFloatKeys = 10,
    DenseBinaryFingerprint = 11,
    SparseBinaryFingerprint = 12,
    DenseCountFingerprint = 13,
    SparseCountFingerprint = 14,
};

struct DescriptorShape {
    std::vector<std::uint64_t> dimensions;
};

using DescriptorPrerequisites = std::uint32_t;

inline constexpr DescriptorPrerequisites kDescriptorPrerequisiteNone = 0u;
inline constexpr DescriptorPrerequisites kDescriptorPrerequisiteGraph = 1u << 0u;
inline constexpr DescriptorPrerequisites kDescriptorPrerequisiteCoordinates2D = 1u << 1u;
inline constexpr DescriptorPrerequisites kDescriptorPrerequisiteCoordinates3D = 1u << 2u;
inline constexpr DescriptorPrerequisites kDescriptorPrerequisiteAll =
    std::numeric_limits<DescriptorPrerequisites>::max();

/// \brief Return prerequisite bits that are not present in an input.
DescriptorPrerequisites MissingDescriptorPrerequisites(
    DescriptorPrerequisites required,
    DescriptorPrerequisites available);

/// \brief Return whether an input satisfies all required prerequisite bits.
bool DescriptorPrerequisitesSatisfied(
    DescriptorPrerequisites required,
    DescriptorPrerequisites available);

struct DescriptorDefinition {
    std::string name;
    DescriptorValueKind value_kind = DescriptorValueKind::Float;
    std::string group;
    std::string source_name;
    std::string source_type;
    std::string source_version;
    std::string parameters;
    std::string units;
    std::string description;
    DescriptorPrerequisites prerequisites = kDescriptorPrerequisiteNone;
    std::optional<DescriptorShape> shape;
};

class DescriptorSchema {
public:
    explicit DescriptorSchema(std::vector<DescriptorDefinition> definitions);

    std::size_t Size() const;
    const std::string& SchemaId() const;
    const DescriptorDefinition& Definition(std::size_t index) const;
    std::size_t IndexOf(const std::string& name) const;
    bool Contains(const std::string& name) const;
    std::vector<std::size_t> IndicesForGroup(const std::string& group) const;
    std::shared_ptr<const DescriptorSchema> Project(const std::vector<std::string>& names) const;
    const std::vector<DescriptorDefinition>& Definitions() const;

private:
    std::vector<DescriptorDefinition> definitions_;
    std::unordered_map<std::string, std::size_t> index_by_name_;
    std::unordered_map<std::string, std::vector<std::size_t>> indices_by_group_;
    std::string schema_id_;
};

class DescriptorSchemaBuilder {
public:
    void Add(DescriptorDefinition definition);
    std::shared_ptr<const DescriptorSchema> Build() const;

private:
    std::vector<DescriptorDefinition> definitions_;
};

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_SCHEMA_H
