#ifndef OEFP_DESCRIPTOR_H
#define OEFP_DESCRIPTOR_H

#include "oefp/descriptor_schema.h"
#include "oefp/descriptor_value.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace OEFP {

class DescriptorSelection;

enum class DescriptorValueType {
    Integer,
    Float,
    String,
};

enum class DescriptorComparisonMode {
    CountOverlap,
    ExactCount,
    Presence,
};

struct DescriptorSpec {
    DescriptorValueType value_type = DescriptorValueType::String;
    std::string source_name;
    std::string source_type;
    std::string source_version;
    std::string parameters;
};

bool operator==(const DescriptorSpec& lhs, const DescriptorSpec& rhs);
bool operator!=(const DescriptorSpec& lhs, const DescriptorSpec& rhs);

class DescriptorSet {
public:
    DescriptorSet() = default;
    DescriptorSet(
        DescriptorSpec spec,
        std::vector<std::string> keys,
        std::vector<std::uint32_t> counts);
    DescriptorSet(
        DescriptorSpec spec,
        std::vector<std::int64_t> keys,
        std::vector<std::uint32_t> counts);
    DescriptorSet(
        DescriptorSpec spec,
        std::vector<double> keys,
        std::vector<std::uint32_t> counts);
    DescriptorSet(
        std::shared_ptr<const DescriptorSchema> schema,
        std::vector<std::optional<DescriptorValue>> values,
        std::string row_id = "");

    static DescriptorSet FromStrings(
        DescriptorSpec spec,
        const std::vector<std::string>& keys);
    static DescriptorSet FromIntegers(
        DescriptorSpec spec,
        const std::vector<std::int64_t>& keys);
    static DescriptorSet FromFloats(
        DescriptorSpec spec,
        const std::vector<double>& keys);
    static DescriptorSet FromStringCounts(
        DescriptorSpec spec,
        const std::vector<std::string>& keys,
        const std::vector<std::uint32_t>& counts);
    static DescriptorSet FromIntegerCounts(
        DescriptorSpec spec,
        const std::vector<std::int64_t>& keys,
        const std::vector<std::uint32_t>& counts);
    static DescriptorSet FromFloatCounts(
        DescriptorSpec spec,
        const std::vector<double>& keys,
        const std::vector<std::uint32_t>& counts);

    const DescriptorSpec& Spec() const;
    DescriptorValueType ValueType() const;
    std::size_t Size() const;
    std::uint64_t TotalCount() const;

    const std::vector<std::string>& StringKeys() const;
    const std::vector<std::int64_t>& IntegerKeys() const;
    const std::vector<double>& FloatKeys() const;
    const std::vector<std::uint32_t>& Counts() const;

    const std::uint32_t* CountData() const;
    std::uint64_t CountDataAddress() const;
    const std::int64_t* IntegerKeyData() const;
    std::uint64_t IntegerKeyDataAddress() const;
    const double* FloatKeyData() const;
    std::uint64_t FloatKeyDataAddress() const;

    const DescriptorSchema& Schema() const;
    std::shared_ptr<const DescriptorSchema> SchemaPtr() const;
    const std::string& RowId() const;
    bool Has(const std::string& name) const;
    bool Has(std::size_t index) const;
    const DescriptorValue& Value(const std::string& name) const;
    const DescriptorValue& Value(std::size_t index) const;
    const std::vector<std::optional<DescriptorValue>>& Values() const;
    bool Bool(const std::string& name) const;
    std::int64_t Int(const std::string& name) const;
    double Float(const std::string& name) const;
    const std::string& String(const std::string& name) const;
    DescriptorSet Subset(const std::vector<std::string>& names) const;
    DescriptorSet Subset(const DescriptorSelection& selection) const;

private:
    DescriptorSpec spec_;
    std::vector<std::string> string_keys_;
    std::vector<std::int64_t> integer_keys_;
    std::vector<double> float_keys_;
    std::vector<std::uint32_t> counts_;
    std::shared_ptr<const DescriptorSchema> schema_;
    std::vector<std::optional<DescriptorValue>> values_;
    std::string row_id_;

    void ValidateStorage() const;
    void ValidateTypedStorage() const;
};

bool operator==(const DescriptorSet& lhs, const DescriptorSet& rhs);
bool operator!=(const DescriptorSet& lhs, const DescriptorSet& rhs);

class DescriptorSetBuilder {
public:
    explicit DescriptorSetBuilder(
        std::shared_ptr<const DescriptorSchema> schema,
        DescriptorPrerequisites available_prerequisites = kDescriptorPrerequisiteAll);

    void Set(const std::string& name, DescriptorValue value);
    DescriptorPrerequisites AvailablePrerequisites() const;
    DescriptorSet Build(std::string row_id = "") const;

private:
    std::shared_ptr<const DescriptorSchema> schema_;
    std::vector<std::optional<DescriptorValue>> values_;
    DescriptorPrerequisites available_prerequisites_ = kDescriptorPrerequisiteAll;
};

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_H
