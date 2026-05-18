#ifndef OEFP_DESCRIPTOR_H
#define OEFP_DESCRIPTOR_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace OEFP {

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

private:
    DescriptorSpec spec_;
    std::vector<std::string> string_keys_;
    std::vector<std::int64_t> integer_keys_;
    std::vector<double> float_keys_;
    std::vector<std::uint32_t> counts_;

    void ValidateStorage() const;
};

bool operator==(const DescriptorSet& lhs, const DescriptorSet& rhs);
bool operator!=(const DescriptorSet& lhs, const DescriptorSet& rhs);

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_H
