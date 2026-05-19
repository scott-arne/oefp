#ifndef OEFP_DESCRIPTOR_VALUE_H
#define OEFP_DESCRIPTOR_VALUE_H

#include "oefp/descriptor_schema.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace OEFP {

using DescriptorScalarStorage = std::variant<
    bool,
    std::int64_t,
    double,
    std::string,
    std::vector<std::int64_t>,
    std::vector<double>
>;

class DescriptorValue {
public:
    static DescriptorValue Bool(bool value);
    static DescriptorValue Int(std::int64_t value);
    static DescriptorValue Float(double value);
    static DescriptorValue String(std::string value);
    static DescriptorValue IntVector(std::vector<std::int64_t> values);
    static DescriptorValue FloatVector(std::vector<double> values);
    static DescriptorValue IntMatrix(
        std::vector<std::uint64_t> shape,
        std::vector<std::int64_t> values);
    static DescriptorValue FloatMatrix(
        std::vector<std::uint64_t> shape,
        std::vector<double> values);

    DescriptorValueKind Kind() const;
    const std::vector<std::uint64_t>& Shape() const;

    bool AsBool() const;
    std::int64_t AsInt() const;
    double AsFloat() const;
    const std::string& AsString() const;
    const std::vector<std::int64_t>& AsIntVector() const;
    const std::vector<double>& AsFloatVector() const;

    bool operator==(const DescriptorValue& other) const;
    bool operator!=(const DescriptorValue& other) const;

private:
    DescriptorValueKind kind_ = DescriptorValueKind::Float;
    DescriptorScalarStorage storage_ = 0.0;
    std::vector<std::uint64_t> shape_;

    DescriptorValue(
        DescriptorValueKind kind,
        DescriptorScalarStorage storage,
        std::vector<std::uint64_t> shape = {});
};

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_VALUE_H
