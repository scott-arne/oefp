#include "oefp/descriptor_value.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

std::uint64_t shape_size(const std::vector<std::uint64_t>& shape) {
    std::uint64_t size = 1;
    for (const auto dimension : shape) {
        if (dimension == 0u) {
            throw std::invalid_argument("Descriptor matrix dimensions must be positive.");
        }
        if (size > std::numeric_limits<std::uint64_t>::max() / dimension) {
            throw std::overflow_error("Descriptor matrix shape is too large.");
        }
        size *= dimension;
    }
    return size;
}

template <typename Value>
const Value& get_value(
    const DescriptorScalarStorage& storage,
    DescriptorValueKind actual,
    DescriptorValueKind expected,
    const char* name) {
    if (actual != expected) {
        throw std::invalid_argument(std::string("Descriptor value is not ") + name + ".");
    }
    return std::get<Value>(storage);
}

template <typename Value>
void validate_matrix_shape(
    const std::vector<std::uint64_t>& shape,
    const std::vector<Value>& values) {
    if (shape.empty()) {
        throw std::invalid_argument("Descriptor matrix shape must not be empty.");
    }
    if (shape_size(shape) != values.size()) {
        throw std::invalid_argument("Descriptor matrix shape does not match value count.");
    }
}

} // namespace

DescriptorValue::DescriptorValue(
    DescriptorValueKind kind,
    DescriptorScalarStorage storage,
    std::vector<std::uint64_t> shape)
    : kind_(kind),
      storage_(std::move(storage)),
      shape_(std::move(shape)) {}

DescriptorValue DescriptorValue::Bool(bool value) {
    return DescriptorValue(DescriptorValueKind::Bool, value);
}

DescriptorValue DescriptorValue::Int(std::int64_t value) {
    return DescriptorValue(DescriptorValueKind::Int, value);
}

DescriptorValue DescriptorValue::Float(double value) {
    return DescriptorValue(DescriptorValueKind::Float, value);
}

DescriptorValue DescriptorValue::String(std::string value) {
    return DescriptorValue(DescriptorValueKind::String, std::move(value));
}

DescriptorValue DescriptorValue::IntVector(std::vector<std::int64_t> values) {
    return DescriptorValue(DescriptorValueKind::IntVector, std::move(values));
}

DescriptorValue DescriptorValue::FloatVector(std::vector<double> values) {
    return DescriptorValue(DescriptorValueKind::FloatVector, std::move(values));
}

DescriptorValue DescriptorValue::IntMatrix(
    std::vector<std::uint64_t> shape,
    std::vector<std::int64_t> values) {
    validate_matrix_shape(shape, values);
    return DescriptorValue(
        DescriptorValueKind::IntMatrix,
        std::move(values),
        std::move(shape));
}

DescriptorValue DescriptorValue::FloatMatrix(
    std::vector<std::uint64_t> shape,
    std::vector<double> values) {
    validate_matrix_shape(shape, values);
    return DescriptorValue(
        DescriptorValueKind::FloatMatrix,
        std::move(values),
        std::move(shape));
}

DescriptorValueKind DescriptorValue::Kind() const {
    return kind_;
}

const std::vector<std::uint64_t>& DescriptorValue::Shape() const {
    return shape_;
}

bool DescriptorValue::AsBool() const {
    return get_value<bool>(storage_, kind_, DescriptorValueKind::Bool, "bool");
}

std::int64_t DescriptorValue::AsInt() const {
    return get_value<std::int64_t>(storage_, kind_, DescriptorValueKind::Int, "int");
}

double DescriptorValue::AsFloat() const {
    return get_value<double>(storage_, kind_, DescriptorValueKind::Float, "float");
}

const std::string& DescriptorValue::AsString() const {
    return get_value<std::string>(storage_, kind_, DescriptorValueKind::String, "string");
}

const std::vector<std::int64_t>& DescriptorValue::AsIntVector() const {
    if (kind_ != DescriptorValueKind::IntVector && kind_ != DescriptorValueKind::IntMatrix) {
        throw std::invalid_argument("Descriptor value is not an int vector or matrix.");
    }
    return std::get<std::vector<std::int64_t>>(storage_);
}

const std::vector<double>& DescriptorValue::AsFloatVector() const {
    if (kind_ != DescriptorValueKind::FloatVector && kind_ != DescriptorValueKind::FloatMatrix) {
        throw std::invalid_argument("Descriptor value is not a float vector or matrix.");
    }
    return std::get<std::vector<double>>(storage_);
}

bool DescriptorValue::operator==(const DescriptorValue& other) const {
    return kind_ == other.kind_ && shape_ == other.shape_ && storage_ == other.storage_;
}

bool DescriptorValue::operator!=(const DescriptorValue& other) const {
    return !(*this == other);
}

} // namespace OEFP
