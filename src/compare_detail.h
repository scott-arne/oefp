#ifndef OEFP_SRC_COMPARE_DETAIL_H
#define OEFP_SRC_COMPARE_DETAIL_H

#include "oefp/metric.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace OEFP {
namespace detail {

/// \brief Accumulated per-pair quantities shared by the numeric metrics.
struct NumericStats {
    std::uint64_t dimensions = 0;
    double l1 = 0.0;
    double squared_l2 = 0.0;
    double max_abs = 0.0;
    double unequal = 0.0;
    double canberra = 0.0;
    double sum_abs = 0.0;
};

/// \brief Multiply two sizes, throwing \p label when the product would overflow.
inline std::size_t checked_product(std::size_t a, std::size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::invalid_argument(label);
    }
    return a * b;
}

/// \brief Number of entries in the condensed pairwise output for \p n rows.
inline std::size_t condensed_size(std::size_t n) {
    if (n < 2) {
        return 0;
    }
    auto left = n;
    auto right = n - 1;
    if (left % 2 == 0) {
        left /= 2;
    } else {
        right /= 2;
    }
    return checked_product(left, right, "Pairwise output size is too large.");
}

/// \brief Recover the row pair \p i, \p j that produced condensed entry \p index.
inline void condensed_pair_from_index(std::size_t index, std::size_t n,
                                      std::size_t& i, std::size_t& j) {
    const auto n_double = static_cast<double>(n);
    const auto index_double = static_cast<double>(index);
    const auto row = n_double - 2.0
        - std::floor(
            std::sqrt(-8.0 * index_double + 4.0 * n_double * (n_double - 1.0) - 7.0) / 2.0 - 0.5);
    i = static_cast<std::size_t>(row);
    j = index + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
}

/// \brief Divide, returning zero rather than NaN when \p denominator is zero.
inline double zero_safe_divide(double numerator, double denominator) {
    if (denominator == 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}

/// \brief Narrow a dimensionality to \c std::size_t, throwing when it does not fit.
inline std::size_t checked_dimension_size(std::uint64_t dimensions) {
    if (dimensions > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("Fingerprint dimensionality is too large for this metric.");
    }
    return static_cast<std::size_t>(dimensions);
}

/// \brief Require a metric parameter vector to match the compared dimensionality.
inline void validate_parameter_size(std::size_t actual, std::uint64_t expected,
                                    const char* label) {
    if (actual != checked_dimension_size(expected)) {
        throw std::invalid_argument(label);
    }
}

/// \brief Close out a numeric metric from its accumulated statistics.
inline double evaluate_numeric_metric(const NumericStats& stats, const Metric& metric) {
    switch (metric.Name()) {
    case MetricName::Euclidean:
        return std::sqrt(stats.squared_l2);
    case MetricName::Manhattan:
        return stats.l1;
    case MetricName::Chebyshev:
        return stats.max_abs;
    case MetricName::Hamming:
        return zero_safe_divide(stats.unequal, static_cast<double>(stats.dimensions));
    case MetricName::Canberra:
        return stats.canberra;
    case MetricName::BrayCurtis:
        return zero_safe_divide(stats.l1, stats.sum_abs);
    default:
        break;
    }
    throw std::invalid_argument("Metric is not valid for numeric fingerprint comparison.");
}

/// \brief Validate a caller-supplied output buffer against its expected length.
inline void validate_output(double* output, std::size_t output_length,
                            std::size_t expected_length) {
    if (output_length != expected_length) {
        throw std::invalid_argument("Output length does not match requested comparison shape.");
    }
    if (expected_length != 0 && output == nullptr) {
        throw std::invalid_argument(
            "Output pointer cannot be null for non-empty comparison output.");
    }
}

/// \brief Reinterpret a caller-supplied integer address as an output pointer.
inline double* address_to_output(std::uint64_t output_address) {
    return reinterpret_cast<double*>(static_cast<std::uintptr_t>(output_address));
}

} // namespace detail
} // namespace OEFP

#endif // OEFP_SRC_COMPARE_DETAIL_H
