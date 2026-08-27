#include "oefp/descriptor_compare.h"

#include "compare_detail.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace OEFP {
namespace {

using detail::checked_product;
using detail::condensed_pair_from_index;
using detail::condensed_size;
using detail::evaluate_numeric_metric;
using detail::NumericStats;
using detail::validate_output;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// \brief Reject metrics that have no defined meaning over continuous descriptor columns.
void validate_numeric_metric(const Metric& metric, DescriptorMissingPolicy missing) {
    (void)missing;  // Tasks 7 and 8 add policy-dependent rejections here.
    switch (metric.Name()) {
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
        return;
    case MetricName::Haversine:
        throw std::invalid_argument(
            "Haversine is not valid for numeric descriptor comparison.");
    case MetricName::Jaccard:
    case MetricName::Matching:
    case MetricName::Dice:
    case MetricName::Kulsinski:
    case MetricName::RogersTanimoto:
    case MetricName::RussellRao:
    case MetricName::SokalMichener:
    case MetricName::SokalSneath:
    case MetricName::Tanimoto:
    case MetricName::Tversky:
        throw std::invalid_argument(
            "Boolean metrics are not valid for numeric descriptor comparison.");
    default:
        break;
    }
    throw std::invalid_argument("Metric is not valid for numeric descriptor comparison.");
}

/// \brief Accumulate one dimension, matching the fingerprint path's definitions exactly.
void accumulate_dimension(NumericStats& stats, double a_value, double b_value) {
    const auto difference = a_value - b_value;
    const auto abs_difference = std::abs(difference);

    ++stats.dimensions;
    stats.l1 += abs_difference;
    stats.squared_l2 += difference * difference;
    stats.max_abs = std::max(stats.max_abs, abs_difference);
    if (difference != 0.0) {
        stats.unequal += 1.0;
    }
    if (a_value != 0.0 || b_value != 0.0) {
        stats.canberra += abs_difference / (std::abs(a_value) + std::abs(b_value));
    }
    stats.sum_abs += std::abs(a_value) + std::abs(b_value);
}

/// \brief Rescale an \c Ignore accumulator so a dropped dimension does not shrink the result.
///
/// Only sum-like accumulators are scaled. Chebyshev is a maximum, and Hamming and BrayCurtis
/// already divide by the count actually used, so all three are dimension-count-free. Scaling
/// a field another metric also reads is safe because exactly one metric is evaluated per call.
void rescale_for_ignore(NumericStats& stats, const Metric& metric, double factor) {
    switch (metric.Name()) {
    case MetricName::Euclidean:
        stats.squared_l2 *= factor;
        break;
    case MetricName::Manhattan:
        stats.l1 *= factor;
        break;
    case MetricName::Canberra:
        stats.canberra *= factor;
        break;
    default:
        break;
    }
}

template <DescriptorMissingPolicy Policy, bool HasValidity>
double compare_numeric_rows(
    const double* a,
    const std::uint8_t* a_valid,
    const double* b,
    const std::uint8_t* b_valid,
    std::size_t columns,
    const Metric& metric) {
    NumericStats stats;

    // A null row mask means that side is fully present. `HasValidity` says only that at
    // least one side has a mask, because CDist may legitimately pair a fully present
    // matrix with a masked one. Both tests are loop-invariant, so they cost nothing.
    const bool a_all_present = a_valid == nullptr;
    const bool b_all_present = b_valid == nullptr;

    for (std::size_t column = 0u; column < columns; ++column) {
        if constexpr (HasValidity) {
            const auto present = (a_all_present || a_valid[column] != 0u)
                              && (b_all_present || b_valid[column] != 0u);
            if (!present) {
                if constexpr (Policy == DescriptorMissingPolicy::Propagate) {
                    return kNaN;
                } else {
                    continue;
                }
            }
        }
        accumulate_dimension(stats, a[column], b[column]);
    }

    if (stats.dimensions == 0u) {
        return kNaN;
    }

    if constexpr (Policy == DescriptorMissingPolicy::Ignore) {
        if (stats.dimensions != columns) {
            rescale_for_ignore(stats, metric,
                               static_cast<double>(columns)
                                   / static_cast<double>(stats.dimensions));
        }
    }

    return evaluate_numeric_metric(stats, metric);
}

using NumericRowComparator = double (*)(const double*, const std::uint8_t*,
                                        const double*, const std::uint8_t*,
                                        std::size_t, const Metric&);

/// \brief Resolve the policy and validity branches once, outside the pair loop.
NumericRowComparator select_numeric_comparator(DescriptorMissingPolicy missing,
                                               bool has_validity) {
    if (missing == DescriptorMissingPolicy::Ignore) {
        return has_validity
            ? &compare_numeric_rows<DescriptorMissingPolicy::Ignore, true>
            : &compare_numeric_rows<DescriptorMissingPolicy::Ignore, false>;
    }
    return has_validity
        ? &compare_numeric_rows<DescriptorMissingPolicy::Propagate, true>
        : &compare_numeric_rows<DescriptorMissingPolicy::Propagate, false>;
}

/// \brief Offset of \p row in a row-major buffer, or \c nullptr for an absent mask.
const std::uint8_t* validity_row(const std::uint8_t* validity, std::size_t row,
                                 std::size_t columns) {
    return validity == nullptr ? nullptr : validity + row * columns;
}

} // namespace

void PDistNumericInto(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel) {
    const auto expected_length = condensed_size(rows);
    validate_output(output, output_length, expected_length);
    validate_numeric_metric(metric, missing);

    const auto comparator = select_numeric_comparator(missing, validity != nullptr);
    detail::ParallelFor(0, expected_length, kernel.chunk_size, kernel.num_threads,
                        [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            std::size_t i = 0u;
            std::size_t j = 0u;
            condensed_pair_from_index(index, rows, i, j);
            output[index] = comparator(values + i * columns, validity_row(validity, i, columns),
                                       values + j * columns, validity_row(validity, j, columns),
                                       columns, metric);
        }
    });
}

std::vector<double> PDistNumeric(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    const BatchKernelOptions& kernel) {
    std::vector<double> output(condensed_size(rows), 0.0);
    PDistNumericInto(values, validity, rows, columns, metric, missing, output.data(),
                     output.size(), kernel);
    return output;
}

void CDistNumericInto(
    const double* a_values,
    const std::uint8_t* a_validity,
    std::size_t a_rows,
    const double* b_values,
    const std::uint8_t* b_validity,
    std::size_t b_rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel) {
    const auto expected_length =
        checked_product(a_rows, b_rows, "CDist output size is too large.");
    validate_output(output, output_length, expected_length);
    validate_numeric_metric(metric, missing);

    // Either side may be unmasked. `compare_numeric_rows` reads a null row mask as
    // fully present, so a masked matrix and an unmasked one compare correctly.
    const auto has_validity = a_validity != nullptr || b_validity != nullptr;

    const auto comparator = select_numeric_comparator(missing, has_validity);
    detail::ParallelFor(0, expected_length, kernel.chunk_size, kernel.num_threads,
                        [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const auto i = index / b_rows;
            const auto j = index % b_rows;
            output[index] =
                comparator(a_values + i * columns, validity_row(a_validity, i, columns),
                           b_values + j * columns, validity_row(b_validity, j, columns),
                           columns, metric);
        }
    });
}

std::vector<double> CDistNumeric(
    const double* a_values,
    const std::uint8_t* a_validity,
    std::size_t a_rows,
    const double* b_values,
    const std::uint8_t* b_validity,
    std::size_t b_rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    const BatchKernelOptions& kernel) {
    std::vector<double> output(
        checked_product(a_rows, b_rows, "CDist output size is too large."), 0.0);
    CDistNumericInto(a_values, a_validity, a_rows, b_values, b_validity, b_rows, columns, metric,
                     missing, output.data(), output.size(), kernel);
    return output;
}

} // namespace OEFP
