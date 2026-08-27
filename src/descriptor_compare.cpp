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
void validate_numeric_metric(const Metric& metric, DescriptorMissingPolicy missing, std::size_t columns) {
    (void)missing;  // Tasks 7 and 8 add policy-dependent rejections here.
    switch (metric.Name()) {
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
        return;
    case MetricName::Minkowski:
        if (!metric.Weights().empty() && metric.Weights().size() != columns) {
            throw std::invalid_argument(
                "Minkowski weights length must match the selected column count.");
        }
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

/// \brief Total weight mass over all selected columns; the unweighted case is the column count.
double numeric_total_weight_mass(const Metric& metric, std::size_t columns) {
    if (metric.Name() != MetricName::Minkowski || metric.Weights().empty()) {
        return static_cast<double>(columns);
    }
    double mass = 0.0;
    for (const auto weight : metric.Weights()) {
        mass += weight;
    }
    return mass;
}

/// \brief True when this metric needs the per-dimension power accumulation.
///
/// Unweighted p = 1 and p = 2 read \c l1 and \c squared_l2 instead, which both avoids a
/// \c std::pow per dimension and keeps those two bit-identical to the fingerprint path.
bool numeric_needs_power(const Metric& metric) {
    if (metric.Name() != MetricName::Minkowski) {
        return false;
    }
    if (!metric.Weights().empty()) {
        return true;
    }
    return metric.P() != 1.0 && metric.P() != 2.0;
}

/// \brief Close out Minkowski from its accumulator, applying the rescale factor.
double evaluate_numeric_minkowski(const NumericStats& stats, const Metric& metric,
                                  double factor) {
    if (metric.Weights().empty()) {
        if (metric.P() == 1.0) {
            return stats.l1 * factor;
        }
        if (metric.P() == 2.0) {
            return std::sqrt(stats.squared_l2 * factor);
        }
    }
    return std::pow(stats.power_sum * factor, 1.0 / metric.P());
}

template <DescriptorMissingPolicy Policy, bool HasValidity, bool Powered>
double compare_numeric_rows(
    const double* a,
    const std::uint8_t* a_valid,
    const double* b,
    const std::uint8_t* b_valid,
    std::size_t columns,
    const Metric& metric,
    double total_weight_mass) {
    NumericStats stats;
    double used_weight_mass = 0.0;
    bool dropped_any = false;

    double power = 0.0;
    const double* weights = nullptr;
    if constexpr (Powered) {
        power = metric.P();
        weights = metric.Weights().empty() ? nullptr : metric.Weights().data();
    }

    // Unchanged from Task 6: a null row mask means that side is fully present, so a
    // masked matrix and an unmasked one may be compared against each other.
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
                    dropped_any = true;
                    continue;
                }
            }
        }

        accumulate_dimension(stats, a[column], b[column]);

        if constexpr (Policy == DescriptorMissingPolicy::Ignore) {
            // Weight mass, not dimension count: dropping a heavily weighted column removes
            // more from the sum than dropping a negligible one, and a count cannot tell
            // them apart. Unweighted metrics pass weight 1 per column, which reduces this
            // to the D / d ratio exactly.
            used_weight_mass += weights == nullptr ? 1.0 : weights[column];
        }

        if constexpr (Powered) {
            const auto term = std::pow(std::abs(a[column] - b[column]), power);
            stats.power_sum += weights == nullptr ? term : weights[column] * term;
        }
    }

    if (stats.dimensions == 0u) {
        return kNaN;
    }

    double factor = 1.0;
    if constexpr (Policy == DescriptorMissingPolicy::Ignore) {
        // Only rescale when a dimension was actually dropped. When nothing was dropped,
        // used_weight_mass equals total_weight_mass and factor is exactly 1.0, but computing
        // the ratio would produce 0/0 when every weight is zero. Under HasValidity == false,
        // dropped_any is trivially false, so the guard vanishes.
        if (dropped_any) {
            if (used_weight_mass == 0.0) {
                return kNaN;
            }
            factor = total_weight_mass / used_weight_mass;
        }
    }

    if (metric.Name() == MetricName::Minkowski) {
        return evaluate_numeric_minkowski(stats, metric, factor);
    }

    rescale_for_ignore(stats, metric, factor);
    return evaluate_numeric_metric(stats, metric);
}

using NumericRowComparator = double (*)(const double*, const std::uint8_t*,
                                        const double*, const std::uint8_t*,
                                        std::size_t, const Metric&, double);

/// \brief Resolve the policy and validity branches once, outside the pair loop.
NumericRowComparator select_numeric_comparator(DescriptorMissingPolicy missing,
                                               bool has_validity,
                                               bool needs_power) {
    if (missing == DescriptorMissingPolicy::Ignore) {
        if (has_validity) {
            return needs_power
                ? &compare_numeric_rows<DescriptorMissingPolicy::Ignore, true, true>
                : &compare_numeric_rows<DescriptorMissingPolicy::Ignore, true, false>;
        }
        return needs_power
            ? &compare_numeric_rows<DescriptorMissingPolicy::Ignore, false, true>
            : &compare_numeric_rows<DescriptorMissingPolicy::Ignore, false, false>;
    }
    if (has_validity) {
        return needs_power
            ? &compare_numeric_rows<DescriptorMissingPolicy::Propagate, true, true>
            : &compare_numeric_rows<DescriptorMissingPolicy::Propagate, true, false>;
    }
    return needs_power
        ? &compare_numeric_rows<DescriptorMissingPolicy::Propagate, false, true>
        : &compare_numeric_rows<DescriptorMissingPolicy::Propagate, false, false>;
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
    validate_numeric_metric(metric, missing, columns);
    const auto expected_length = condensed_size(rows);
    validate_output(output, output_length, expected_length);

    const auto total_weight_mass = numeric_total_weight_mass(metric, columns);
    const auto comparator =
        select_numeric_comparator(missing, validity != nullptr, numeric_needs_power(metric));
    detail::ParallelFor(0, expected_length, kernel.chunk_size, kernel.num_threads,
                        [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            std::size_t i = 0u;
            std::size_t j = 0u;
            condensed_pair_from_index(index, rows, i, j);
            output[index] = comparator(values + i * columns, validity_row(validity, i, columns),
                                       values + j * columns, validity_row(validity, j, columns),
                                       columns, metric, total_weight_mass);
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
    validate_numeric_metric(metric, missing, columns);
    const auto expected_length =
        checked_product(a_rows, b_rows, "CDist output size is too large.");
    validate_output(output, output_length, expected_length);

    // Either side may be unmasked. `compare_numeric_rows` reads a null row mask as
    // fully present, so a masked matrix and an unmasked one compare correctly.
    const auto has_validity = a_validity != nullptr || b_validity != nullptr;

    const auto total_weight_mass = numeric_total_weight_mass(metric, columns);
    const auto comparator = select_numeric_comparator(missing, has_validity, numeric_needs_power(metric));
    detail::ParallelFor(0, expected_length, kernel.chunk_size, kernel.num_threads,
                        [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const auto i = index / b_rows;
            const auto j = index % b_rows;
            output[index] =
                comparator(a_values + i * columns, validity_row(a_validity, i, columns),
                           b_values + j * columns, validity_row(b_validity, j, columns),
                           columns, metric, total_weight_mass);
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
