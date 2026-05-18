#include "oefp/compare.h"

#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace OEFP {
namespace {

std::uint64_t count_bits(std::uint64_t word) {
#if defined(_MSC_VER)
    return static_cast<std::uint64_t>(__popcnt64(word));
#elif defined(__clang__) || defined(__GNUC__)
    return static_cast<std::uint64_t>(
        __builtin_popcountll(static_cast<unsigned long long>(word)));
#else
    std::uint64_t count = 0;
    while (word != 0) {
        word &= word - 1;
        ++count;
    }
    return count;
#endif
}

struct DenseCounts {
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    std::uint64_t intersection = 0;
    std::uint64_t xor_count = 0;
};

struct BooleanStats {
    std::uint64_t dimensions = 0;
    std::uint64_t true_true = 0;
    std::uint64_t true_false = 0;
    std::uint64_t false_true = 0;

    std::uint64_t Unequal() const {
        return true_false + false_true;
    }

    std::uint64_t Nonzero() const {
        return true_true + true_false + false_true;
    }
};

struct NumericStats {
    std::uint64_t dimensions = 0;
    double l1 = 0.0;
    double squared_l2 = 0.0;
    double max_abs = 0.0;
    double unequal = 0.0;
    double canberra = 0.0;
    double sum_abs = 0.0;
};

struct SparseCountStats {
    double a = 0.0;
    double b = 0.0;
    double overlap = 0.0;
    double union_count = 0.0;
    double dot = 0.0;
    double square_product = 0.0;
    double l1 = 0.0;
    double squared_l2 = 0.0;
    double max_abs = 0.0;
    double unequal = 0.0;
    double canberra = 0.0;
    std::uint64_t bool_intersection = 0;
};

struct Difference {
    std::uint64_t index = 0;
    double value = 0.0;
};

struct DescriptorMetricData {
    BooleanStats boolean_stats;
    NumericStats numeric_stats;
    std::vector<Difference> differences;
};

std::size_t checked_product(std::size_t a, std::size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::invalid_argument(label);
    }
    return a * b;
}

std::size_t condensed_size(std::size_t n) {
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

void condensed_pair_from_index(
    std::size_t index,
    std::size_t n,
    std::size_t& i,
    std::size_t& j) {
    const auto n_double = static_cast<double>(n);
    const auto index_double = static_cast<double>(index);
    const auto row = n_double - 2.0
        - std::floor(
            std::sqrt(-8.0 * index_double + 4.0 * n_double * (n_double - 1.0) - 7.0) / 2.0 - 0.5);
    i = static_cast<std::size_t>(row);
    j = index + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
}

double zero_safe_divide(double numerator, double denominator) {
    if (denominator == 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}

std::size_t checked_dimension_size(std::uint64_t dimensions) {
    if (dimensions > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("Fingerprint dimensionality is too large for this metric.");
    }
    return static_cast<std::size_t>(dimensions);
}

void validate_parameter_size(std::size_t actual, std::uint64_t expected, const char* label) {
    if (actual != checked_dimension_size(expected)) {
        throw std::invalid_argument(label);
    }
}

void validate_square_parameter_size(std::size_t actual, std::uint64_t dimensions, const char* label) {
    const auto dimension_size = checked_dimension_size(dimensions);
    const auto expected = checked_product(dimension_size, dimension_size, label);
    if (actual != expected) {
        throw std::invalid_argument(label);
    }
}

BooleanStats boolean_stats_from_dense_counts(const DenseCounts& counts, std::uint64_t dimensions) {
    BooleanStats stats;
    stats.dimensions = dimensions;
    stats.true_true = counts.intersection;
    stats.true_false = counts.a - counts.intersection;
    stats.false_true = counts.b - counts.intersection;
    return stats;
}

NumericStats numeric_stats_from_binary_counts(const DenseCounts& counts, std::uint64_t dimensions) {
    NumericStats stats;
    stats.dimensions = dimensions;
    stats.l1 = static_cast<double>(counts.xor_count);
    stats.squared_l2 = static_cast<double>(counts.xor_count);
    stats.max_abs = counts.xor_count == 0 ? 0.0 : 1.0;
    stats.unequal = static_cast<double>(counts.xor_count);
    stats.canberra = static_cast<double>(counts.xor_count);
    stats.sum_abs = static_cast<double>(counts.a + counts.b);
    return stats;
}

BooleanStats boolean_stats_from_sparse_count_stats(
    const SparseCountStats& stats,
    std::size_t a_size,
    std::size_t b_size,
    std::uint64_t dimensions) {
    BooleanStats output;
    output.dimensions = dimensions;
    output.true_true = stats.bool_intersection;
    output.true_false = static_cast<std::uint64_t>(a_size) - stats.bool_intersection;
    output.false_true = static_cast<std::uint64_t>(b_size) - stats.bool_intersection;
    return output;
}

NumericStats numeric_stats_from_sparse_count_stats(
    const SparseCountStats& stats,
    std::uint64_t dimensions) {
    NumericStats output;
    output.dimensions = dimensions;
    output.l1 = stats.l1;
    output.squared_l2 = stats.squared_l2;
    output.max_abs = stats.max_abs;
    output.unequal = stats.unequal;
    output.canberra = stats.canberra;
    output.sum_abs = stats.a + stats.b;
    return output;
}

double evaluate_boolean_metric(const BooleanStats& stats, const Metric& metric) {
    const auto n = static_cast<double>(stats.dimensions);
    const auto ntt = static_cast<double>(stats.true_true);
    const auto nne = static_cast<double>(stats.Unequal());
    const auto nnz = static_cast<double>(stats.Nonzero());

    switch (metric.Name()) {
    case MetricName::Jaccard:
        return zero_safe_divide(nne, nnz);
    case MetricName::Matching:
        return zero_safe_divide(nne, n);
    case MetricName::Dice:
        return nne / (ntt + nnz);
    case MetricName::Kulsinski:
        return (nne + n - ntt) / (nne + n);
    case MetricName::RogersTanimoto:
    case MetricName::SokalMichener:
        return 2.0 * nne / (n + nne);
    case MetricName::RussellRao:
        return (n - ntt) / n;
    case MetricName::SokalSneath:
        return nne / (nne + 0.5 * ntt);
    case MetricName::Tanimoto:
        return zero_safe_divide(ntt, nnz);
    case MetricName::Tversky:
        return zero_safe_divide(
            ntt,
            ntt
                + metric.Alpha() * static_cast<double>(stats.true_false)
                + metric.Beta() * static_cast<double>(stats.false_true));
    default:
        break;
    }

    throw std::invalid_argument("Metric is not valid for boolean fingerprint comparison.");
}

double evaluate_numeric_metric(const NumericStats& stats, const Metric& metric) {
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

double evaluate_minkowski(
    const std::vector<Difference>& differences,
    const NumericStats& stats,
    const Metric& metric) {
    if (metric.Weights().empty()) {
        if (metric.P() == 1.0) {
            return stats.l1;
        }
        if (metric.P() == 2.0) {
            return std::sqrt(stats.squared_l2);
        }

        double sum = 0.0;
        for (const auto& difference : differences) {
            sum += std::pow(std::abs(difference.value), metric.P());
        }
        return std::pow(sum, 1.0 / metric.P());
    }

    validate_parameter_size(
        metric.Weights().size(),
        stats.dimensions,
        "Minkowski weights length must match fingerprint dimensionality.");
    double sum = 0.0;
    for (const auto& difference : differences) {
        sum += metric.Weights()[checked_dimension_size(difference.index)]
            * std::pow(std::abs(difference.value), metric.P());
    }
    return std::pow(sum, 1.0 / metric.P());
}

double evaluate_standardized_euclidean(
    const std::vector<Difference>& differences,
    std::uint64_t dimensions,
    const Metric& metric) {
    validate_parameter_size(
        metric.Variances().size(),
        dimensions,
        "Standardized Euclidean variances length must match fingerprint dimensionality.");

    std::vector<double> dense_differences(metric.Variances().size(), 0.0);
    for (const auto& difference : differences) {
        dense_differences[checked_dimension_size(difference.index)] = difference.value;
    }

    double sum = 0.0;
    for (std::size_t index = 0; index < dense_differences.size(); ++index) {
        sum += dense_differences[index] * dense_differences[index] / metric.Variances()[index];
    }
    return std::sqrt(sum);
}

double evaluate_mahalanobis(
    const std::vector<Difference>& differences,
    std::uint64_t dimensions,
    const Metric& metric) {
    validate_square_parameter_size(
        metric.InverseCovariance().size(),
        dimensions,
        "Mahalanobis inverse covariance must be a square matrix matching fingerprint dimensionality.");

    const auto dimension_size = checked_dimension_size(dimensions);
    double sum = 0.0;
    for (const auto& left : differences) {
        const auto left_index = checked_dimension_size(left.index);
        for (const auto& right : differences) {
            const auto right_index = checked_dimension_size(right.index);
            sum += left.value
                * metric.InverseCovariance()[left_index * dimension_size + right_index]
                * right.value;
        }
    }
    return std::sqrt(sum);
}

double evaluate_haversine(double a_latitude, double a_longitude, double b_latitude, double b_longitude) {
    const auto latitude_delta = 0.5 * (a_latitude - b_latitude);
    const auto longitude_delta = 0.5 * (a_longitude - b_longitude);
    const auto term = std::sin(latitude_delta) * std::sin(latitude_delta)
        + std::cos(a_latitude) * std::cos(b_latitude)
            * std::sin(longitude_delta) * std::sin(longitude_delta);
    return 2.0 * std::asin(std::sqrt(term));
}

double evaluate_metric_with_differences(
    const std::vector<Difference>& differences,
    const NumericStats& stats,
    const Metric& metric) {
    switch (metric.Name()) {
    case MetricName::Minkowski:
        return evaluate_minkowski(differences, stats, metric);
    case MetricName::StandardizedEuclidean:
        return evaluate_standardized_euclidean(differences, stats.dimensions, metric);
    case MetricName::Mahalanobis:
        return evaluate_mahalanobis(differences, stats.dimensions, metric);
    default:
        break;
    }

    return evaluate_numeric_metric(stats, metric);
}

bool is_descriptor_boolean_metric(const Metric& metric) {
    switch (metric.Name()) {
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
        return true;
    default:
        break;
    }
    return false;
}

void reject_invalid_descriptor_metric(const Metric& metric) {
    switch (metric.Name()) {
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
    case MetricName::Haversine:
        throw std::invalid_argument("Metric is not valid for descriptor comparison.");
    default:
        break;
    }
}

void add_descriptor_dimension(
    DescriptorMetricData& data,
    double a_value,
    double b_value,
    std::uint64_t true_true,
    std::uint64_t true_false,
    std::uint64_t false_true,
    std::uint64_t boolean_dimensions) {
    const auto difference = a_value - b_value;
    const auto abs_difference = std::abs(difference);
    const auto dimension = data.numeric_stats.dimensions;

    data.boolean_stats.dimensions += boolean_dimensions;
    data.boolean_stats.true_true += true_true;
    data.boolean_stats.true_false += true_false;
    data.boolean_stats.false_true += false_true;

    ++data.numeric_stats.dimensions;
    data.numeric_stats.l1 += abs_difference;
    data.numeric_stats.squared_l2 += difference * difference;
    data.numeric_stats.max_abs = std::max(data.numeric_stats.max_abs, abs_difference);
    if (difference != 0.0) {
        data.numeric_stats.unequal += 1.0;
        data.differences.push_back({dimension, difference});
    }
    if (a_value != 0.0 || b_value != 0.0) {
        data.numeric_stats.canberra += abs_difference / (std::abs(a_value) + std::abs(b_value));
    }
    data.numeric_stats.sum_abs += std::abs(a_value) + std::abs(b_value);
}

void add_count_overlap_descriptor_dimension(
    DescriptorMetricData& data,
    std::uint32_t a_count,
    std::uint32_t b_count,
    DescriptorComparisonMode mode) {
    const auto a_value = mode == DescriptorComparisonMode::Presence && a_count != 0u ? 1u : a_count;
    const auto b_value = mode == DescriptorComparisonMode::Presence && b_count != 0u ? 1u : b_count;
    const auto overlap = std::min(a_value, b_value);
    add_descriptor_dimension(
        data,
        static_cast<double>(a_value),
        static_cast<double>(b_value),
        overlap,
        a_value - overlap,
        b_value - overlap,
        std::max(a_value, b_value));
}

void add_exact_count_descriptor_dimension(
    DescriptorMetricData& data,
    double a_value,
    double b_value) {
    const auto true_true = a_value != 0.0 && b_value != 0.0 ? 1u : 0u;
    const auto true_false = a_value != 0.0 && b_value == 0.0 ? 1u : 0u;
    const auto false_true = a_value == 0.0 && b_value != 0.0 ? 1u : 0u;
    add_descriptor_dimension(data, a_value, b_value, true_true, true_false, false_true, 1u);
}

template <typename Key>
DescriptorMetricData collect_descriptor_count_overlap_data(
    const std::vector<Key>& a_keys,
    const std::vector<std::uint32_t>& a_counts,
    std::size_t a_begin,
    std::size_t a_end,
    const std::vector<Key>& b_keys,
    const std::vector<std::uint32_t>& b_counts,
    std::size_t b_begin,
    std::size_t b_end,
    DescriptorComparisonMode mode) {
    DescriptorMetricData data;
    data.differences.reserve((a_end - a_begin) + (b_end - b_begin));

    std::size_t a_row = a_begin;
    std::size_t b_row = b_begin;
    while (a_row < a_end || b_row < b_end) {
        if (b_row == b_end || (a_row < a_end && a_keys[a_row] < b_keys[b_row])) {
            add_count_overlap_descriptor_dimension(data, a_counts[a_row], 0u, mode);
            ++a_row;
        } else if (a_row == a_end || b_keys[b_row] < a_keys[a_row]) {
            add_count_overlap_descriptor_dimension(data, 0u, b_counts[b_row], mode);
            ++b_row;
        } else {
            add_count_overlap_descriptor_dimension(data, a_counts[a_row], b_counts[b_row], mode);
            ++a_row;
            ++b_row;
        }
    }
    return data;
}

template <typename Key>
DescriptorMetricData collect_descriptor_count_overlap_data(
    const std::vector<Key>& a_keys,
    const std::vector<std::uint32_t>& a_counts,
    const std::vector<Key>& b_keys,
    const std::vector<std::uint32_t>& b_counts,
    DescriptorComparisonMode mode) {
    return collect_descriptor_count_overlap_data(
        a_keys,
        a_counts,
        0,
        a_keys.size(),
        b_keys,
        b_counts,
        0,
        b_keys.size(),
        mode);
}

template <typename Key>
DescriptorMetricData collect_descriptor_exact_count_data(
    const std::vector<Key>& a_keys,
    const std::vector<std::uint32_t>& a_counts,
    std::size_t a_begin,
    std::size_t a_end,
    const std::vector<Key>& b_keys,
    const std::vector<std::uint32_t>& b_counts,
    std::size_t b_begin,
    std::size_t b_end) {
    DescriptorMetricData data;
    data.differences.reserve((a_end - a_begin) + (b_end - b_begin));

    std::size_t a_row = a_begin;
    std::size_t b_row = b_begin;
    while (a_row < a_end || b_row < b_end) {
        if (b_row == b_end || (a_row < a_end && a_keys[a_row] < b_keys[b_row])) {
            add_exact_count_descriptor_dimension(data, 1.0, 0.0);
            ++a_row;
        } else if (a_row == a_end || b_keys[b_row] < a_keys[a_row]) {
            add_exact_count_descriptor_dimension(data, 0.0, 1.0);
            ++b_row;
        } else {
            if (a_counts[a_row] == b_counts[b_row]) {
                add_exact_count_descriptor_dimension(data, 1.0, 1.0);
            } else {
                add_exact_count_descriptor_dimension(data, 1.0, 0.0);
                add_exact_count_descriptor_dimension(data, 0.0, 1.0);
            }
            ++a_row;
            ++b_row;
        }
    }
    return data;
}

template <typename Key>
DescriptorMetricData collect_descriptor_exact_count_data(
    const std::vector<Key>& a_keys,
    const std::vector<std::uint32_t>& a_counts,
    const std::vector<Key>& b_keys,
    const std::vector<std::uint32_t>& b_counts) {
    return collect_descriptor_exact_count_data(
        a_keys,
        a_counts,
        0,
        a_keys.size(),
        b_keys,
        b_counts,
        0,
        b_keys.size());
}

template <typename Key>
DescriptorMetricData collect_descriptor_data(
    const std::vector<Key>& a_keys,
    const std::vector<std::uint32_t>& a_counts,
    std::size_t a_begin,
    std::size_t a_end,
    const std::vector<Key>& b_keys,
    const std::vector<std::uint32_t>& b_counts,
    std::size_t b_begin,
    std::size_t b_end,
    DescriptorComparisonMode mode) {
    if (mode == DescriptorComparisonMode::ExactCount) {
        return collect_descriptor_exact_count_data(
            a_keys,
            a_counts,
            a_begin,
            a_end,
            b_keys,
            b_counts,
            b_begin,
            b_end);
    }
    return collect_descriptor_count_overlap_data(
        a_keys,
        a_counts,
        a_begin,
        a_end,
        b_keys,
        b_counts,
        b_begin,
        b_end,
        mode);
}

template <typename Key>
DescriptorMetricData collect_descriptor_data(
    const std::vector<Key>& a_keys,
    const std::vector<std::uint32_t>& a_counts,
    const std::vector<Key>& b_keys,
    const std::vector<std::uint32_t>& b_counts,
    DescriptorComparisonMode mode) {
    if (mode == DescriptorComparisonMode::ExactCount) {
        return collect_descriptor_exact_count_data(a_keys, a_counts, b_keys, b_counts);
    }
    return collect_descriptor_count_overlap_data(a_keys, a_counts, b_keys, b_counts, mode);
}

DescriptorMetricData collect_descriptor_data(
    const DescriptorSet& a,
    const DescriptorSet& b,
    DescriptorComparisonMode mode) {
    switch (a.ValueType()) {
    case DescriptorValueType::Integer:
        return collect_descriptor_data(
            a.IntegerKeys(),
            a.Counts(),
            b.IntegerKeys(),
            b.Counts(),
            mode);
    case DescriptorValueType::Float:
        return collect_descriptor_data(
            a.FloatKeys(),
            a.Counts(),
            b.FloatKeys(),
            b.Counts(),
            mode);
    case DescriptorValueType::String:
        return collect_descriptor_data(
            a.StringKeys(),
            a.Counts(),
            b.StringKeys(),
            b.Counts(),
            mode);
    }
    throw std::invalid_argument("Unsupported descriptor value type.");
}

std::size_t checked_descriptor_offset(std::uint64_t offset) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::invalid_argument("Descriptor batch row offset is too large.");
    }
    return static_cast<std::size_t>(offset);
}

struct DescriptorRowBounds {
    std::size_t begin = 0;
    std::size_t end = 0;
};

DescriptorRowBounds descriptor_row_bounds(const DescriptorBatch& batch, std::size_t row) {
    return {
        checked_descriptor_offset(batch.RowOffset(row)),
        checked_descriptor_offset(batch.RowOffset(row + 1)),
    };
}

double evaluate_descriptor_metric(
    const DescriptorMetricData& data,
    const Metric& metric) {
    reject_invalid_descriptor_metric(metric);
    if (is_descriptor_boolean_metric(metric)) {
        return evaluate_boolean_metric(data.boolean_stats, metric);
    }
    if (metric.Name() == MetricName::Minkowski) {
        return evaluate_minkowski(data.differences, data.numeric_stats, metric);
    }
    return evaluate_numeric_metric(data.numeric_stats, metric);
}

template <typename Key>
double evaluate_descriptor_row_metric(
    const std::vector<Key>& a_keys,
    const std::vector<std::uint32_t>& a_counts,
    DescriptorRowBounds a_bounds,
    const std::vector<Key>& b_keys,
    const std::vector<std::uint32_t>& b_counts,
    DescriptorRowBounds b_bounds,
    const Metric& metric,
    DescriptorComparisonMode mode) {
    return evaluate_descriptor_metric(
        collect_descriptor_data(
            a_keys,
            a_counts,
            a_bounds.begin,
            a_bounds.end,
            b_keys,
            b_counts,
            b_bounds.begin,
            b_bounds.end,
            mode),
        metric);
}

double evaluate_descriptor_batch_row_metric(
    const DescriptorSet& query,
    const DescriptorBatch& library,
    std::size_t row,
    const Metric& metric,
    DescriptorComparisonMode mode) {
    const auto row_bounds = descriptor_row_bounds(library, row);
    switch (query.ValueType()) {
    case DescriptorValueType::Integer:
        return evaluate_descriptor_row_metric(
            query.IntegerKeys(),
            query.Counts(),
            {0, query.Size()},
            library.IntegerKeys(),
            library.Counts(),
            row_bounds,
            metric,
            mode);
    case DescriptorValueType::Float:
        return evaluate_descriptor_row_metric(
            query.FloatKeys(),
            query.Counts(),
            {0, query.Size()},
            library.FloatKeys(),
            library.Counts(),
            row_bounds,
            metric,
            mode);
    case DescriptorValueType::String:
        return evaluate_descriptor_row_metric(
            query.StringKeys(),
            query.Counts(),
            {0, query.Size()},
            library.StringKeys(),
            library.Counts(),
            row_bounds,
            metric,
            mode);
    }
    throw std::invalid_argument("Unsupported descriptor value type.");
}

double evaluate_descriptor_batch_row_metric(
    const DescriptorBatch& a,
    std::size_t row_a,
    const DescriptorBatch& b,
    std::size_t row_b,
    const Metric& metric,
    DescriptorComparisonMode mode) {
    const auto a_bounds = descriptor_row_bounds(a, row_a);
    const auto b_bounds = descriptor_row_bounds(b, row_b);
    switch (a.ValueType()) {
    case DescriptorValueType::Integer:
        return evaluate_descriptor_row_metric(
            a.IntegerKeys(),
            a.Counts(),
            a_bounds,
            b.IntegerKeys(),
            b.Counts(),
            b_bounds,
            metric,
            mode);
    case DescriptorValueType::Float:
        return evaluate_descriptor_row_metric(
            a.FloatKeys(),
            a.Counts(),
            a_bounds,
            b.FloatKeys(),
            b.Counts(),
            b_bounds,
            metric,
            mode);
    case DescriptorValueType::String:
        return evaluate_descriptor_row_metric(
            a.StringKeys(),
            a.Counts(),
            a_bounds,
            b.StringKeys(),
            b.Counts(),
            b_bounds,
            metric,
            mode);
    }
    throw std::invalid_argument("Unsupported descriptor value type.");
}

DenseCounts count_dense_pair(
    const std::uint64_t* a_words,
    const std::uint64_t* b_words,
    std::size_t word_count,
    std::uint64_t a_popcount,
    std::uint64_t b_popcount) {
    DenseCounts counts;
    counts.a = a_popcount;
    counts.b = b_popcount;

    for (std::size_t i = 0; i < word_count; ++i) {
        const auto aw = a_words[i];
        const auto bw = b_words[i];
        counts.intersection += count_bits(aw & bw);
        counts.xor_count += count_bits(aw ^ bw);
    }
    return counts;
}

bool dense_bit_at(const OEFP& fingerprint, std::uint64_t index) {
    const auto word_index = checked_dimension_size(index / 64u);
    const auto bit_index = static_cast<unsigned int>(index % 64u);
    return ((fingerprint.Words()[word_index] >> bit_index) & 1u) != 0u;
}

bool dense_row_bit_at(const std::uint64_t* words, std::uint64_t index) {
    const auto word_index = checked_dimension_size(index / 64u);
    const auto bit_index = static_cast<unsigned int>(index % 64u);
    return ((words[word_index] >> bit_index) & 1u) != 0u;
}

std::vector<Difference> collect_dense_binary_row_differences(
    const std::uint64_t* a_words,
    const std::uint64_t* b_words,
    std::uint64_t dimensions,
    std::uint64_t reserve_size) {
    std::vector<Difference> differences;
    differences.reserve(checked_dimension_size(reserve_size));
    for (std::uint64_t index = 0; index < dimensions; ++index) {
        const auto a_value = dense_row_bit_at(a_words, index) ? 1.0 : 0.0;
        const auto b_value = dense_row_bit_at(b_words, index) ? 1.0 : 0.0;
        if (a_value != b_value) {
            differences.push_back({index, a_value - b_value});
        }
    }
    return differences;
}

std::vector<Difference> collect_dense_binary_differences(const OEFP& a, const OEFP& b) {
    return collect_dense_binary_row_differences(
        a.WordData(),
        b.WordData(),
        a.SizeBits(),
        a.CountOnBits() + b.CountOnBits());
}

double dense_binary_value_at(const OEFP& fingerprint, std::uint64_t index) {
    return dense_bit_at(fingerprint, index) ? 1.0 : 0.0;
}

double dense_binary_row_value_at(const std::uint64_t* words, std::uint64_t index) {
    return dense_row_bit_at(words, index) ? 1.0 : 0.0;
}

SparseCountStats count_sparse_rows(
    const std::uint32_t* a_indices,
    const std::uint32_t* a_counts,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    const std::uint32_t* b_counts,
    std::size_t b_size) {
    SparseCountStats stats;
    double a_square = 0.0;
    double b_square = 0.0;
    std::size_t a_row = 0;
    std::size_t b_row = 0;

    while (a_row < a_size || b_row < b_size) {
        if (b_row == b_size || (a_row < a_size && a_indices[a_row] < b_indices[b_row])) {
            const auto a_count = static_cast<double>(a_counts[a_row]);
            stats.a += a_count;
            stats.union_count += a_count;
            stats.l1 += a_count;
            stats.squared_l2 += a_count * a_count;
            stats.max_abs = std::max(stats.max_abs, a_count);
            stats.unequal += 1.0;
            stats.canberra += 1.0;
            a_square += a_count * a_count;
            ++a_row;
        } else if (a_row == a_size || b_indices[b_row] < a_indices[a_row]) {
            const auto b_count = static_cast<double>(b_counts[b_row]);
            stats.b += b_count;
            stats.union_count += b_count;
            stats.l1 += b_count;
            stats.squared_l2 += b_count * b_count;
            stats.max_abs = std::max(stats.max_abs, b_count);
            stats.unequal += 1.0;
            stats.canberra += 1.0;
            b_square += b_count * b_count;
            ++b_row;
        } else {
            const auto a_count = static_cast<double>(a_counts[a_row]);
            const auto b_count = static_cast<double>(b_counts[b_row]);
            const auto difference = a_count > b_count ? a_count - b_count : b_count - a_count;
            stats.a += a_count;
            stats.b += b_count;
            stats.overlap += a_count < b_count ? a_count : b_count;
            stats.union_count += a_count > b_count ? a_count : b_count;
            stats.dot += a_count * b_count;
            stats.l1 += difference;
            stats.squared_l2 += difference * difference;
            stats.max_abs = std::max(stats.max_abs, difference);
            if (difference != 0.0) {
                stats.unequal += 1.0;
            }
            stats.canberra += difference / (a_count + b_count);
            ++stats.bool_intersection;
            a_square += a_count * a_count;
            b_square += b_count * b_count;
            ++a_row;
            ++b_row;
        }
    }

    stats.square_product = a_square * b_square;
    return stats;
}

SparseCountStats count_sparse_pair(const OEFPCount& a, const OEFPCount& b) {
    return count_sparse_rows(
        a.IndexData(),
        a.CountData(),
        a.NonzeroCount(),
        b.IndexData(),
        b.CountData(),
        b.NonzeroCount());
}

DenseCounts count_sparse_binary_rows(
    const std::uint32_t* a_indices,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    std::size_t b_size) {
    DenseCounts counts;
    counts.a = static_cast<std::uint64_t>(a_size);
    counts.b = static_cast<std::uint64_t>(b_size);

    std::size_t a_row = 0;
    std::size_t b_row = 0;
    while (a_row < a_size && b_row < b_size) {
        if (a_indices[a_row] < b_indices[b_row]) {
            ++a_row;
        } else if (b_indices[b_row] < a_indices[a_row]) {
            ++b_row;
        } else {
            ++counts.intersection;
            ++a_row;
            ++b_row;
        }
    }

    counts.xor_count = counts.a + counts.b - 2u * counts.intersection;
    return counts;
}

DenseCounts count_sparse_binary_pair(const OEFPSparse& a, const OEFPSparse& b) {
    return count_sparse_binary_rows(
        a.IndexData(),
        a.CountOnBits(),
        b.IndexData(),
        b.CountOnBits());
}

std::vector<Difference> collect_sparse_binary_row_differences(
    const std::uint32_t* a_indices,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    std::size_t b_size);

std::vector<Difference> collect_sparse_count_row_differences(
    const std::uint32_t* a_indices,
    const std::uint32_t* a_counts,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    const std::uint32_t* b_counts,
    std::size_t b_size);

std::vector<Difference> collect_sparse_binary_differences(const OEFPSparse& a, const OEFPSparse& b) {
    return collect_sparse_binary_row_differences(
        a.IndexData(),
        a.CountOnBits(),
        b.IndexData(),
        b.CountOnBits());
}

std::vector<Difference> collect_sparse_binary_row_differences(
    const std::uint32_t* a_indices,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    std::size_t b_size) {
    std::vector<Difference> differences;
    differences.reserve(a_size + b_size);
    std::size_t a_row = 0;
    std::size_t b_row = 0;
    while (a_row < a_size || b_row < b_size) {
        if (b_row == b_size || (a_row < a_size && a_indices[a_row] < b_indices[b_row])) {
            differences.push_back({a_indices[a_row], 1.0});
            ++a_row;
        } else if (a_row == a_size || b_indices[b_row] < a_indices[a_row]) {
            differences.push_back({b_indices[b_row], -1.0});
            ++b_row;
        } else {
            ++a_row;
            ++b_row;
        }
    }
    return differences;
}

std::vector<Difference> collect_sparse_count_differences(const OEFPCount& a, const OEFPCount& b) {
    return collect_sparse_count_row_differences(
        a.IndexData(),
        a.CountData(),
        a.NonzeroCount(),
        b.IndexData(),
        b.CountData(),
        b.NonzeroCount());
}

std::vector<Difference> collect_sparse_count_row_differences(
    const std::uint32_t* a_indices,
    const std::uint32_t* a_counts,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    const std::uint32_t* b_counts,
    std::size_t b_size) {
    std::vector<Difference> differences;
    differences.reserve(a_size + b_size);
    std::size_t a_row = 0;
    std::size_t b_row = 0;
    while (a_row < a_size || b_row < b_size) {
        if (b_row == b_size || (a_row < a_size && a_indices[a_row] < b_indices[b_row])) {
            differences.push_back({a_indices[a_row], static_cast<double>(a_counts[a_row])});
            ++a_row;
        } else if (a_row == a_size || b_indices[b_row] < a_indices[a_row]) {
            differences.push_back({b_indices[b_row], -static_cast<double>(b_counts[b_row])});
            ++b_row;
        } else {
            const auto difference = static_cast<double>(a_counts[a_row]) - static_cast<double>(b_counts[b_row]);
            if (difference != 0.0) {
                differences.push_back({a_indices[a_row], difference});
            }
            ++a_row;
            ++b_row;
        }
    }
    return differences;
}

double sparse_binary_value_at(const OEFPSparse& fingerprint, std::uint64_t index) {
    const auto& indices = fingerprint.Indices();
    return std::binary_search(indices.begin(), indices.end(), static_cast<std::uint32_t>(index)) ? 1.0 : 0.0;
}

double sparse_binary_row_value_at(
    const std::uint32_t* indices,
    std::size_t size,
    std::uint64_t index) {
    return std::binary_search(indices, indices + size, static_cast<std::uint32_t>(index)) ? 1.0 : 0.0;
}

double sparse_count_value_at(const OEFPCount& fingerprint, std::uint64_t index) {
    const auto& indices = fingerprint.Indices();
    const auto found = std::lower_bound(indices.begin(), indices.end(), static_cast<std::uint32_t>(index));
    if (found == indices.end() || *found != index) {
        return 0.0;
    }
    return static_cast<double>(fingerprint.Count(static_cast<std::size_t>(found - indices.begin())));
}

double sparse_count_row_value_at(
    const std::uint32_t* indices,
    const std::uint32_t* counts,
    std::size_t size,
    std::uint64_t index) {
    const auto found = std::lower_bound(indices, indices + size, static_cast<std::uint32_t>(index));
    if (found == indices + size || *found != index) {
        return 0.0;
    }
    return static_cast<double>(counts[found - indices]);
}

double evaluate_dense_binary_row_metric(
    const std::uint64_t* a_words,
    const std::uint64_t* b_words,
    std::size_t word_count,
    std::uint64_t a_popcount,
    std::uint64_t b_popcount,
    std::uint64_t dimensions,
    const Metric& metric) {
    const auto counts = count_dense_pair(a_words, b_words, word_count, a_popcount, b_popcount);
    switch (metric.Name()) {
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
        return evaluate_boolean_metric(boolean_stats_from_dense_counts(counts, dimensions), metric);
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
        return evaluate_numeric_metric(numeric_stats_from_binary_counts(counts, dimensions), metric);
    case MetricName::Minkowski:
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
        return evaluate_metric_with_differences(
            collect_dense_binary_row_differences(
                a_words,
                b_words,
                dimensions,
                a_popcount + b_popcount),
            numeric_stats_from_binary_counts(counts, dimensions),
            metric);
    case MetricName::Haversine:
        if (dimensions != 2u) {
            throw std::invalid_argument("Haversine distance requires two-dimensional fingerprints.");
        }
        return evaluate_haversine(
            dense_binary_row_value_at(a_words, 0),
            dense_binary_row_value_at(a_words, 1),
            dense_binary_row_value_at(b_words, 0),
            dense_binary_row_value_at(b_words, 1));
    }

    throw std::invalid_argument("Unsupported fingerprint metric.");
}

double evaluate_sparse_binary_row_metric(
    const std::uint32_t* a_indices,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    std::size_t b_size,
    std::uint64_t dimensions,
    const Metric& metric) {
    const auto counts = count_sparse_binary_rows(a_indices, a_size, b_indices, b_size);
    switch (metric.Name()) {
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
        return evaluate_boolean_metric(boolean_stats_from_dense_counts(counts, dimensions), metric);
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
        return evaluate_numeric_metric(numeric_stats_from_binary_counts(counts, dimensions), metric);
    case MetricName::Minkowski:
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
        return evaluate_metric_with_differences(
            collect_sparse_binary_row_differences(a_indices, a_size, b_indices, b_size),
            numeric_stats_from_binary_counts(counts, dimensions),
            metric);
    case MetricName::Haversine:
        if (dimensions != 2u) {
            throw std::invalid_argument("Haversine distance requires two-dimensional fingerprints.");
        }
        return evaluate_haversine(
            sparse_binary_row_value_at(a_indices, a_size, 0),
            sparse_binary_row_value_at(a_indices, a_size, 1),
            sparse_binary_row_value_at(b_indices, b_size, 0),
            sparse_binary_row_value_at(b_indices, b_size, 1));
    }

    throw std::invalid_argument("Unsupported fingerprint metric.");
}

double evaluate_sparse_count_row_metric(
    const std::uint32_t* a_indices,
    const std::uint32_t* a_counts,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    const std::uint32_t* b_counts,
    std::size_t b_size,
    std::uint64_t dimensions,
    const Metric& metric) {
    const auto stats = count_sparse_rows(a_indices, a_counts, a_size, b_indices, b_counts, b_size);
    switch (metric.Name()) {
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
        return evaluate_boolean_metric(
            boolean_stats_from_sparse_count_stats(stats, a_size, b_size, dimensions),
            metric);
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
        return evaluate_numeric_metric(numeric_stats_from_sparse_count_stats(stats, dimensions), metric);
    case MetricName::Minkowski:
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
        return evaluate_metric_with_differences(
            collect_sparse_count_row_differences(a_indices, a_counts, a_size, b_indices, b_counts, b_size),
            numeric_stats_from_sparse_count_stats(stats, dimensions),
            metric);
    case MetricName::Haversine:
        if (dimensions != 2u) {
            throw std::invalid_argument("Haversine distance requires two-dimensional fingerprints.");
        }
        return evaluate_haversine(
            sparse_count_row_value_at(a_indices, a_counts, a_size, 0),
            sparse_count_row_value_at(a_indices, a_counts, a_size, 1),
            sparse_count_row_value_at(b_indices, b_counts, b_size, 0),
            sparse_count_row_value_at(b_indices, b_counts, b_size, 1));
    }

    throw std::invalid_argument("Unsupported fingerprint metric.");
}

double evaluate_dense_binary_metric(const OEFP& a, const OEFP& b, const DenseCounts& counts, const Metric& metric) {
    const auto dimensions = a.SizeBits();
    switch (metric.Name()) {
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
        return evaluate_boolean_metric(boolean_stats_from_dense_counts(counts, dimensions), metric);
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
        return evaluate_numeric_metric(numeric_stats_from_binary_counts(counts, dimensions), metric);
    case MetricName::Minkowski:
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
        return evaluate_metric_with_differences(
            collect_dense_binary_differences(a, b),
            numeric_stats_from_binary_counts(counts, dimensions),
            metric);
    case MetricName::Haversine:
        if (dimensions != 2u) {
            throw std::invalid_argument("Haversine distance requires two-dimensional fingerprints.");
        }
        return evaluate_haversine(
            dense_binary_value_at(a, 0),
            dense_binary_value_at(a, 1),
            dense_binary_value_at(b, 0),
            dense_binary_value_at(b, 1));
    }

    throw std::invalid_argument("Unsupported fingerprint metric.");
}

double evaluate_sparse_binary_metric(
    const OEFPSparse& a,
    const OEFPSparse& b,
    const DenseCounts& counts,
    const Metric& metric) {
    const auto dimensions = a.SizeBits();
    switch (metric.Name()) {
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
        return evaluate_boolean_metric(boolean_stats_from_dense_counts(counts, dimensions), metric);
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
        return evaluate_numeric_metric(numeric_stats_from_binary_counts(counts, dimensions), metric);
    case MetricName::Minkowski:
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
        return evaluate_metric_with_differences(
            collect_sparse_binary_differences(a, b),
            numeric_stats_from_binary_counts(counts, dimensions),
            metric);
    case MetricName::Haversine:
        if (dimensions != 2u) {
            throw std::invalid_argument("Haversine distance requires two-dimensional fingerprints.");
        }
        return evaluate_haversine(
            sparse_binary_value_at(a, 0),
            sparse_binary_value_at(a, 1),
            sparse_binary_value_at(b, 0),
            sparse_binary_value_at(b, 1));
    }

    throw std::invalid_argument("Unsupported fingerprint metric.");
}

double evaluate_sparse_count_metric(
    const OEFPCount& a,
    const OEFPCount& b,
    const SparseCountStats& stats,
    const Metric& metric) {
    const auto dimensions = a.SizeBits();
    switch (metric.Name()) {
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
        return evaluate_boolean_metric(
            boolean_stats_from_sparse_count_stats(stats, a.NonzeroCount(), b.NonzeroCount(), dimensions),
            metric);
    case MetricName::Euclidean:
    case MetricName::Manhattan:
    case MetricName::Chebyshev:
    case MetricName::Hamming:
    case MetricName::Canberra:
    case MetricName::BrayCurtis:
        return evaluate_numeric_metric(numeric_stats_from_sparse_count_stats(stats, dimensions), metric);
    case MetricName::Minkowski:
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
        return evaluate_metric_with_differences(
            collect_sparse_count_differences(a, b),
            numeric_stats_from_sparse_count_stats(stats, dimensions),
            metric);
    case MetricName::Haversine:
        if (dimensions != 2u) {
            throw std::invalid_argument("Haversine distance requires two-dimensional fingerprints.");
        }
        return evaluate_haversine(
            sparse_count_value_at(a, 0),
            sparse_count_value_at(a, 1),
            sparse_count_value_at(b, 0),
            sparse_count_value_at(b, 1));
    }

    throw std::invalid_argument("Unsupported fingerprint metric.");
}

void validate_output(double* output, std::size_t output_length, std::size_t expected_length) {
    if (output_length != expected_length) {
        throw std::invalid_argument("Output length does not match requested comparison shape.");
    }
    if (expected_length != 0 && output == nullptr) {
        throw std::invalid_argument("Output pointer cannot be null for non-empty comparison output.");
    }
}

void validate_fingerprint_batch_compatibility(const OEFP& query, const OEFPBatch& library) {
    if (library.Size() == 0) {
        return;
    }
    if (query.Spec() != library.Spec()) {
        throw std::invalid_argument("Fingerprint specification must match batch specification.");
    }
    if (query.WordCount() != library.WordsPerFingerprint()) {
        throw std::invalid_argument("Fingerprint word count must match batch row width.");
    }
}

void validate_batch_compatibility(const OEFPBatch& a, const OEFPBatch& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return;
    }
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Batch fingerprint specifications must match.");
    }
    if (a.WordsPerFingerprint() != b.WordsPerFingerprint()) {
        throw std::invalid_argument("Batch row widths must match.");
    }
}

void validate_count_fingerprint_batch_compatibility(
    const OEFPCount& query,
    const OEFPCountBatch& library) {
    if (library.Size() == 0) {
        return;
    }
    if (query.Spec() != library.Spec()) {
        throw std::invalid_argument("Count fingerprint specification must match batch specification.");
    }
}

void validate_sparse_fingerprint_batch_compatibility(
    const OEFPSparse& query,
    const OEFPSparseBatch& library) {
    if (library.Size() == 0) {
        return;
    }
    if (query.Spec() != library.Spec()) {
        throw std::invalid_argument("Sparse fingerprint specification must match batch specification.");
    }
}

void validate_count_batch_compatibility(const OEFPCountBatch& a, const OEFPCountBatch& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return;
    }
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Count batch fingerprint specifications must match.");
    }
}

void validate_sparse_batch_compatibility(const OEFPSparseBatch& a, const OEFPSparseBatch& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return;
    }
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Sparse batch fingerprint specifications must match.");
    }
}

void validate_descriptor_batch_compatibility(
    const DescriptorSet& query,
    const DescriptorBatch& library) {
    if (library.Size() == 0) {
        return;
    }
    if (query.Spec() != library.Spec()) {
        throw std::invalid_argument("Descriptor specification must match batch specification.");
    }
    if (query.ValueType() != library.ValueType()) {
        throw std::invalid_argument("Descriptor value type must match batch value type.");
    }
}

void validate_descriptor_batch_compatibility(const DescriptorBatch& a, const DescriptorBatch& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return;
    }
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Descriptor batch specifications must match.");
    }
    if (a.ValueType() != b.ValueType()) {
        throw std::invalid_argument("Descriptor batch value types must match.");
    }
}

double* address_to_output(std::uint64_t output_address) {
    return reinterpret_cast<double*>(static_cast<std::uintptr_t>(output_address));
}

} // namespace

double Compare(const OEFP& a, const OEFP& b, const Metric& metric) {
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Fingerprint specifications must match for comparison.");
    }

    if (a.WordCount() != b.WordCount()) {
        throw std::invalid_argument("Fingerprint word counts must match for comparison.");
    }

    const auto& a_words = a.Words();
    const auto& b_words = b.Words();
    const auto counts = count_dense_pair(
        a_words.data(),
        b_words.data(),
        a_words.size(),
        a.CountOnBits(),
        b.CountOnBits());
    return evaluate_dense_binary_metric(a, b, counts, metric);
}

double Compare(const OEFPCount& a, const OEFPCount& b, const Metric& metric) {
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Count fingerprint specifications must match for comparison.");
    }

    return evaluate_sparse_count_metric(a, b, count_sparse_pair(a, b), metric);
}

double Compare(const OEFPSparse& a, const OEFPSparse& b, const Metric& metric) {
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Sparse fingerprint specifications must match for comparison.");
    }

    return evaluate_sparse_binary_metric(a, b, count_sparse_binary_pair(a, b), metric);
}

double Compare(
    const DescriptorSet& a,
    const DescriptorSet& b,
    const Metric& metric,
    DescriptorComparisonMode mode) {
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Descriptor specifications must match for comparison.");
    }
    if (a.ValueType() != b.ValueType()) {
        throw std::invalid_argument("Descriptor value types must match for comparison.");
    }

    return evaluate_descriptor_metric(collect_descriptor_data(a, b, mode), metric);
}

std::vector<double> Compare(
    const DescriptorSet& query,
    const DescriptorBatch& library,
    const Metric& metric,
    DescriptorComparisonMode mode,
    const BatchKernelOptions& options) {
    std::vector<double> output(library.Size(), 0.0);
    CompareInto(query, library, metric, mode, output.data(), output.size(), options);
    return output;
}

void CompareInto(
    const DescriptorSet& query,
    const DescriptorBatch& library,
    const Metric& metric,
    DescriptorComparisonMode mode,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    validate_output(output, output_length, library.Size());
    validate_descriptor_batch_compatibility(query, library);

    detail::ParallelFor(0, library.Size(), options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            output[row] = evaluate_descriptor_batch_row_metric(query, library, row, metric, mode);
        }
    });
}

std::vector<double> Compare(
    const OEFPSparse& query,
    const OEFPSparseBatch& library,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(library.Size(), 0.0);
    CompareInto(query, library, metric, output.data(), output.size(), options);
    return output;
}

void CompareInto(
    const OEFPSparse& query,
    const OEFPSparseBatch& library,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    validate_output(output, output_length, library.Size());
    validate_sparse_fingerprint_batch_compatibility(query, library);

    const auto* query_indices = query.IndexData();
    const auto query_size = query.CountOnBits();
    const auto dimensions = query.SizeBits();
    detail::ParallelFor(0, library.Size(), options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            output[row] = evaluate_sparse_binary_row_metric(
                query_indices,
                query_size,
                library.RowIndices(row),
                library.RowEntryCount(row),
                dimensions,
                metric);
        }
    });
}

std::vector<double> Compare(
    const OEFPCount& query,
    const OEFPCountBatch& library,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(library.Size(), 0.0);
    CompareInto(query, library, metric, output.data(), output.size(), options);
    return output;
}

void CompareInto(
    const OEFPCount& query,
    const OEFPCountBatch& library,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    validate_output(output, output_length, library.Size());
    validate_count_fingerprint_batch_compatibility(query, library);

    const auto* query_indices = query.IndexData();
    const auto* query_counts = query.CountData();
    const auto query_size = query.NonzeroCount();
    const auto dimensions = query.SizeBits();
    detail::ParallelFor(0, library.Size(), options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            output[row] = evaluate_sparse_count_row_metric(
                query_indices,
                query_counts,
                query_size,
                library.RowIndices(row),
                library.RowCounts(row),
                library.RowEntryCount(row),
                dimensions,
                metric);
        }
    });
}

std::vector<double> Compare(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(library.Size(), 0.0);
    CompareInto(query, library, metric, output.data(), output.size(), options);
    return output;
}

void CompareInto(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    validate_output(output, output_length, library.Size());
    validate_fingerprint_batch_compatibility(query, library);

    const auto& query_words = query.Words();
    const auto query_popcount = query.CountOnBits();
    const auto word_count = query.WordCount();
    const auto dimensions = query.SizeBits();
    detail::ParallelFor(0, library.Size(), options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            output[row] = evaluate_dense_binary_row_metric(
                query_words.data(),
                library.RowWords(row),
                word_count,
                query_popcount,
                library.PopCount(row),
                dimensions,
                metric);
        }
    });
}

std::vector<double> CDist(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(checked_product(a.Size(), b.Size(), "CDist output size is too large."), 0.0);
    CDistInto(a, b, metric, output.data(), output.size(), options);
    return output;
}

void CDistInto(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    const auto expected_length = checked_product(a.Size(), b.Size(), "CDist output size is too large.");
    validate_output(output, output_length, expected_length);
    validate_batch_compatibility(a, b);

    const auto word_count = a.WordsPerFingerprint();
    const auto b_size = b.Size();
    const auto dimensions = a.SizeBits();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        auto row_a = begin / b_size;
        auto row_b = begin % b_size;
        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint64_t* a_words = nullptr;
        std::uint32_t a_popcount = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_words = a.RowWords(row_a);
                a_popcount = a.PopCount(row_a);
            }
            output[output_index] = evaluate_dense_binary_row_metric(
                a_words,
                b.RowWords(row_b),
                word_count,
                a_popcount,
                b.PopCount(row_b),
                dimensions,
                metric);

            ++row_b;
            if (row_b == b_size) {
                row_b = 0;
                ++row_a;
            }
        }
    });
}

std::vector<double> CDist(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(checked_product(a.Size(), b.Size(), "CDist output size is too large."), 0.0);
    CDistInto(a, b, metric, output.data(), output.size(), options);
    return output;
}

void CDistInto(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    const auto expected_length = checked_product(a.Size(), b.Size(), "CDist output size is too large.");
    validate_output(output, output_length, expected_length);
    validate_count_batch_compatibility(a, b);

    const auto b_size = b.Size();
    const auto dimensions = a.SizeBits();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        auto row_a = begin / b_size;
        auto row_b = begin % b_size;
        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint32_t* a_indices = nullptr;
        const std::uint32_t* a_counts = nullptr;
        std::size_t a_count = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_indices = a.RowIndices(row_a);
                a_counts = a.RowCounts(row_a);
                a_count = a.RowEntryCount(row_a);
            }
            output[output_index] = evaluate_sparse_count_row_metric(
                a_indices,
                a_counts,
                a_count,
                b.RowIndices(row_b),
                b.RowCounts(row_b),
                b.RowEntryCount(row_b),
                dimensions,
                metric);

            ++row_b;
            if (row_b == b_size) {
                row_b = 0;
                ++row_a;
            }
        }
    });
}

std::vector<double> CDist(
    const OEFPSparseBatch& a,
    const OEFPSparseBatch& b,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(checked_product(a.Size(), b.Size(), "CDist output size is too large."), 0.0);
    CDistInto(a, b, metric, output.data(), output.size(), options);
    return output;
}

void CDistInto(
    const OEFPSparseBatch& a,
    const OEFPSparseBatch& b,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    const auto expected_length = checked_product(a.Size(), b.Size(), "CDist output size is too large.");
    validate_output(output, output_length, expected_length);
    validate_sparse_batch_compatibility(a, b);

    const auto b_size = b.Size();
    const auto dimensions = a.SizeBits();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        auto row_a = begin / b_size;
        auto row_b = begin % b_size;
        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint32_t* a_indices = nullptr;
        std::size_t a_count = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_indices = a.RowIndices(row_a);
                a_count = a.RowEntryCount(row_a);
            }
            output[output_index] = evaluate_sparse_binary_row_metric(
                a_indices,
                a_count,
                b.RowIndices(row_b),
                b.RowEntryCount(row_b),
                dimensions,
                metric);

            ++row_b;
            if (row_b == b_size) {
                row_b = 0;
                ++row_a;
            }
        }
    });
}

std::vector<double> CDist(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    DescriptorComparisonMode mode,
    const BatchKernelOptions& options) {
    std::vector<double> output(checked_product(a.Size(), b.Size(), "CDist output size is too large."), 0.0);
    CDistInto(a, b, metric, mode, output.data(), output.size(), options);
    return output;
}

void CDistInto(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    DescriptorComparisonMode mode,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    const auto expected_length = checked_product(a.Size(), b.Size(), "CDist output size is too large.");
    validate_output(output, output_length, expected_length);
    validate_descriptor_batch_compatibility(a, b);

    const auto b_size = b.Size();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        auto row_a = begin / b_size;
        auto row_b = begin % b_size;
        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            output[output_index] = evaluate_descriptor_batch_row_metric(
                a,
                row_a,
                b,
                row_b,
                metric,
                mode);

            ++row_b;
            if (row_b == b_size) {
                row_b = 0;
                ++row_a;
            }
        }
    });
}

std::vector<double> PDist(
    const OEFPBatch& batch,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(condensed_size(batch.Size()), 0.0);
    PDistInto(batch, metric, output.data(), output.size(), options);
    return output;
}

void PDistInto(
    const OEFPBatch& batch,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    metric.ValidateForPDist();
    const auto expected_length = condensed_size(batch.Size());
    validate_output(output, output_length, expected_length);

    const auto batch_size = batch.Size();
    const auto word_count = batch.WordsPerFingerprint();
    const auto dimensions = batch.SizeBits();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        std::size_t row_a = 0;
        std::size_t row_b = 0;
        condensed_pair_from_index(begin, batch_size, row_a, row_b);

        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint64_t* a_words = nullptr;
        std::uint32_t a_popcount = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_words = batch.RowWords(row_a);
                a_popcount = batch.PopCount(row_a);
            }
            output[output_index] = evaluate_dense_binary_row_metric(
                a_words,
                batch.RowWords(row_b),
                word_count,
                a_popcount,
                batch.PopCount(row_b),
                dimensions,
                metric);

            ++row_b;
            if (row_b == batch_size) {
                ++row_a;
                row_b = row_a + 1;
            }
        }
    });
}

std::vector<double> PDist(
    const OEFPCountBatch& batch,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(condensed_size(batch.Size()), 0.0);
    PDistInto(batch, metric, output.data(), output.size(), options);
    return output;
}

void PDistInto(
    const OEFPCountBatch& batch,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    metric.ValidateForPDist();
    const auto expected_length = condensed_size(batch.Size());
    validate_output(output, output_length, expected_length);

    const auto batch_size = batch.Size();
    const auto dimensions = batch.SizeBits();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        std::size_t row_a = 0;
        std::size_t row_b = 0;
        condensed_pair_from_index(begin, batch_size, row_a, row_b);

        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint32_t* a_indices = nullptr;
        const std::uint32_t* a_counts = nullptr;
        std::size_t a_count = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_indices = batch.RowIndices(row_a);
                a_counts = batch.RowCounts(row_a);
                a_count = batch.RowEntryCount(row_a);
            }
            output[output_index] = evaluate_sparse_count_row_metric(
                a_indices,
                a_counts,
                a_count,
                batch.RowIndices(row_b),
                batch.RowCounts(row_b),
                batch.RowEntryCount(row_b),
                dimensions,
                metric);

            ++row_b;
            if (row_b == batch_size) {
                ++row_a;
                row_b = row_a + 1;
            }
        }
    });
}

std::vector<double> PDist(
    const OEFPSparseBatch& batch,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(condensed_size(batch.Size()), 0.0);
    PDistInto(batch, metric, output.data(), output.size(), options);
    return output;
}

void PDistInto(
    const OEFPSparseBatch& batch,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    metric.ValidateForPDist();
    const auto expected_length = condensed_size(batch.Size());
    validate_output(output, output_length, expected_length);

    const auto batch_size = batch.Size();
    const auto dimensions = batch.SizeBits();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        std::size_t row_a = 0;
        std::size_t row_b = 0;
        condensed_pair_from_index(begin, batch_size, row_a, row_b);

        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint32_t* a_indices = nullptr;
        std::size_t a_count = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_indices = batch.RowIndices(row_a);
                a_count = batch.RowEntryCount(row_a);
            }
            output[output_index] = evaluate_sparse_binary_row_metric(
                a_indices,
                a_count,
                batch.RowIndices(row_b),
                batch.RowEntryCount(row_b),
                dimensions,
                metric);

            ++row_b;
            if (row_b == batch_size) {
                ++row_a;
                row_b = row_a + 1;
            }
        }
    });
}

std::vector<double> PDist(
    const DescriptorBatch& batch,
    const Metric& metric,
    DescriptorComparisonMode mode,
    const BatchKernelOptions& options) {
    std::vector<double> output(condensed_size(batch.Size()), 0.0);
    PDistInto(batch, metric, mode, output.data(), output.size(), options);
    return output;
}

void PDistInto(
    const DescriptorBatch& batch,
    const Metric& metric,
    DescriptorComparisonMode mode,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    metric.ValidateForPDist();
    const auto expected_length = condensed_size(batch.Size());
    validate_output(output, output_length, expected_length);

    const auto batch_size = batch.Size();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        std::size_t row_a = 0;
        std::size_t row_b = 0;
        condensed_pair_from_index(begin, batch_size, row_a, row_b);

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            output[output_index] = evaluate_descriptor_batch_row_metric(
                batch,
                row_a,
                batch,
                row_b,
                metric,
                mode);

            ++row_b;
            if (row_b == batch_size) {
                ++row_a;
                row_b = row_a + 1;
            }
        }
    });
}

void CompareIntoAddress(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CompareInto(query, library, metric, address_to_output(output_address), output_length, options);
}

void CompareIntoAddress(
    const OEFPCount& query,
    const OEFPCountBatch& library,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CompareInto(query, library, metric, address_to_output(output_address), output_length, options);
}

void CompareIntoAddress(
    const OEFPSparse& query,
    const OEFPSparseBatch& library,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CompareInto(query, library, metric, address_to_output(output_address), output_length, options);
}

void CompareIntoAddress(
    const DescriptorSet& query,
    const DescriptorBatch& library,
    const Metric& metric,
    DescriptorComparisonMode mode,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CompareInto(query, library, metric, mode, address_to_output(output_address), output_length, options);
}

void CDistIntoAddress(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CDistInto(a, b, metric, address_to_output(output_address), output_length, options);
}

void CDistIntoAddress(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CDistInto(a, b, metric, address_to_output(output_address), output_length, options);
}

void CDistIntoAddress(
    const OEFPSparseBatch& a,
    const OEFPSparseBatch& b,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CDistInto(a, b, metric, address_to_output(output_address), output_length, options);
}

void CDistIntoAddress(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    DescriptorComparisonMode mode,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CDistInto(a, b, metric, mode, address_to_output(output_address), output_length, options);
}

void PDistIntoAddress(
    const OEFPBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    PDistInto(batch, metric, address_to_output(output_address), output_length, options);
}

void PDistIntoAddress(
    const OEFPCountBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    PDistInto(batch, metric, address_to_output(output_address), output_length, options);
}

void PDistIntoAddress(
    const OEFPSparseBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    PDistInto(batch, metric, address_to_output(output_address), output_length, options);
}

void PDistIntoAddress(
    const DescriptorBatch& batch,
    const Metric& metric,
    DescriptorComparisonMode mode,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    PDistInto(batch, metric, mode, address_to_output(output_address), output_length, options);
}

} // namespace OEFP
