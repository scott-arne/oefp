#include "oefp/descriptor_compare.h"

#include "compare_detail.h"
#include "linear_algebra.h"
#include "thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace OEFP {
namespace {

using detail::checked_product;
using detail::condensed_pair_from_index;
using detail::condensed_size;
using detail::evaluate_numeric_metric;
using detail::NumericStats;
using detail::validate_output;

// symmetric_eigensystem_cyclic, not symmetric_eigensystem_jacobi: the cyclic wrapper has the
// same signature under both of Task 3's branches, so nothing here has to know which one was
// chosen. Do not name JacobiPivot in this file; under Branch A that enum does not exist.
using detail::symmetric_eigensystem_cyclic;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool numeric_needs_pre_transform(const Metric& metric) {
    return metric.Name() == MetricName::StandardizedEuclidean
        || metric.Name() == MetricName::Mahalanobis;
}

/// \brief Name a column for a diagnostic, falling back to its index when names are absent.
std::string numeric_column_label(const std::vector<std::string>* names, std::size_t index) {
    if (names != nullptr && index < names->size()) {
        return "'" + (*names)[index] + "'";
    }
    return "index " + std::to_string(index);
}

/// \brief Require every Standardized Euclidean variance to be finite and strictly positive.
///
/// \c Metric::StandardizedEuclidean stores its vector without inspecting it, so NaN,
/// infinity, and negative values all reach the transform otherwise, each yielding a silently
/// NaN distance matrix. NaN is not hypothetical: \c ColumnStatistics reports NaN variance for
/// a column with fewer than two present values, and feeding that straight into the
/// constructor is the documented usage.
void validate_standardized_variances(const std::vector<double>& variances,
                                     std::size_t columns,
                                     const std::vector<std::string>* names) {
    if (variances.size() != columns) {
        throw std::invalid_argument(
            "Standardized Euclidean variance count must match the selected column count.");
    }
    for (std::size_t index = 0u; index < variances.size(); ++index) {
        const auto variance = variances[index];
        if (!std::isfinite(variance) || variance <= 0.0) {
            throw std::invalid_argument(
                "Standardized Euclidean variance for column "
                + numeric_column_label(names, index)
                + " must be finite and strictly positive.");
        }
    }
}

/// \brief Reject a column count whose square cannot be represented.
///
/// Both pre-transform metrics index a columns x columns buffer, and the Mahalanobis size check
/// forms the product before comparing it, so an unchecked multiplication defeats the very check
/// it feeds: columns = 2^32 on a 64-bit size_t makes the product exactly zero, an empty inverse
/// covariance then satisfies the size check, and the eigensolver writes identity entries into a
/// zero-length eigenvector buffer. Mirrors the guard in pseudo_inverse_symmetric.
void validate_pre_transform_columns(std::size_t columns) {
    if (columns != 0u && columns > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::invalid_argument("Pre-transform column count is too large to index.");
    }
}

/// \brief Require the Mahalanobis inverse covariance to be square in \p columns and finite.
///
/// A NaN off-diagonal is invisible to the solver's convergence test — std::max(x, NaN) returns
/// x, so the sweep reports convergence on its first pass and the input diagonal comes back with
/// an identity eigenbasis, yielding finite plausible distances from an invalid matrix. An
/// infinite entry is worse: it makes the eigenvalue tolerance band infinite, which disables the
/// positive-semidefinite rejection for every finite eigenvalue. pseudo_inverse_symmetric rejects
/// non-finite entries for the same reason.
void validate_inverse_covariance(const std::vector<double>& inverse_covariance,
                                 std::size_t columns) {
    if (inverse_covariance.size() != columns * columns) {
        throw std::invalid_argument(
            "Mahalanobis inverse covariance must be square in the selected column count.");
    }
    if (std::any_of(inverse_covariance.begin(), inverse_covariance.end(),
                    [](double value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("Mahalanobis inverse covariance entries must be finite.");
    }
}

/// \brief Build the whitening factor L such that L * L^T equals the metric's quadratic form.
///
/// For Standardized Euclidean this is the diagonal of 1 / sqrt(variance). For Mahalanobis it is
/// Q * sqrt(Lambda) from the eigendecomposition of the supplied inverse covariance.
///
/// Parameter shape and values have already been rejected by \c validate_numeric_metric, so the
/// only rejection left here is the positive-semidefinite one, which cannot be read off the
/// parameters and needs the eigenvalues.
///
/// An asymmetric inverse covariance is interpreted as its symmetric part, (VI + VI^T) / 2, and
/// symmetrized here rather than rejected. That is the matrix the quadratic form d^T * VI * d
/// already defines, so it is also what the fingerprint Mahalanobis path computes with its full
/// double loop over both triangles, and symmetrizing is what keeps the two surfaces in
/// agreement. Rejecting instead would be hostile: a numerically inverted covariance is routinely
/// symmetric only to within rounding.
std::vector<double> whitening_factor(const Metric& metric, std::size_t columns) {
    std::vector<double> factor(columns * columns, 0.0);

    if (metric.Name() == MetricName::StandardizedEuclidean) {
        for (std::size_t index = 0u; index < columns; ++index) {
            factor[index * columns + index] = 1.0 / std::sqrt(metric.Variances()[index]);
        }
        return factor;
    }

    const auto& inverse_covariance = metric.InverseCovariance();

    // Symmetrize first, so everything below -- the scaling, the eigendecomposition, and the
    // positive-semidefinite verdict -- describes the matrix the distance actually uses.
    //
    // A pair that already agrees is copied rather than averaged, which keeps a symmetric input
    // bit-exact at every magnitude. That matters at both ends of the range: 0.5 * (a + a)
    // overflows to infinity for |a| > DBL_MAX / 2, turning a finite DBL_MAX matrix into an
    // infinite one before the scaling below ever sees it, and halving rounds in the subnormal
    // range. Genuinely differing entries average as 0.5 * a + 0.5 * b, which cannot overflow for
    // finite inputs; 0.5 * (a + b) can.
    std::vector<double> symmetric(inverse_covariance.size(), 0.0);
    for (std::size_t row = 0u; row < columns; ++row) {
        for (std::size_t column = 0u; column < columns; ++column) {
            const auto upper = inverse_covariance[row * columns + column];
            const auto lower = inverse_covariance[column * columns + row];
            symmetric[row * columns + column] =
                upper == lower ? upper : 0.5 * upper + 0.5 * lower;
        }
    }

    // The solver converges on a fixed absolute off-diagonal threshold, so a uniformly tiny
    // inverse covariance would come back undiagonalized and every distance would be silently
    // wrong. Decompose a copy scaled so its largest entry lands in [0.5, 1) and undo the scale
    // on the eigenvalues afterwards; eigenvectors are invariant under a uniform scale. The
    // scale is a power of two, so both directions are exact and no existing result moves.
    // Measured on the symmetrized entries, which are the ones being decomposed.
    double max_magnitude = 0.0;
    for (const auto value : symmetric) {
        max_magnitude = std::max(max_magnitude, std::abs(value));
    }
    int exponent = 0;
    if (max_magnitude > 0.0) {
        std::frexp(max_magnitude, &exponent);
    }

    std::vector<double> scaled(symmetric.size(), 0.0);
    for (std::size_t index = 0u; index < symmetric.size(); ++index) {
        scaled[index] = std::ldexp(symmetric[index], -exponent);
    }

    auto eigensystem = symmetric_eigensystem_cyclic(std::move(scaled), columns);
    if (!eigensystem.has_value()) {
        throw std::runtime_error("Symmetric eigendecomposition did not converge.");
    }

    // The sign test and its tolerance both run in the scaled domain, before the scale is
    // restored. Restoring first can overflow a large eigenvalue to infinity, which makes the
    // tolerance infinite and reduces the rejection to -inf < -inf, i.e. false: a wildly
    // indefinite matrix would then be clamped to zero and yield distances of 0.0. Scaling by a
    // positive power of two is exact and monotone, so it preserves both the sign and the ratio
    // a relative tolerance compares, and the verdict here is identical to the unscaled one
    // wherever the unscaled one is representable. In this domain the largest entry is below 1,
    // so no eigenvalue exceeds the column count in magnitude and nothing can overflow.
    double max_eigenvalue = 0.0;
    for (const auto value : eigensystem->eigenvalues) {
        max_eigenvalue = std::max(max_eigenvalue, std::abs(value));
    }
    // Matches the rcond default used by InverseCovarianceMatrix, so a matrix this library
    // produced always passes. Eigenvalues inside the band are clamped rather than rejected.
    const auto tolerance =
        std::numeric_limits<double>::epsilon() * static_cast<double>(columns) * max_eigenvalue;

    for (std::size_t k = 0u; k < columns; ++k) {
        auto eigenvalue = eigensystem->eigenvalues[k];
        if (eigenvalue < -tolerance) {
            throw std::invalid_argument(
                "Mahalanobis inverse covariance must be positive semidefinite.");
        }
        if (eigenvalue < 0.0) {
            eigenvalue = 0.0;
        }
        // Restore the scale only here, where the magnitude is actually needed.
        //
        // Halve the exponent and take the root before restoring the scale. Restoring first can
        // overflow the intermediate to infinity for a legitimate positive semidefinite matrix
        // whose eigenvalues approach the double range, and an infinite factor then turns even a
        // zero-length difference into 0.0 * inf, i.e. NaN. The halving is exact, the * 2.0 /
        // * 0.5 that absorbs an odd remainder is exact, and scaling a correctly rounded square
        // root by a power of two lands on the same double as rounding the scaled exact value, so
        // this is bit-identical to the direct form wherever that form's intermediate stays
        // normal. Where the intermediate would instead underflow to a subnormal the two differ,
        // and this form is the correctly rounded one.
        const int half = exponent / 2;
        const int remainder = exponent - 2 * half;
        const auto mantissa =
            remainder == 0 ? eigenvalue : (remainder > 0 ? eigenvalue * 2.0 : eigenvalue * 0.5);
        const auto root = std::ldexp(std::sqrt(mantissa), half);
        for (std::size_t row = 0u; row < columns; ++row) {
            factor[row * columns + k] = eigensystem->eigenvectors[row * columns + k] * root;
        }
    }

    return factor;
}

struct NumericPreTransform {
    std::vector<double> values;          ///< Row-major, rows x columns, whitened.
    std::vector<std::uint8_t> complete;  ///< One entry per row; 1 when every column is present.
};

NumericPreTransform build_pre_transform(const double* values, const std::uint8_t* validity,
                                        std::size_t rows, std::size_t columns,
                                        const std::vector<double>& factor) {
    NumericPreTransform transformed;
    transformed.values.assign(rows * columns, 0.0);
    transformed.complete.assign(rows, 1u);

    for (std::size_t row = 0u; row < rows; ++row) {
        if (validity != nullptr) {
            for (std::size_t column = 0u; column < columns; ++column) {
                if (validity[row * columns + column] == 0u) {
                    transformed.complete[row] = 0u;
                    break;
                }
            }
        }

        // Transform every row regardless. Incomplete rows carry meaningless but finite
        // values, so there is no NaN poisoning and no branch in the pair loop's inner loop.
        for (std::size_t k = 0u; k < columns; ++k) {
            double sum = 0.0;
            for (std::size_t i = 0u; i < columns; ++i) {
                sum += values[row * columns + i] * factor[i * columns + k];
            }
            transformed.values[row * columns + k] = sum;
        }
    }

    return transformed;
}

/// \brief Reject metrics that have no defined meaning over continuous descriptor columns.
void validate_numeric_metric(const Metric& metric, DescriptorMissingPolicy missing,
                             std::size_t columns, const std::vector<std::string>* names) {
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
    case MetricName::StandardizedEuclidean:
    case MetricName::Mahalanobis:
        if (missing == DescriptorMissingPolicy::Ignore) {
            throw std::invalid_argument(
                "The Ignore missing-value policy is not valid for Standardized Euclidean or "
                "Mahalanobis, because their pre-transform mixes columns.");
        }
        // Parameter shape and values are rejected here rather than in whitening_factor so the
        // established metric-then-size-then-output order holds for these metrics too. The
        // positive-semidefinite check is derived from an eigendecomposition rather than read off
        // the parameters, so it stays with the decomposition.
        validate_pre_transform_columns(columns);
        if (metric.Name() == MetricName::StandardizedEuclidean) {
            validate_standardized_variances(metric.Variances(), columns, names);
        } else {
            validate_inverse_covariance(metric.InverseCovariance(), columns);
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
    // A present value may legitimately be NaN -- RDKit's BCUT2D columns are emitted as NaN when
    // an element has no Gasteiger parameters (src/rdkit_descriptors.cpp) -- and std::max returns
    // its first argument whenever the comparison is false, so a NaN difference would leave the
    // running maximum at the last finite value. Chebyshev would then report a confident distance
    // derived from indefinite data, and one that depends on column order. Propagate instead, as
    // every other metric here already does through ordinary arithmetic. Once max_abs is NaN it
    // stays NaN, because std::max(NaN, x) also returns its first argument.
    stats.max_abs = std::isnan(abs_difference) ? abs_difference
                                               : std::max(stats.max_abs, abs_difference);
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

void pdist_numeric_impl(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel,
    const std::vector<std::string>* names) {
    validate_numeric_metric(metric, missing, columns, names);
    const auto expected_length = condensed_size(rows);
    validate_output(output, output_length, expected_length);

    if (numeric_needs_pre_transform(metric)) {
        const auto factor = whitening_factor(metric, columns);
        const auto transformed = build_pre_transform(values, validity, rows, columns, factor);
        const auto euclidean = Metric::Euclidean();
        detail::ParallelFor(0, expected_length, kernel.chunk_size, kernel.num_threads,
                            [&](std::size_t begin, std::size_t end) {
            for (std::size_t index = begin; index < end; ++index) {
                std::size_t i = 0u;
                std::size_t j = 0u;
                condensed_pair_from_index(index, rows, i, j);
                if (transformed.complete[i] == 0u || transformed.complete[j] == 0u) {
                    output[index] = kNaN;
                    continue;
                }
                output[index] = compare_numeric_rows<DescriptorMissingPolicy::Propagate,
                                                     false, false>(
                    transformed.values.data() + i * columns, nullptr,
                    transformed.values.data() + j * columns, nullptr,
                    columns, euclidean, static_cast<double>(columns));
            }
        });
        return;
    }

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

void cdist_numeric_impl(
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
    const BatchKernelOptions& kernel,
    const std::vector<std::string>* names) {
    validate_numeric_metric(metric, missing, columns, names);
    const auto expected_length =
        checked_product(a_rows, b_rows, "CDist output size is too large.");
    validate_output(output, output_length, expected_length);

    if (numeric_needs_pre_transform(metric)) {
        const auto factor = whitening_factor(metric, columns);
        const auto a_transformed = build_pre_transform(a_values, a_validity, a_rows, columns, factor);
        const auto b_transformed = build_pre_transform(b_values, b_validity, b_rows, columns, factor);
        const auto euclidean = Metric::Euclidean();
        detail::ParallelFor(0, expected_length, kernel.chunk_size, kernel.num_threads,
                            [&](std::size_t begin, std::size_t end) {
            for (std::size_t index = begin; index < end; ++index) {
                const auto i = index / b_rows;
                const auto j = index % b_rows;
                if (a_transformed.complete[i] == 0u || b_transformed.complete[j] == 0u) {
                    output[index] = kNaN;
                    continue;
                }
                output[index] = compare_numeric_rows<DescriptorMissingPolicy::Propagate,
                                                     false, false>(
                    a_transformed.values.data() + i * columns, nullptr,
                    b_transformed.values.data() + j * columns, nullptr,
                    columns, euclidean, static_cast<double>(columns));
            }
        });
        return;
    }

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

const double* numeric_values_from_address(std::uint64_t address, const char* what) {
    if (address == 0u) {
        throw std::invalid_argument(std::string(what) + " address must not be zero.");
    }
    return reinterpret_cast<const double*>(static_cast<std::uintptr_t>(address));
}

/// \brief Reject a names vector that cannot describe the compared columns.
///
/// The address forms take names only to improve a rejection message, so an empty vector is the
/// ordinary "no names available" case. A non-empty vector of the wrong length is a binding
/// mistake, and reporting it beats silently labelling the wrong column or falling back to an
/// index the caller did not ask for.
void validate_numeric_names(const std::vector<std::string>& names, std::size_t columns) {
    if (!names.empty() && names.size() != columns) {
        throw std::invalid_argument(
            "The column name count must match the compared column count.");
    }
}

const std::uint8_t* numeric_validity_from_address(std::uint64_t address) {
    // Zero is the documented "everything is present" sentinel, not an error.
    return address == 0u ? nullptr
                         : reinterpret_cast<const std::uint8_t*>(
                               static_cast<std::uintptr_t>(address));
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
    pdist_numeric_impl(values, validity, rows, columns, metric, missing, output, output_length,
                       kernel, nullptr);
}

std::vector<double> PDistNumeric(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    const BatchKernelOptions& kernel) {
    // Metric first, then the output, as spec 6.1 requires. condensed_size and the allocation both
    // precede the kernel's own validation, so without this an invalid metric on a large row count
    // surfaces as "Pairwise output size is too large" or std::bad_alloc instead of the metric
    // rejection -- the diagnostic would depend on the row count. Deliberately duplicated: the
    // kernel validates again, and the check is O(columns) and idempotent.
    validate_numeric_metric(metric, missing, columns, nullptr);

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
    cdist_numeric_impl(a_values, a_validity, a_rows, b_values, b_validity, b_rows, columns,
                       metric, missing, output, output_length, kernel, nullptr);
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
    // Metric first, then the output; see PDistNumeric for why the allocation cannot come first.
    validate_numeric_metric(metric, missing, columns, nullptr);

    std::vector<double> output(
        checked_product(a_rows, b_rows, "CDist output size is too large."), 0.0);
    CDistNumericInto(a_values, a_validity, a_rows, b_values, b_validity, b_rows, columns, metric,
                     missing, output.data(), output.size(), kernel);
    return output;
}

void PDistInto(
    const DescriptorBatch& batch,
    const Metric& metric,
    const DescriptorNumericOptions& options,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel) {
    const auto matrix = batch.ToNumericMatrix(options.columns);
    pdist_numeric_impl(matrix.values.data(), matrix.validity.data(), matrix.rows, matrix.columns,
                       metric, options.missing, output, output_length, kernel, &matrix.names);
}

std::vector<double> PDist(
    const DescriptorBatch& batch,
    const Metric& metric,
    const DescriptorNumericOptions& options,
    const BatchKernelOptions& kernel) {
    // Metric first, then the output, as spec 6.1 requires. Delegating to PDistInto would invert
    // that: the output is quadratic in the row count, so a rejected metric on a large batch would
    // surface as std::bad_alloc instead of std::invalid_argument. The matrix is only
    // rows x columns, so materializing it first is the cheap half of the work.
    const auto matrix = batch.ToNumericMatrix(options.columns);
    // Deliberately duplicated: pdist_numeric_impl validates again. The check is O(columns) and
    // idempotent, and paying it twice is what gets the rejection out before the allocation.
    validate_numeric_metric(metric, options.missing, matrix.columns, &matrix.names);

    std::vector<double> output(condensed_size(matrix.rows), 0.0);
    pdist_numeric_impl(matrix.values.data(), matrix.validity.data(), matrix.rows, matrix.columns,
                       metric, options.missing, output.data(), output.size(), kernel,
                       &matrix.names);
    return output;
}

void CDistInto(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    const DescriptorNumericOptions& options,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel) {
    if (a.Schema().SchemaId() != b.Schema().SchemaId()) {
        throw std::invalid_argument(
            "Descriptor batches must share a schema identifier for numeric comparison.");
    }

    const auto a_matrix = a.ToNumericMatrix(options.columns);
    const auto b_matrix = b.ToNumericMatrix(options.columns);
    // This guard cannot fire today: schema_id_ is produced by schema_id_for
    // (src/descriptor_schema.cpp:92-171), which serializes the ordered definition list, so equal
    // ids imply identical definitions in identical order, and all three DescriptorSelection modes
    // (Names, Group, Indices) resolve identically against identical definitions. Keep it as
    // defense-in-depth because it is cheap and the invariant it depends on lives in another file.
    if (a_matrix.names != b_matrix.names) {
        throw std::invalid_argument(
            "The column selection must resolve identically against both descriptor batches.");
    }

    cdist_numeric_impl(a_matrix.values.data(), a_matrix.validity.data(), a_matrix.rows,
                       b_matrix.values.data(), b_matrix.validity.data(), b_matrix.rows,
                       a_matrix.columns, metric, options.missing, output, output_length, kernel,
                       &a_matrix.names);
}

std::vector<double> CDist(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    const DescriptorNumericOptions& options,
    const BatchKernelOptions& kernel) {
    // Metric first, then the output, as spec 6.1 requires; see PDist for why delegating to
    // CDistInto would invert that. CDistInto's two guards are carried over, not dropped.
    if (a.Schema().SchemaId() != b.Schema().SchemaId()) {
        throw std::invalid_argument(
            "Descriptor batches must share a schema identifier for numeric comparison.");
    }

    const auto a_matrix = a.ToNumericMatrix(options.columns);
    const auto b_matrix = b.ToNumericMatrix(options.columns);
    // This guard cannot fire today: schema_id_ is produced by schema_id_for
    // (src/descriptor_schema.cpp:92-171), which serializes the ordered definition list, so equal
    // ids imply identical definitions in identical order, and all three DescriptorSelection modes
    // (Names, Group, Indices) resolve identically against identical definitions. Keep it as
    // defense-in-depth because it is cheap and the invariant it depends on lives in another file.
    if (a_matrix.names != b_matrix.names) {
        throw std::invalid_argument(
            "The column selection must resolve identically against both descriptor batches.");
    }

    // Deliberately duplicated: cdist_numeric_impl validates again. The check is O(columns) and
    // idempotent, and paying it twice is what gets the rejection out before the allocation.
    validate_numeric_metric(metric, options.missing, a_matrix.columns, &a_matrix.names);

    std::vector<double> output(
        checked_product(a_matrix.rows, b_matrix.rows, "CDist output size is too large."), 0.0);
    cdist_numeric_impl(a_matrix.values.data(), a_matrix.validity.data(), a_matrix.rows,
                       b_matrix.values.data(), b_matrix.validity.data(), b_matrix.rows,
                       a_matrix.columns, metric, options.missing, output.data(), output.size(),
                       kernel, &a_matrix.names);
    return output;
}

void PDistNumericIntoAddress(
    std::uint64_t values_address,
    std::uint64_t validity_address,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& kernel,
    const std::vector<std::string>& names) {
    validate_numeric_names(names, columns);
    const auto* values = numeric_values_from_address(values_address, "The value buffer");
    auto* output = reinterpret_cast<double*>(static_cast<std::uintptr_t>(output_address));
    if (output == nullptr) {
        throw std::invalid_argument("The output buffer address must not be zero.");
    }
    // Call the kernel directly rather than PDistNumericInto so the resolved column names reach
    // the rejection messages; PDistNumericInto is a nameless buffer API and passes nullptr.
    pdist_numeric_impl(values, numeric_validity_from_address(validity_address), rows, columns,
                       metric, missing, output, output_length, kernel,
                       names.empty() ? nullptr : &names);
}

std::vector<double> PDistNumericAddress(
    std::uint64_t values_address,
    std::uint64_t validity_address,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    const BatchKernelOptions& kernel,
    const std::vector<std::string>& names) {
    // Call the kernel directly, exactly as PDistNumeric does through its pointer form, rather
    // than routing through PDistNumericIntoAddress. Going through the address form would reject
    // the empty output a one-row pdist legitimately produces, because an empty vector's data()
    // may be null and the *IntoAddress forms treat a zero output address as an error.
    // Short-circuiting before the call instead would skip validate_numeric_metric, making metric
    // and missing-policy validation depend on the row count. The kernel does both correctly: it
    // validates first, and validate_output accepts a null output at length zero.
    validate_numeric_names(names, columns);
    const auto* values = numeric_values_from_address(values_address, "The value buffer");
    const auto* validity = numeric_validity_from_address(validity_address);

    // Metric first, then the output; see PDistNumeric. Deliberately duplicated with the kernel.
    validate_numeric_metric(metric, missing, columns, names.empty() ? nullptr : &names);

    std::vector<double> output(condensed_size(rows), 0.0);
    pdist_numeric_impl(values, validity, rows, columns, metric, missing, output.data(),
                       output.size(), kernel, names.empty() ? nullptr : &names);
    return output;
}

void CDistNumericIntoAddress(
    std::uint64_t a_values_address,
    std::uint64_t a_validity_address,
    std::size_t a_rows,
    std::uint64_t b_values_address,
    std::uint64_t b_validity_address,
    std::size_t b_rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& kernel,
    const std::vector<std::string>& names) {
    validate_numeric_names(names, columns);
    const auto* a_values = numeric_values_from_address(a_values_address, "The first value buffer");
    const auto* b_values =
        numeric_values_from_address(b_values_address, "The second value buffer");
    auto* output = reinterpret_cast<double*>(static_cast<std::uintptr_t>(output_address));
    if (output == nullptr) {
        throw std::invalid_argument("The output buffer address must not be zero.");
    }
    // Call the kernel directly rather than CDistNumericInto so the resolved column names reach
    // the rejection messages; CDistNumericInto is a nameless buffer API and passes nullptr.
    cdist_numeric_impl(a_values, numeric_validity_from_address(a_validity_address), a_rows,
                       b_values, numeric_validity_from_address(b_validity_address), b_rows,
                       columns, metric, missing, output, output_length, kernel,
                       names.empty() ? nullptr : &names);
}

std::vector<double> CDistNumericAddress(
    std::uint64_t a_values_address,
    std::uint64_t a_validity_address,
    std::size_t a_rows,
    std::uint64_t b_values_address,
    std::uint64_t b_validity_address,
    std::size_t b_rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    const BatchKernelOptions& kernel,
    const std::vector<std::string>& names) {
    // Calls the kernel directly for the reason given in PDistNumericAddress.
    validate_numeric_names(names, columns);
    const auto* a_values = numeric_values_from_address(a_values_address, "The first value buffer");
    const auto* b_values =
        numeric_values_from_address(b_values_address, "The second value buffer");

    // Metric first, then the output; see PDistNumeric. Deliberately duplicated with the kernel.
    validate_numeric_metric(metric, missing, columns, names.empty() ? nullptr : &names);

    std::vector<double> output(
        checked_product(a_rows, b_rows, "CDist output size is too large."), 0.0);
    cdist_numeric_impl(a_values, numeric_validity_from_address(a_validity_address), a_rows,
                       b_values, numeric_validity_from_address(b_validity_address), b_rows,
                       columns, metric, missing, output.data(), output.size(), kernel,
                       names.empty() ? nullptr : &names);
    return output;
}

} // namespace OEFP
