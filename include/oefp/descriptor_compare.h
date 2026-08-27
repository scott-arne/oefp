#ifndef OEFP_DESCRIPTOR_COMPARE_H
#define OEFP_DESCRIPTOR_COMPARE_H

#include "oefp/batch_kernel_options.h"
#include "oefp/metric.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Policy for descriptor dimensions missing from either compared row.
///
/// A dimension is missing for a pair when either row's validity mask marks it absent. The
/// two policies differ only in what happens then; a pair with nothing missing is computed
/// identically under both.
enum class DescriptorMissingPolicy {
    /// \brief Any dimension missing from either row yields NaN for that pair. Default.
    Propagate,

    /// \brief Drop the missing dimension for that pair, then rescale so the result stays
    ///        comparable with a pair that used every dimension.
    ///
    /// The rescale multiplies the accumulator by the total weight mass over the used weight
    /// mass, where the total weight mass is the sum of weights for all columns and the used
    /// weight mass is the sum of weights for columns present in both rows. For unweighted
    /// metrics, each column has weight 1, so this reduces to the total column count over the
    /// number of dimensions present in both rows. It applies to Euclidean, Manhattan, Canberra,
    /// and Minkowski, whose accumulators grow with the number of dimensions summed. Chebyshev,
    /// Hamming, and BrayCurtis are left alone because none of them varies with how many
    /// dimensions survived: Chebyshev is a maximum, Hamming already divides by the number of
    /// dimensions actually used, and BrayCurtis is a ratio of two sums of the same degree, so
    /// a common factor would cancel out of it anyway.
    ///
    /// The pair is NaN when no dimension is present in both rows, or when at least one
    /// dimension was dropped and the used weight mass is zero, because there is nothing to
    /// rescale from.
    Ignore,
};

/// \brief Condensed pairwise distances over a dense numeric descriptor matrix.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// \param values Row-major \p rows x \p columns buffer of descriptor values.
/// \param validity Row-major \p rows x \p columns presence mask, or \c nullptr when every
///        value in this matrix is present.
/// \param rows Row count.
/// \param columns Column count. Every output entry is NaN when the column count is zero.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported. Type: see Metric.
/// \param missing Missing-value policy.
/// \param kernel Threading options.
/// \return \c rows * (rows - 1) / 2 distances in condensed upper-triangular order.
/// \throws std::invalid_argument: When the metric is not valid for numeric comparison, or when
///        weighted Minkowski weights length does not match \p columns.
/// \throws std::runtime_error: When the symmetric eigendecomposition for Mahalanobis does not converge.
std::vector<double> PDistNumeric(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing = DescriptorMissingPolicy::Propagate,
    const BatchKernelOptions& kernel = {});

/// \brief Condensed pairwise distances written into a caller-supplied buffer.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// \param values Row-major \p rows x \p columns buffer of descriptor values.
/// \param validity Row-major \p rows x \p columns presence mask, or \c nullptr when every
///        value in this matrix is present.
/// \param rows Row count.
/// \param columns Column count. Every output entry is NaN when the column count is zero.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported. Type: see Metric.
/// \param missing Missing-value policy.
/// \param output Destination buffer, caller-owned.
/// \param output_length Destination length; must equal \c rows * (rows - 1) / 2.
/// \param kernel Threading options.
/// \throws std::invalid_argument: When \p output_length is wrong, the metric is invalid, or when
///        weighted Minkowski weights length does not match \p columns.
/// \throws std::runtime_error: When the symmetric eigendecomposition for Mahalanobis does not converge.
void PDistNumericInto(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel = {});

/// \brief Rectangular distances between two dense numeric descriptor matrices.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// \param a_values Row-major \p a_rows x \p columns buffer of descriptor values.
/// \param a_validity Row-major \p a_rows x \p columns presence mask, or \c nullptr when every
///        value in this matrix is present. The two validity masks are independent; a mask for
///        one matrix and \c nullptr for the other is ordinary usage, not an error.
/// \param a_rows Row count for the first matrix.
/// \param b_values Row-major \p b_rows x \p columns buffer of descriptor values.
/// \param b_validity Row-major \p b_rows x \p columns presence mask, or \c nullptr when every
///        value in this matrix is present. See \p a_validity for the independence contract.
/// \param b_rows Row count for the second matrix.
/// \param columns Column count. Every output entry is NaN when the column count is zero.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported. Type: see Metric.
/// \param missing Missing-value policy.
/// \param kernel Threading options.
/// \return \c a_rows * \c b_rows distances in row-major order.
/// \throws std::invalid_argument: When the metric is not valid for numeric comparison, or when
///        weighted Minkowski weights length does not match \p columns.
/// \throws std::runtime_error: When the symmetric eigendecomposition for Mahalanobis does not converge.
std::vector<double> CDistNumeric(
    const double* a_values,
    const std::uint8_t* a_validity,
    std::size_t a_rows,
    const double* b_values,
    const std::uint8_t* b_validity,
    std::size_t b_rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing = DescriptorMissingPolicy::Propagate,
    const BatchKernelOptions& kernel = {});

/// \brief Rectangular distances written into a caller-supplied buffer.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// \param a_values Row-major \p a_rows x \p columns buffer of descriptor values.
/// \param a_validity Row-major \p a_rows x \p columns presence mask, or \c nullptr when every
///        value in this matrix is present. The two validity masks are independent; a mask for
///        one matrix and \c nullptr for the other is ordinary usage, not an error.
/// \param a_rows Row count for the first matrix.
/// \param b_values Row-major \p b_rows x \p columns buffer of descriptor values.
/// \param b_validity Row-major \p b_rows x \p columns presence mask, or \c nullptr when every
///        value in this matrix is present. See \p a_validity for the independence contract.
/// \param b_rows Row count for the second matrix.
/// \param columns Column count. Every output entry is NaN when the column count is zero.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported. Type: see Metric.
/// \param missing Missing-value policy.
/// \param output Destination buffer, caller-owned.
/// \param output_length Destination length; must equal \c a_rows * \c b_rows.
/// \param kernel Threading options.
/// \throws std::invalid_argument: When \p output_length is wrong, the metric is invalid, or when
///        weighted Minkowski weights length does not match \p columns.
/// \throws std::runtime_error: When the symmetric eigendecomposition for Mahalanobis does not converge.
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
    const BatchKernelOptions& kernel = {});

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_COMPARE_H
