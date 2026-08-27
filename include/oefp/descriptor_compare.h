#ifndef OEFP_DESCRIPTOR_COMPARE_H
#define OEFP_DESCRIPTOR_COMPARE_H

#include "oefp/batch_kernel_options.h"
#include "oefp/descriptor_batch.h"
#include "oefp/descriptor_selection.h"
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

/// \brief Column selection and missing-value policy for numeric descriptor comparison.
///
/// This struct has no default constructor: \c DescriptorSelection is only constructible
/// through its static factories, so \c DescriptorNumericOptions must be aggregate-initialized,
/// for example <tt>{DescriptorSelection::Names({"MW"}), DescriptorMissingPolicy::Ignore}</tt>.
struct DescriptorNumericOptions {
    DescriptorSelection columns;  ///< Scalar Bool, Int, or Float columns, in selection order.
    DescriptorMissingPolicy missing = DescriptorMissingPolicy::Propagate;
};

/// \brief Condensed pairwise distances over a dense numeric descriptor matrix.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// Standardized Euclidean and Mahalanobis whiten every row before comparing, which mixes
/// columns, so a row with any absent value yields NaN for every pair it takes part in. The
/// Mahalanobis inverse covariance must be symmetric; that precondition is not checked.
///
/// \param values Row-major \p rows x \p columns buffer of descriptor values.
/// \param validity Row-major \p rows x \p columns presence mask, or \c nullptr when every
///        value in this matrix is present.
/// \param rows Row count.
/// \param columns Column count. Every output entry is NaN when the column count is zero, unless
///        the metric's own parameters are rejected first.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported.
/// \param missing Missing-value policy.
/// \param kernel Threading options.
/// \return \c rows * (rows - 1) / 2 distances in condensed upper-triangular order.
/// \throws std::invalid_argument: When the metric is not valid for numeric comparison; when
///        weighted Minkowski weights length does not match \p columns; when \p missing is
///        \c Ignore for Standardized Euclidean or Mahalanobis; when \p columns is too large to
///        square; when the Standardized Euclidean variance count does not match \p columns or a
///        variance is not finite and strictly positive; when the Mahalanobis inverse covariance
///        is not square in \p columns or has a non-finite entry; or when the Mahalanobis inverse
///        covariance is not positive semidefinite.
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
/// Standardized Euclidean and Mahalanobis whiten every row before comparing, which mixes
/// columns, so a row with any absent value yields NaN for every pair it takes part in. The
/// Mahalanobis inverse covariance must be symmetric; that precondition is not checked.
///
/// \param values Row-major \p rows x \p columns buffer of descriptor values.
/// \param validity Row-major \p rows x \p columns presence mask, or \c nullptr when every
///        value in this matrix is present.
/// \param rows Row count.
/// \param columns Column count. Every output entry is NaN when the column count is zero, unless
///        the metric's own parameters are rejected first.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported.
/// \param missing Missing-value policy.
/// \param output Destination buffer, caller-owned.
/// \param output_length Destination length; must equal \c rows * (rows - 1) / 2.
/// \param kernel Threading options.
/// \throws std::invalid_argument: When \p output_length is wrong; when the metric is not valid for
///        numeric comparison; when weighted Minkowski weights length does not match \p columns;
///        when \p missing is \c Ignore for Standardized Euclidean or Mahalanobis; when \p columns
///        is too large to square; when the Standardized Euclidean variance count does not match
///        \p columns or a variance is not finite and strictly positive; when the Mahalanobis
///        inverse covariance is not square in \p columns or has a non-finite entry; or when the
///        Mahalanobis inverse covariance is not positive semidefinite.
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
/// Standardized Euclidean and Mahalanobis whiten every row before comparing, which mixes
/// columns, so a row with any absent value yields NaN for every pair it takes part in. The
/// Mahalanobis inverse covariance must be symmetric; that precondition is not checked.
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
/// \param columns Column count. Every output entry is NaN when the column count is zero, unless
///        the metric's own parameters are rejected first.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported.
/// \param missing Missing-value policy.
/// \param kernel Threading options.
/// \return \c a_rows * \c b_rows distances in row-major order.
/// \throws std::invalid_argument: When the metric is not valid for numeric comparison; when
///        weighted Minkowski weights length does not match \p columns; when \p missing is
///        \c Ignore for Standardized Euclidean or Mahalanobis; when \p columns is too large to
///        square; when the Standardized Euclidean variance count does not match \p columns or a
///        variance is not finite and strictly positive; when the Mahalanobis inverse covariance
///        is not square in \p columns or has a non-finite entry; or when the Mahalanobis inverse
///        covariance is not positive semidefinite.
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
/// Standardized Euclidean and Mahalanobis whiten every row before comparing, which mixes
/// columns, so a row with any absent value yields NaN for every pair it takes part in. The
/// Mahalanobis inverse covariance must be symmetric; that precondition is not checked.
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
/// \param columns Column count. Every output entry is NaN when the column count is zero, unless
///        the metric's own parameters are rejected first.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported.
/// \param missing Missing-value policy.
/// \param output Destination buffer, caller-owned.
/// \param output_length Destination length; must equal \c a_rows * \c b_rows.
/// \param kernel Threading options.
/// \throws std::invalid_argument: When \p output_length is wrong; when the metric is not valid for
///        numeric comparison; when weighted Minkowski weights length does not match \p columns;
///        when \p missing is \c Ignore for Standardized Euclidean or Mahalanobis; when \p columns
///        is too large to square; when the Standardized Euclidean variance count does not match
///        \p columns or a variance is not finite and strictly positive; when the Mahalanobis
///        inverse covariance is not square in \p columns or has a non-finite entry; or when the
///        Mahalanobis inverse covariance is not positive semidefinite.
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

/// \brief Condensed pairwise distances over selected numeric columns of a descriptor batch.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// Standardized Euclidean and Mahalanobis whiten every row before comparing, which mixes
/// columns, so a row with any absent value yields NaN for every pair it takes part in. The
/// Mahalanobis inverse covariance must be symmetric; that precondition is not checked.
///
/// \param batch Descriptor batch with a schema.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported.
/// \param options Column selection and missing-value policy.
/// \param kernel Threading options.
/// \return \c batch.Size() * (batch.Size() - 1) / 2 distances in condensed upper-triangular order.
///        Every output entry is NaN when the selected column count is zero, unless the metric's
///        own parameters are rejected first.
/// \throws std::invalid_argument: When the batch is not schema-backed; when a selected column is
///        not a numeric scalar; when the metric is not valid for numeric comparison; when
///        weighted Minkowski weights length does not match the selected column count; when the
///        missing-value policy is \c Ignore for Standardized Euclidean or Mahalanobis; when the
///        selected column count is too large to square; when the Standardized Euclidean variance
///        count does not match the selected column count or a variance is not finite and strictly
///        positive; when the Mahalanobis inverse covariance is not square in the selected column
///        count or has a non-finite entry; or when the Mahalanobis inverse covariance is not
///        positive semidefinite.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an index
///        past the schema's column count.
/// \throws std::runtime_error: When the symmetric eigendecomposition for Mahalanobis does not converge.
std::vector<double> PDist(
    const DescriptorBatch& batch,
    const Metric& metric,
    const DescriptorNumericOptions& options,
    const BatchKernelOptions& kernel = {});

/// \brief Condensed pairwise distances written into a caller-supplied buffer.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// Standardized Euclidean and Mahalanobis whiten every row before comparing, which mixes
/// columns, so a row with any absent value yields NaN for every pair it takes part in. The
/// Mahalanobis inverse covariance must be symmetric; that precondition is not checked.
///
/// \param batch Descriptor batch with a schema.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported.
/// \param options Column selection and missing-value policy.
/// \param output Destination buffer, caller-owned.
/// \param output_length Destination length; must equal \c batch.Size() * (batch.Size() - 1) / 2.
/// \param kernel Threading options.
/// \throws std::invalid_argument: When \p output_length is wrong; when the batch is not
///        schema-backed; when a selected column is not a numeric scalar; when the metric is not
///        valid for numeric comparison; when weighted Minkowski weights length does not match the
///        selected column count; when the missing-value policy is \c Ignore for Standardized
///        Euclidean or Mahalanobis; when the selected column count is too large to square; when the
///        Standardized Euclidean variance count does not match the selected column count or a
///        variance is not finite and strictly positive; when the Mahalanobis inverse covariance is
///        not square in the selected column count or has a non-finite entry; or when the
///        Mahalanobis inverse covariance is not positive semidefinite.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an index
///        past the schema's column count.
/// \throws std::runtime_error: When the symmetric eigendecomposition for Mahalanobis does not converge.
void PDistInto(
    const DescriptorBatch& batch,
    const Metric& metric,
    const DescriptorNumericOptions& options,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel = {});

/// \brief Row-major cross distances over selected numeric columns of two descriptor batches.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// Standardized Euclidean and Mahalanobis whiten every row before comparing, which mixes
/// columns, so a row with any absent value yields NaN for every pair it takes part in. The
/// Mahalanobis inverse covariance must be symmetric; that precondition is not checked.
///
/// \param a First descriptor batch with a schema.
/// \param b Second descriptor batch with a schema.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported.
/// \param options Column selection and missing-value policy.
/// \param kernel Threading options.
/// \return \c a.Size() * \c b.Size() distances in row-major order. Every output entry is NaN when
///        the selected column count is zero, unless the metric's own parameters are rejected first.
/// \throws std::invalid_argument: When the two batches have different schema identifiers; when the
///        selection resolves differently against them; when either batch is not schema-backed; when
///        a selected column is not a numeric scalar; when the metric is not valid for numeric
///        comparison; when weighted Minkowski weights length does not match the selected column
///        count; when the missing-value policy is \c Ignore for Standardized Euclidean or
///        Mahalanobis; when the selected column count is too large to square; when the Standardized
///        Euclidean variance count does not match the selected column count or a variance is not
///        finite and strictly positive; when the Mahalanobis inverse covariance is not square in the
///        selected column count or has a non-finite entry; or when the Mahalanobis inverse
///        covariance is not positive semidefinite.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an index
///        past the schema's column count.
/// \throws std::runtime_error: When the symmetric eigendecomposition for Mahalanobis does not converge.
std::vector<double> CDist(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    const DescriptorNumericOptions& options,
    const BatchKernelOptions& kernel = {});

/// \brief Row-major cross distances written into a caller-supplied buffer.
///
/// Distances are computed without intermediate scaling, so extreme descriptor values or a large
/// Minkowski \c p can cause accumulators to overflow to infinity before the final root or
/// normalization is applied.
///
/// Standardized Euclidean and Mahalanobis whiten every row before comparing, which mixes
/// columns, so a row with any absent value yields NaN for every pair it takes part in. The
/// Mahalanobis inverse covariance must be symmetric; that precondition is not checked.
///
/// \param a First descriptor batch with a schema.
/// \param b Second descriptor batch with a schema.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, Minkowski,
///        BrayCurtis, Standardized Euclidean, and Mahalanobis are supported.
/// \param options Column selection and missing-value policy.
/// \param output Destination buffer, caller-owned.
/// \param output_length Destination length; must equal \c a.Size() * \c b.Size().
/// \param kernel Threading options.
/// \throws std::invalid_argument: When \p output_length is wrong; when the two batches have
///        different schema identifiers; when the selection resolves differently against them; when
///        either batch is not schema-backed; when a selected column is not a numeric scalar; when
///        the metric is not valid for numeric comparison; when weighted Minkowski weights length
///        does not match the selected column count; when the missing-value policy is \c Ignore for
///        Standardized Euclidean or Mahalanobis; when the selected column count is too large to
///        square; when the Standardized Euclidean variance count does not match the selected column
///        count or a variance is not finite and strictly positive; when the Mahalanobis inverse
///        covariance is not square in the selected column count or has a non-finite entry; or when
///        the Mahalanobis inverse covariance is not positive semidefinite.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an index
///        past the schema's column count.
/// \throws std::runtime_error: When the symmetric eigendecomposition for Mahalanobis does not converge.
void CDistInto(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    const DescriptorNumericOptions& options,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel = {});

/// \cond OEFP_BINDING_DETAIL
/// \brief Address-based numeric pdist helper for Python bindings.
///
/// \param values_address Address of a C-contiguous row-major
///        <tt>rows x columns</tt> \c double buffer. Zero throws.
/// \param validity_address Address of a matching \c std::uint8_t mask, or zero when every
///        value is present.
std::vector<double> PDistNumericAddress(
    std::uint64_t values_address,
    std::uint64_t validity_address,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    const BatchKernelOptions& kernel = {});

/// \brief Address-based numeric pdist output helper for Python bindings.
void PDistNumericIntoAddress(
    std::uint64_t values_address,
    std::uint64_t validity_address,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& kernel = {});

/// \brief Address-based numeric cdist helper for Python bindings.
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
    const BatchKernelOptions& kernel = {});

/// \brief Address-based numeric cdist output helper for Python bindings.
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
    const BatchKernelOptions& kernel = {});
/// \endcond

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_COMPARE_H
