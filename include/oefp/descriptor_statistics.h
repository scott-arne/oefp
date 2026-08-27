#ifndef OEFP_DESCRIPTOR_STATISTICS_H
#define OEFP_DESCRIPTOR_STATISTICS_H

#include "oefp/descriptor_batch.h"
#include "oefp/descriptor_selection.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace OEFP {

/// \brief Per-column summary statistics over the present values of a numeric matrix.
///
/// Each column is summarized independently, using only the values its validity mask marks
/// present. A column with no present values reports NaN for \c mean, \c minimum, and
/// \c maximum; a column with fewer than two present values reports NaN for \c variance.
struct DescriptorColumnStatistics {
    std::vector<std::string> names;         ///< Column names; empty for the buffer overloads.
    std::vector<double> mean;               ///< Arithmetic mean of the present values.
    std::vector<double> variance;           ///< Sample variance, denominator n - 1.
    std::vector<double> minimum;            ///< Smallest present value.
    std::vector<double> maximum;            ///< Largest present value.
    std::vector<std::uint64_t> present_count;  ///< Number of present values per column.
};

/// \brief Sample covariance matrix over complete rows.
struct DescriptorCovariance {
    std::vector<double> matrix;        ///< Row-major, \c columns x \c columns.
    std::uint64_t row_count = 0;       ///< Number of complete rows that contributed.
};

/// \brief Moore-Penrose pseudo-inverse of a covariance matrix.
struct DescriptorInverseCovariance {
    std::vector<double> matrix;        ///< Row-major, \c columns x \c columns.
    std::uint64_t row_count = 0;       ///< Number of complete rows that contributed.
    std::size_t rank = 0;              ///< Number of retained eigenvalues.
};

/// \brief Summary statistics for the selected numeric columns of a descriptor batch.
///
/// \throws std::invalid_argument: When the batch is not schema-backed, a selected column is
///         not a numeric scalar, the matrix extents overflow, a column's buffer holds fewer
///         values than the batch has rows, a present \c Int value has magnitude strictly
///         greater than 2^53, or the selection resolves to zero columns.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an
///         index past the schema's column count.
DescriptorColumnStatistics ColumnStatistics(
    const DescriptorBatch& batch,
    const DescriptorSelection& columns);

/// \brief Summary statistics for a caller-supplied row-major matrix.
///
/// \param values Row-major matrix of \p rows x \p columns doubles. May be null only when
///        \p rows is zero, which is what an empty batch's <tt>std::vector::data()</tt> is
///        allowed to return; the result is then all-NaN with zero present counts.
/// \param validity Row-major mask of the same shape, or null when every value is present.
/// \throws std::invalid_argument: When \p values is null and \p rows is non-zero, when
///         \p columns is zero, or when <tt>columns * columns</tt> overflows \c std::size_t.
DescriptorColumnStatistics ColumnStatistics(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns);

/// \brief Sample covariance over the rows where every selected column is present.
///
/// Rows with any missing selected value are dropped entirely (listwise deletion). The
/// denominator is <tt>row_count - 1</tt>.
///
/// \note Columns of large magnitude can overflow the centred products, in which case the
///       returned matrix holds infinities or NaN and says nothing about it. Nothing is
///       rejected here; the finiteness check lives in \c InverseCovarianceMatrix, which
///       cannot decompose such a matrix.
/// \throws std::invalid_argument: When the batch is not schema-backed, a selected column is
///         not a numeric scalar, the matrix extents overflow, a column's buffer holds fewer
///         values than the batch has rows, a present \c Int value has magnitude strictly
///         greater than 2^53, the selection resolves to zero columns, or fewer than two
///         complete rows remain.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an
///         index past the schema's column count.
DescriptorCovariance CovarianceMatrix(
    const DescriptorBatch& batch,
    const DescriptorSelection& columns);

/// \brief Sample covariance for a caller-supplied row-major matrix.
///
/// Rows with any missing value are dropped entirely (listwise deletion), so a value can be
/// excluded because a different column in its row was absent. The denominator is
/// <tt>row_count - 1</tt>, the count of rows that survived that deletion.
///
/// \param validity Row-major mask of the same shape, or null when every value is present.
/// \note Columns of large magnitude can overflow the centred products, in which case the
///       returned matrix holds infinities or NaN and says nothing about it. Nothing is
///       rejected here; the finiteness check lives in \c InverseCovarianceMatrix, which
///       cannot decompose such a matrix.
/// \throws std::invalid_argument: When \p values is null and \p rows is non-zero, when
///         \p columns is zero, when <tt>columns * columns</tt> overflows \c std::size_t, or
///         when fewer than two complete rows remain.
DescriptorCovariance CovarianceMatrix(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns);

/// \brief Moore-Penrose pseudo-inverse of the sample covariance matrix.
///
/// Eigenvalues are retained when <tt>value > cutoff</tt>, where \c cutoff is the effective
/// \p rcond — the caller's value, or the default below when that value is zero — times
/// <tt>max|eigenvalue|</tt>, the largest eigenvalue magnitude. The comparison is strict, so a
/// covariance matrix whose largest eigenvalue is zero retains nothing.
///
/// \param rcond Relative eigenvalue cutoff. Zero selects
///        <tt>std::numeric_limits<double>::epsilon() * columns</tt>.
/// \note A covariance entry can be non-finite without any non-finite value in the batch:
///       columns of large magnitude overflow the centred products. \c CovarianceMatrix
///       returns that matrix without complaint, so the rejection surfaces here.
/// \throws std::invalid_argument: When the batch is not schema-backed, a selected column is
///         not a numeric scalar, the matrix extents overflow, a column's buffer holds fewer
///         values than the batch has rows, a present \c Int value has magnitude strictly
///         greater than 2^53, the selection resolves to zero columns, fewer than two complete
///         rows remain, <tt>columns * columns</tt> overflows \c std::size_t, a covariance
///         entry is not finite, \p rcond is negative or not finite, no eigenvalue survives the
///         cutoff, or a retained eigenvalue's reciprocal overflows.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an
///         index past the schema's column count.
/// \throws std::runtime_error: When the eigendecomposition does not converge.
DescriptorInverseCovariance InverseCovarianceMatrix(
    const DescriptorBatch& batch,
    const DescriptorSelection& columns,
    double rcond = 0.0);

/// \brief Pseudo-inverse of the sample covariance for a caller-supplied row-major matrix.
///
/// Rows with any missing value are dropped entirely (listwise deletion), so a value can be
/// excluded because a different column in its row was absent. The covariance denominator is
/// <tt>row_count - 1</tt>, the count of rows that survived that deletion.
///
/// Eigenvalues are retained when <tt>value > cutoff</tt>, where \c cutoff is the effective
/// \p rcond — the caller's value, or the default below when that value is zero — times
/// <tt>max|eigenvalue|</tt>, the largest eigenvalue magnitude. The comparison is strict, so a
/// covariance matrix whose largest eigenvalue is zero retains nothing.
///
/// \param validity Row-major mask of the same shape, or null when every value is present.
/// \param rcond Relative eigenvalue cutoff. Zero selects
///        <tt>std::numeric_limits<double>::epsilon() * columns</tt>.
/// \note A covariance entry can be non-finite without any non-finite value in \p values:
///       columns of large magnitude overflow the centred products. \c CovarianceMatrix
///       returns that matrix without complaint, so the rejection surfaces here.
/// \throws std::invalid_argument: When \p values is null and \p rows is non-zero, when
///         \p columns is zero, when <tt>columns * columns</tt> overflows \c std::size_t, when
///         fewer than two complete rows remain, when a covariance entry is not finite, when
///         \p rcond is negative or not finite, when no eigenvalue survives the cutoff, or when
///         a retained eigenvalue's reciprocal overflows.
/// \throws std::runtime_error: When the eigendecomposition does not converge.
DescriptorInverseCovariance InverseCovarianceMatrix(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    double rcond = 0.0);

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_STATISTICS_H
