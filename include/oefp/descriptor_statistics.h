#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "oefp/descriptor_batch.h"
#include "oefp/descriptor_selection.h"

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
/// \throws std::invalid_argument: When the batch is not schema-backed or a selected column
///         is not a numeric scalar.
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
/// \throws std::invalid_argument: When \p values is null and \p rows is non-zero, or when
///         \p columns is zero.
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
/// \throws std::invalid_argument: When the batch is not schema-backed, a selected column is
///         not a numeric scalar, fewer than two complete rows remain, or a present \c Int
///         value has magnitude strictly greater than 2^53.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an
///         index past the schema's column count.
DescriptorCovariance CovarianceMatrix(
    const DescriptorBatch& batch,
    const DescriptorSelection& columns);

/// \brief Sample covariance for a caller-supplied row-major matrix.
///
/// \throws std::invalid_argument: When \p values is null and \p rows is non-zero, when
///         \p columns is zero, or when fewer than two complete rows remain.
DescriptorCovariance CovarianceMatrix(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns);

/// \brief Moore-Penrose pseudo-inverse of the sample covariance matrix.
///
/// Eigenvalues are retained when <tt>value > rcond * max_eigenvalue</tt>; the comparison is
/// strict, so a covariance matrix whose largest eigenvalue is zero retains nothing.
///
/// \param rcond Relative eigenvalue cutoff. Zero selects
///        <tt>std::numeric_limits<double>::epsilon() * columns</tt>.
/// \throws std::invalid_argument: When the batch is not schema-backed, a selected column is
///         not a numeric scalar, fewer than two complete rows remain, a present \c Int value
///         has magnitude strictly greater than 2^53, \p rcond is negative or not finite, or
///         when no eigenvalue survives the cutoff.
/// \throws std::out_of_range: When the selection contains an unresolvable column name or an
///         index past the schema's column count.
/// \throws std::runtime_error: When the eigendecomposition does not converge.
DescriptorInverseCovariance InverseCovarianceMatrix(
    const DescriptorBatch& batch,
    const DescriptorSelection& columns,
    double rcond = 0.0);

/// \brief Pseudo-inverse of the sample covariance for a caller-supplied row-major matrix.
///
/// \throws std::invalid_argument: When \p values is null and \p rows is non-zero, when
///         \p columns is zero, when fewer than two complete rows remain, when \p rcond is
///         negative or not finite, or when no eigenvalue survives the cutoff.
/// \throws std::runtime_error: When the eigendecomposition does not converge.
DescriptorInverseCovariance InverseCovarianceMatrix(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    double rcond = 0.0);

}  // namespace OEFP
