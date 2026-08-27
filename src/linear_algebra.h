#ifndef OEFP_SRC_LINEAR_ALGEBRA_H
#define OEFP_SRC_LINEAR_ALGEBRA_H

#include <cstddef>
#include <optional>
#include <vector>

namespace OEFP {
namespace detail {

/// \brief Eigenvalues and eigenvectors of a real symmetric matrix.
///
/// Eigenvectors are stored as columns in row-major order, matching Mordred and NumPy:
/// component \c row of eigenvector \c k is \c eigenvectors[row * dimension + k].
struct SymmetricEigensystem {
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
};

/// \brief Diagonalize a real symmetric matrix by Jacobi rotations.
///
/// \param matrix Row-major \p dimension x \p dimension symmetric matrix, taken by value
///        because it is destroyed during rotation. It must hold exactly
///        <tt>dimension * dimension</tt> entries; this is not checked.
/// \param dimension Matrix order. Zero yields an empty eigensystem.
/// \return The eigensystem, or \c std::nullopt when the iteration limit is reached without
///         converging.
/// \note Convergence is tested against an absolute off-diagonal threshold of 1.0e-13 rather
///       than one scaled to the matrix norm. A matrix whose entries are all of that order or
///       smaller is therefore reported as already diagonal: the eigenvalues come back as the
///       input diagonal and the eigenvectors as the identity. Callers whose input scale is
///       not O(1) should rescale before calling.
std::optional<SymmetricEigensystem> symmetric_eigensystem_jacobi(
    std::vector<double> matrix,
    std::size_t dimension);

/// \brief Diagonalize a real symmetric matrix using the cyclic sweep.
///
/// This is the entry point for code with no bit-for-bit conformance baseline to preserve:
/// the pseudo-inverse below and the Mahalanobis whitening factor. It exists so that those
/// callers never name a pivot, which keeps them compiling unchanged whichever solver
/// branch Task 3 selected. Parameters and return value match
/// \c symmetric_eigensystem_jacobi.
std::optional<SymmetricEigensystem> symmetric_eigensystem_cyclic(
    std::vector<double> matrix,
    std::size_t dimension);

/// \brief Moore-Penrose pseudo-inverse of a real symmetric matrix, and its numerical rank.
struct PseudoInverseResult {
    std::vector<double> matrix;  ///< Row-major, dimension x dimension.
    std::size_t rank = 0;        ///< Count of retained eigenvalues.
};

/// \brief Moore-Penrose pseudo-inverse of a real symmetric positive semi-definite matrix.
///
/// Eigenvalues are retained if and only if they are **strictly greater** than
/// <tt>rcond * max|eigenvalue|</tt>. The strictness matters at the degenerate end: for an
/// all-zero matrix the cutoff is also zero, and a non-strict predicate would invert zeros.
/// A matrix in which nothing survives that cutoff has no usable inverse at all, so this
/// throws rather than returning a rank-zero result for a caller to misread as valid.
///
/// The input is assumed positive semi-definite, which is what a covariance matrix is by
/// construction. The cutoff is built from a magnitude but the retention test is on the signed
/// eigenvalue, so negative eigenvalues are discarded rather than inverted. That is deliberate:
/// it projects away the small negative eigenvalues that floating-point rounding leaves in a
/// sample covariance matrix instead of amplifying them. The consequence is that for a
/// genuinely indefinite matrix the result is the positive semi-definite projected inverse and
/// not the Moore-Penrose pseudo-inverse, so <tt>A * X * A == A</tt> will not hold.
///
/// \param matrix Row-major \p dimension x \p dimension symmetric matrix. Every entry must be
///        finite.
/// \param dimension Matrix order. Must be non-zero.
/// \param rcond Relative eigenvalue cutoff. Must be finite and non-negative. Zero selects
///        <tt>std::numeric_limits<double>::epsilon() * dimension</tt>.
/// \return The pseudo-inverse and the number of retained eigenvalues, which is at least one.
/// \throws std::invalid_argument When \p dimension is zero, when \p dimension is so large
///         that <tt>dimension * dimension</tt> would overflow <tt>std::size_t</tt>, when
///         \p matrix does not hold exactly <tt>dimension * dimension</tt> entries, when any
///         entry of \p matrix is not finite, when \p rcond is negative or not finite, when no
///         eigenvalue survives the cutoff, or when a retained eigenvalue is so close to zero
///         that its reciprocal overflows.
/// \throws std::runtime_error When the eigendecomposition does not converge.
PseudoInverseResult pseudo_inverse_symmetric(
    const std::vector<double>& matrix,
    std::size_t dimension,
    double rcond);

} // namespace detail
} // namespace OEFP

#endif // OEFP_SRC_LINEAR_ALGEBRA_H
