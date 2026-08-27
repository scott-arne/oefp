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
///        because it is destroyed during rotation.
/// \param dimension Matrix order. Zero yields an empty eigensystem.
/// \return The eigensystem, or \c std::nullopt when the iteration limit is reached without
///         converging.
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

/// \brief Moore-Penrose pseudo-inverse of a real symmetric matrix.
///
/// Eigenvalues are retained if and only if they are **strictly greater** than
/// <tt>rcond * max_eigenvalue</tt>. The strictness matters at the degenerate end: for an
/// all-zero matrix the cutoff is also zero, and a non-strict predicate would invert zeros.
/// A matrix in which nothing survives that cutoff has no usable inverse at all, so this
/// throws rather than returning a rank-zero result for a caller to misread as valid.
///
/// \param matrix Row-major \p dimension x \p dimension symmetric matrix.
/// \param dimension Matrix order. Must be non-zero.
/// \param rcond Relative eigenvalue cutoff. Must be finite and non-negative. Zero selects
///        <tt>std::numeric_limits<double>::epsilon() * dimension</tt>.
/// \return The pseudo-inverse and the number of retained eigenvalues, which is at least one.
/// \throws std::invalid_argument When \p dimension is zero, when \p matrix does not hold
///         exactly <tt>dimension * dimension</tt> entries, when \p rcond is negative or not
///         finite, or when no eigenvalue survives the cutoff.
/// \throws std::runtime_error When the eigendecomposition does not converge.
PseudoInverseResult pseudo_inverse_symmetric(
    const std::vector<double>& matrix,
    std::size_t dimension,
    double rcond);

} // namespace detail
} // namespace OEFP

#endif // OEFP_SRC_LINEAR_ALGEBRA_H
