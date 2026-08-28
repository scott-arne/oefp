#include "linear_algebra.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace OEFP {
namespace detail {
namespace {

/// \brief Apply one Jacobi rotation at (\p p, \p q) to \p matrix and \p vectors.
void apply_jacobi_rotation(std::vector<double>& matrix, std::vector<double>& vectors,
                           std::size_t dimension, std::size_t p, std::size_t q) {
    const auto at = [dimension](std::size_t row, std::size_t column) {
        return row * dimension + column;
    };

    const auto app = matrix[at(p, p)];
    const auto aqq = matrix[at(q, q)];
    const auto apq = matrix[at(p, q)];
    const auto tau = (aqq - app) / (2.0 * apq);
    const auto t = std::copysign(1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau)), tau);
    const auto c = 1.0 / std::sqrt(1.0 + t * t);
    const auto s = t * c;

    matrix[at(p, p)] = app - t * apq;
    matrix[at(q, q)] = aqq + t * apq;
    matrix[at(p, q)] = 0.0;
    matrix[at(q, p)] = 0.0;

    for (std::size_t index = 0u; index < dimension; ++index) {
        if (index == p || index == q) {
            continue;
        }

        const auto aip = matrix[at(index, p)];
        const auto aiq = matrix[at(index, q)];
        const auto rotated_ip = c * aip - s * aiq;
        const auto rotated_iq = s * aip + c * aiq;
        matrix[at(index, p)] = rotated_ip;
        matrix[at(p, index)] = rotated_ip;
        matrix[at(index, q)] = rotated_iq;
        matrix[at(q, index)] = rotated_iq;
    }

    // Mordred/NumPy returns eigenvectors as columns; keep the same layout.
    for (std::size_t row = 0u; row < dimension; ++row) {
        const auto vip = vectors[at(row, p)];
        const auto viq = vectors[at(row, q)];
        vectors[at(row, p)] = c * vip - s * viq;
        vectors[at(row, q)] = s * vip + c * viq;
    }
}

} // namespace

std::optional<SymmetricEigensystem> symmetric_eigensystem_jacobi(
    std::vector<double> matrix,
    std::size_t dimension) {
    SymmetricEigensystem eigensystem;
    if (dimension == 0u) {
        return eigensystem;
    }

    eigensystem.eigenvectors.assign(dimension * dimension, 0.0);
    for (std::size_t index = 0u; index < dimension; ++index) {
        eigensystem.eigenvectors[index * dimension + index] = 1.0;
    }

    if (dimension == 1u) {
        eigensystem.eigenvalues = {matrix.front()};
        return eigensystem;
    }

    constexpr double kTolerance = 1.0e-13;
    const auto at = [dimension](std::size_t row, std::size_t column) {
        return row * dimension + column;
    };

    const auto collect = [&]() {
        eigensystem.eigenvalues.clear();
        eigensystem.eigenvalues.reserve(dimension);
        for (std::size_t index = 0u; index < dimension; ++index) {
            eigensystem.eigenvalues.push_back(matrix[at(index, index)]);
        }
    };

    const auto max_sweeps = std::max<std::size_t>(100u, 20u * dimension);
    for (std::size_t sweep = 0u; sweep < max_sweeps; ++sweep) {
        double max_off_diagonal = 0.0;
        for (std::size_t row = 0u; row < dimension; ++row) {
            for (std::size_t column = row + 1u; column < dimension; ++column) {
                max_off_diagonal = std::max(max_off_diagonal, std::abs(matrix[at(row, column)]));
            }
        }

        if (max_off_diagonal <= kTolerance) {
            collect();
            return eigensystem;
        }

        for (std::size_t row = 0u; row < dimension; ++row) {
            for (std::size_t column = row + 1u; column < dimension; ++column) {
                if (std::abs(matrix[at(row, column)]) <= kTolerance) {
                    continue;
                }
                apply_jacobi_rotation(matrix, eigensystem.eigenvectors, dimension, row, column);
            }
        }
    }

    return std::nullopt;
}

std::optional<SymmetricEigensystem> symmetric_eigensystem_cyclic(
    std::vector<double> matrix,
    std::size_t dimension) {
    return symmetric_eigensystem_jacobi(std::move(matrix), dimension);
}

PseudoInverseResult pseudo_inverse_symmetric(
    const std::vector<double>& matrix,
    std::size_t dimension,
    double rcond) {
    // This function is the sole owner of the rcond and rank contracts. Task 10's
    // InverseCovarianceMatrix documents both to its own callers and adds no checks of its
    // own, so every guard the public API promises has to live here.
    if (dimension == 0u) {
        throw std::invalid_argument("Pseudo-inverse requires a non-empty matrix.");
    }
    // The product is formed before it is compared, so an unchecked multiplication would
    // wrap and defeat the very check it feeds: dimension = 2^32 on a 64-bit size_t makes
    // dimension * dimension exactly zero, an empty matrix then satisfies the size check,
    // and the solver writes identity entries into a zero-length eigenvector buffer.
    if (dimension > std::numeric_limits<std::size_t>::max() / dimension) {
        throw std::invalid_argument("Pseudo-inverse dimension is too large to index.");
    }
    const auto entry_count = dimension * dimension;
    if (matrix.size() != entry_count) {
        throw std::invalid_argument("Pseudo-inverse matrix size does not match its dimension.");
    }
    // Negative and NaN are rejected rather than folded into the default: a caller passing
    // either has a bug, and silently substituting epsilon * dimension would hide it behind a
    // plausible-looking result. Infinity is rejected for the same reason — it would push the
    // cutoff above every eigenvalue and reach the rank-zero throw with a misleading message.
    if (!std::isfinite(rcond) || rcond < 0.0) {
        throw std::invalid_argument("Pseudo-inverse rcond must be finite and non-negative.");
    }

    // A NaN off-diagonal is invisible to the solver's convergence test: std::max(x, NaN)
    // returns x, so a NaN never raises max_off_diagonal, the sweep reports convergence on its
    // first pass, and the input diagonal comes back as the eigenvalues with an identity
    // eigenbasis. The result is finite and plausible and completely wrong. Reject it here for
    // the same reason the rcond guard above rejects a NaN cutoff.
    if (std::any_of(matrix.begin(), matrix.end(),
                    [](double value) { return !std::isfinite(value); })) {
        throw std::invalid_argument("Pseudo-inverse matrix entries must be finite.");
    }

    PseudoInverseResult result;
    result.matrix.assign(entry_count, 0.0);

    // The solver converges on a fixed absolute off-diagonal threshold, so a uniformly tiny
    // matrix would come back undiagonalized: the eigenvalues would be the input diagonal and
    // the eigenbasis the identity. That failure is silent rather than detectable — a positive
    // diagonal is positive definite — and the magnitude of a sample covariance is set by the
    // caller's units, not by anything this function can require. Decompose a copy scaled so
    // its largest entry lands in [0.5, 1) and undo the scale afterwards; eigenvectors are
    // invariant under a uniform scale. The scale is a power of two, so both directions are
    // exact and no retained-or-dropped decision moves. An all-zero matrix has nothing to
    // normalize, and leaving its exponent at zero is what lets it reach the rank-zero throw.
    double max_magnitude = 0.0;
    for (const auto value : matrix) {
        max_magnitude = std::max(max_magnitude, std::abs(value));
    }
    int exponent = 0;
    auto scaled = matrix;
    if (max_magnitude > 0.0) {
        std::frexp(max_magnitude, &exponent);
        for (auto& value : scaled) {
            value = std::ldexp(value, -exponent);
        }
    }

    auto eigensystem = symmetric_eigensystem_cyclic(std::move(scaled), dimension);
    if (!eigensystem.has_value()) {
        throw std::runtime_error("Symmetric eigendecomposition did not converge.");
    }

    const auto effective_rcond = rcond > 0.0
        ? rcond
        : std::numeric_limits<double>::epsilon() * static_cast<double>(dimension);

    // The cutoff and the retention test below both stay in the scaled domain. The cutoff is
    // relative, so the power-of-two scale divides out of the comparison exactly and the
    // verdict is the one the unscaled matrix would have reached.
    double max_eigenvalue = 0.0;
    for (const auto value : eigensystem->eigenvalues) {
        max_eigenvalue = std::max(max_eigenvalue, std::abs(value));
    }
    const auto cutoff = effective_rcond * max_eigenvalue;

    // Strictly greater, deliberately: for an all-zero matrix the cutoff is also zero, and
    // a non-strict predicate would invert zero eigenvalues and return garbage.
    std::vector<double> inverted(dimension, 0.0);
    for (std::size_t k = 0u; k < dimension; ++k) {
        const auto value = eigensystem->eigenvalues[k];
        if (value > cutoff) {
            // Restore the scale on the reciprocal rather than on the eigenvalue: the
            // reassembly needs the reciprocal of the true eigenvalue, and taking it in this
            // order is also what leaves the guard below anything to catch.
            inverted[k] = std::ldexp(1.0 / value, -exponent);
            // A matrix whose true eigenvalues are denormal clears the relative cutoff easily —
            // scaled up it is perfectly well conditioned — but its inverse is not a
            // representable double, so restoring the scale overflows to infinity. The
            // reassembly below would then evaluate 0.0 * inf and hand back an all-NaN matrix
            // with a positive rank.
            if (!std::isfinite(inverted[k])) {
                throw std::invalid_argument(
                    "Pseudo-inverse eigenvalue reciprocal overflowed: the matrix is too close "
                    "to zero for a relative cutoff to be meaningful.");
            }
            ++result.rank;
        }
    }

    // Rank zero means the all-zero matrix, or one so degenerate that nothing clears the
    // cutoff. The pseudo-inverse is then all zeros, which as a Mahalanobis VI would silently
    // report every pair as distance zero. Throwing keeps that from reaching a caller.
    if (result.rank == 0u) {
        throw std::invalid_argument(
            "Pseudo-inverse is rank zero: no positive eigenvalue exceeds the rcond cutoff.");
    }

    // Reassemble Q * inv(Lambda) * Q^T; eigenvectors are columns.
    const auto& q = eigensystem->eigenvectors;
    for (std::size_t row = 0u; row < dimension; ++row) {
        for (std::size_t column = row; column < dimension; ++column) {
            double sum = 0.0;
            for (std::size_t k = 0u; k < dimension; ++k) {
                if (inverted[k] == 0.0) {
                    continue;
                }
                sum += q[row * dimension + k] * inverted[k] * q[column * dimension + k];
            }
            result.matrix[row * dimension + column] = sum;
            result.matrix[column * dimension + row] = sum;
        }
    }

    return result;
}

} // namespace detail
} // namespace OEFP
