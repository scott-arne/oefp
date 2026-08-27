#include "linear_algebra.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

namespace OEFP {
namespace detail {
namespace {

// Helper to build a diagonal matrix
std::vector<double> diag(const std::vector<double>& values) {
    const std::size_t n = values.size();
    std::vector<double> matrix(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        matrix[i * n + i] = values[i];
    }
    return matrix;
}

// Helper to compute matrix-matrix product A * B for square matrices
std::vector<double> multiply(const std::vector<double>& a,
                              const std::vector<double>& b,
                              std::size_t n) {
    std::vector<double> result(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                sum += a[i * n + k] * b[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    return result;
}

// Helper to check if a matrix is symmetric
bool is_symmetric(const std::vector<double>& matrix, std::size_t n, double tol) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            if (std::abs(matrix[i * n + j] - matrix[j * n + i]) > tol) {
                return false;
            }
        }
    }
    return true;
}

// Helper to check matrix equality
bool matrices_equal(const std::vector<double>& a,
                   const std::vector<double>& b,
                   double tol) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) {
            return false;
        }
    }
    return true;
}

TEST(PseudoInverseTest, Identity3x3) {
    const auto matrix = diag({1.0, 1.0, 1.0});
    const auto result = pseudo_inverse_symmetric(matrix, 3, 0.0);

    EXPECT_EQ(result.rank, 3u);
    EXPECT_TRUE(matrices_equal(result.matrix, matrix, 1e-12));
}

TEST(PseudoInverseTest, DiagonalWithZero) {
    const auto matrix = diag({4.0, 0.0, 1.0});
    const auto result = pseudo_inverse_symmetric(matrix, 3, 0.0);

    EXPECT_EQ(result.rank, 2u);

    const auto expected = diag({0.25, 0.0, 1.0});
    EXPECT_TRUE(matrices_equal(result.matrix, expected, 1e-12));
}

TEST(PseudoInverseTest, NonDiagonalSPD3x3) {
    // Symmetric positive-definite matrix
    const std::vector<double> matrix = {
        4.0, 1.0, 2.0,
        1.0, 3.0, 0.0,
        2.0, 0.0, 5.0
    };

    const auto result = pseudo_inverse_symmetric(matrix, 3, 0.0);

    EXPECT_EQ(result.rank, 3u);

    // Check A * pinv is identity
    const auto product = multiply(matrix, result.matrix, 3);
    const auto identity = diag({1.0, 1.0, 1.0});
    EXPECT_TRUE(matrices_equal(product, identity, 1e-9));

    // Check result is symmetric
    EXPECT_TRUE(is_symmetric(result.matrix, 3, 1e-9));
}

TEST(PseudoInverseTest, Rank1OuterProduct) {
    // v = {1, 2, 3}, A = v * v^T
    const std::vector<double> v = {1.0, 2.0, 3.0};
    std::vector<double> matrix(9, 0.0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            matrix[i * 3 + j] = v[i] * v[j];
        }
    }

    const auto result = pseudo_inverse_symmetric(matrix, 3, 0.0);

    EXPECT_EQ(result.rank, 1u);

    // pinv = v * v^T / (v · v)^2
    const double dot_product = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];  // 14.0
    const double scale = 1.0 / (dot_product * dot_product);  // 1/196
    std::vector<double> expected(9, 0.0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            expected[i * 3 + j] = v[i] * v[j] * scale;
        }
    }

    EXPECT_TRUE(matrices_equal(result.matrix, expected, 1e-9));
}

TEST(PseudoInverseTest, SPD8x8) {
    // Construct an 8x8 SPD matrix
    const std::size_t n = 8;
    std::vector<double> matrix(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                matrix[i * n + j] = 9.0;
            } else {
                matrix[i * n + j] = 1.0 / (1.0 + i + j);
            }
        }
    }

    const auto result = pseudo_inverse_symmetric(matrix, n, 0.0);

    EXPECT_EQ(result.rank, n);

    // Check A * pinv is identity
    const auto product = multiply(matrix, result.matrix, n);
    const auto identity = diag({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    EXPECT_TRUE(matrices_equal(product, identity, 1e-9));
}

TEST(PseudoInverseTest, MoorePenroseConditionsSPD3x3) {
    const std::vector<double> matrix = {
        4.0, 1.0, 2.0,
        1.0, 3.0, 0.0,
        2.0, 0.0, 5.0
    };

    const auto result = pseudo_inverse_symmetric(matrix, 3, 0.0);
    const auto& pinv = result.matrix;

    // Condition 1: A * A+ * A == A
    const auto a_pinv = multiply(matrix, pinv, 3);
    const auto a_pinv_a = multiply(a_pinv, matrix, 3);
    EXPECT_TRUE(matrices_equal(a_pinv_a, matrix, 1e-9));

    // Condition 2: A+ * A * A+ == A+
    const auto pinv_a = multiply(pinv, matrix, 3);
    const auto pinv_a_pinv = multiply(pinv_a, pinv, 3);
    EXPECT_TRUE(matrices_equal(pinv_a_pinv, pinv, 1e-9));

    // Condition 3: (A * A+) is symmetric
    EXPECT_TRUE(is_symmetric(a_pinv, 3, 1e-9));

    // Condition 4: (A+ * A) is symmetric
    EXPECT_TRUE(is_symmetric(pinv_a, 3, 1e-9));
}

TEST(PseudoInverseTest, MoorePenroseConditionsRank1) {
    // v = {1, 2, 3}, A = v * v^T
    const std::vector<double> v = {1.0, 2.0, 3.0};
    std::vector<double> matrix(9, 0.0);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            matrix[i * 3 + j] = v[i] * v[j];
        }
    }

    const auto result = pseudo_inverse_symmetric(matrix, 3, 0.0);
    const auto& pinv = result.matrix;

    // Condition 1: A * A+ * A == A
    const auto a_pinv = multiply(matrix, pinv, 3);
    const auto a_pinv_a = multiply(a_pinv, matrix, 3);
    EXPECT_TRUE(matrices_equal(a_pinv_a, matrix, 1e-9));

    // Condition 2: A+ * A * A+ == A+
    const auto pinv_a = multiply(pinv, matrix, 3);
    const auto pinv_a_pinv = multiply(pinv_a, pinv, 3);
    EXPECT_TRUE(matrices_equal(pinv_a_pinv, pinv, 1e-9));

    // Condition 3: (A * A+) is symmetric
    EXPECT_TRUE(is_symmetric(a_pinv, 3, 1e-9));

    // Condition 4: (A+ * A) is symmetric
    EXPECT_TRUE(is_symmetric(pinv_a, 3, 1e-9));
}

TEST(PseudoInverseTest, MoorePenroseConditionsSPD8x8) {
    const std::size_t n = 8;
    std::vector<double> matrix(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                matrix[i * n + j] = 9.0;
            } else {
                matrix[i * n + j] = 1.0 / (1.0 + i + j);
            }
        }
    }

    const auto result = pseudo_inverse_symmetric(matrix, n, 0.0);
    const auto& pinv = result.matrix;

    // Condition 1: A * A+ * A == A
    const auto a_pinv = multiply(matrix, pinv, n);
    const auto a_pinv_a = multiply(a_pinv, matrix, n);
    EXPECT_TRUE(matrices_equal(a_pinv_a, matrix, 1e-9));

    // Condition 2: A+ * A * A+ == A+
    const auto pinv_a = multiply(pinv, matrix, n);
    const auto pinv_a_pinv = multiply(pinv_a, pinv, n);
    EXPECT_TRUE(matrices_equal(pinv_a_pinv, pinv, 1e-9));

    // Condition 3: (A * A+) is symmetric
    EXPECT_TRUE(is_symmetric(a_pinv, n, 1e-9));

    // Condition 4: (A+ * A) is symmetric
    EXPECT_TRUE(is_symmetric(pinv_a, n, 1e-9));
}

TEST(PseudoInverseTest, RejectsZeroDimension) {
    const std::vector<double> matrix = {1.0};
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 0, 0.0), std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsSizeMismatch) {
    const std::vector<double> matrix = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 2, 0.0), std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsNegativeRcond) {
    const auto matrix = diag({1.0, 1.0, 1.0});
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 3, -0.1), std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsNaNRcond) {
    const auto matrix = diag({1.0, 1.0, 1.0});
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 3, std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsInfiniteRcond) {
    const auto matrix = diag({1.0, 1.0, 1.0});
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 3, std::numeric_limits<double>::infinity()),
                 std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsAllZeroMatrix) {
    const auto matrix = diag({0.0, 0.0, 0.0});
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 3, 0.0), std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsNaNEntry) {
    const std::vector<double> matrix = {
        2.0, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN(), 2.0
    };
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 2, 0.0), std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsInfiniteEntry) {
    const std::vector<double> matrix = {
        2.0, 1.0,
        1.0, std::numeric_limits<double>::infinity()
    };
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 2, 0.0), std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsDenormalInput) {
    const auto matrix = diag({5e-320, 5e-320});
    EXPECT_THROW(pseudo_inverse_symmetric(matrix, 2, 0.0), std::invalid_argument);
}

TEST(PseudoInverseTest, RejectsOverflowingDimension) {
    // Regression test for the unsigned overflow bug fixed in fix round 2.
    // dimension = 2^32 makes dimension * dimension wrap to 0 on 64-bit size_t,
    // which before the fix would pass the size check with an empty matrix and
    // trigger an out-of-bounds write when the solver initialized the eigenvector buffer.
    const std::vector<double> empty_matrix;
    const std::size_t huge_dimension = std::size_t{1} << 32;
    EXPECT_THROW(pseudo_inverse_symmetric(empty_matrix, huge_dimension, 0.0),
                 std::invalid_argument);
}

TEST(PseudoInverseTest, NegativeEigenvaluesAreDropped) {
    // The header documents that negative eigenvalues are discarded for PSD matrices.
    // This test asserts the documented behavior so it doesn't get "fixed" later.
    const auto matrix = diag({-100.0, 1.0});
    const auto result = pseudo_inverse_symmetric(matrix, 2, 0.0);

    EXPECT_EQ(result.rank, 1u);

    const auto expected = diag({0.0, 1.0});
    EXPECT_TRUE(matrices_equal(result.matrix, expected, 1e-12));
}

TEST(PseudoInverseTest, SmallMagnitudeMatrixKeepsItsRank) {
    // Regression test for the absolute 1.0e-13 off-diagonal threshold the Jacobi sweep
    // converges against. This is 1e-14 * [[1, 2], [2, 4]], exactly singular, and its 2e-14
    // off-diagonal is below that threshold: before pseudo_inverse_symmetric normalized its
    // input, the first sweep declared the matrix already diagonal and the result came back as
    // rank 2 with the whole off-diagonal lost. The true pseudo-inverse is
    // 4e12 * [[1, 2], [2, 4]], which is rank 1 and emphatically not diagonal.
    const std::vector<double> matrix = {1.0e-14, 2.0e-14, 2.0e-14, 4.0e-14};
    const auto result = pseudo_inverse_symmetric(matrix, 2, 0.0);

    EXPECT_EQ(result.rank, 1u);
    ASSERT_NE(result.matrix[1], 0.0);
    EXPECT_DOUBLE_EQ(result.matrix[1], result.matrix[2]);

    const auto expected = std::vector<double>{4.0e12, 8.0e12, 8.0e12, 1.6e13};
    EXPECT_TRUE(matrices_equal(result.matrix, expected, 1.0e1));
}

TEST(PseudoInverseTest, TinyEigenvalueBelowDefaultCutoff) {
    // epsilon() * dimension * max|eigenvalue| is roughly 2.22e-16 * 2 * 1 ≈ 4.4e-16.
    // 1e-18 is well below that, so it should be dropped.
    const auto matrix = diag({1.0, 1e-18});
    const auto result = pseudo_inverse_symmetric(matrix, 2, 0.0);

    EXPECT_EQ(result.rank, 1u);
}

} // namespace
} // namespace detail
} // namespace OEFP
