#include <gtest/gtest.h>

#include "oefp/descriptor_compare.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace OEFP {
namespace test {
namespace {

// Two rows, three columns, row-major.
const std::vector<double> kValues = {1.0, 2.0, 3.0, 4.0, 6.0, 3.0};

} // namespace

TEST(DescriptorNumericCompareTest, EuclideanMatchesHandComputation) {
    const auto result = PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Euclidean());
    ASSERT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result[0], 5.0);  // sqrt(9 + 16 + 0)
}

TEST(DescriptorNumericCompareTest, ManhattanChebyshevHammingMatchHandComputation) {
    EXPECT_DOUBLE_EQ(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Manhattan())[0], 7.0);
    EXPECT_DOUBLE_EQ(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Chebyshev())[0], 4.0);
    EXPECT_DOUBLE_EQ(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Hamming())[0],
                     2.0 / 3.0);
}

TEST(DescriptorNumericCompareTest, CanberraAndBrayCurtisMatchHandComputation) {
    // Canberra: 3/5 + 4/8 + 0/6
    EXPECT_DOUBLE_EQ(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Canberra())[0],
                     3.0 / 5.0 + 4.0 / 8.0);
    // BrayCurtis: 7 / (5 + 8 + 6)
    EXPECT_DOUBLE_EQ(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::BrayCurtis())[0],
                     7.0 / 19.0);
}

TEST(DescriptorNumericCompareTest, CDistProducesARectangularMatrix) {
    const std::vector<double> a = {0.0, 0.0};
    const std::vector<double> b = {3.0, 4.0, 6.0, 8.0, 0.0, 0.0};
    const auto result =
        CDistNumeric(a.data(), nullptr, 1u, b.data(), nullptr, 3u, 2u, Metric::Euclidean());
    ASSERT_EQ(result.size(), 3u);
    EXPECT_DOUBLE_EQ(result[0], 5.0);
    EXPECT_DOUBLE_EQ(result[1], 10.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}

TEST(DescriptorNumericCompareTest, CDistAcceptsAMaskOnOnlyOneSide) {
    // `a` is fully present and unmasked; `b` is masked and missing its second column in
    // row 0. Python passes a zero validity address for an all-present side, so this is
    // the ordinary case, not a corner case.
    const std::vector<double> a = {0.0, 0.0};
    const std::vector<double> b = {3.0, 4.0, 3.0, 4.0};
    const std::vector<std::uint8_t> b_validity = {1u, 0u, 1u, 1u};

    const auto propagate =
        CDistNumeric(a.data(), nullptr, 1u, b.data(), b_validity.data(), 2u, 2u,
                     Metric::Euclidean(), DescriptorMissingPolicy::Propagate);
    ASSERT_EQ(propagate.size(), 2u);
    EXPECT_TRUE(std::isnan(propagate[0]));
    EXPECT_DOUBLE_EQ(propagate[1], 5.0);

    // Mirrored: the mask is on the first argument instead. Euclidean is symmetric, so
    // the two results must agree exactly.
    const auto mirrored =
        CDistNumeric(b.data(), b_validity.data(), 2u, a.data(), nullptr, 1u, 2u,
                     Metric::Euclidean(), DescriptorMissingPolicy::Propagate);
    ASSERT_EQ(mirrored.size(), 2u);
    EXPECT_TRUE(std::isnan(mirrored[0]));
    EXPECT_DOUBLE_EQ(mirrored[1], 5.0);

    // Under Ignore, row 0 keeps only column 0 (0 against 3), and the accumulator is
    // rescaled by 2 / 1, so the distance is sqrt(2 * 3^2) = sqrt(18).
    const auto ignore =
        CDistNumeric(a.data(), nullptr, 1u, b.data(), b_validity.data(), 2u, 2u,
                     Metric::Euclidean(), DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(ignore[0], std::sqrt(18.0));
}

TEST(DescriptorNumericCompareTest, PropagateYieldsNaNForPairsTouchingAMissingValue) {
    // Three rows; row 1 is missing its first column.
    const std::vector<double> values = {1.0, 2.0, 0.0, 5.0, 1.0, 2.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 1u, 0u, 1u, 1u};
    const auto result = PDistNumeric(values.data(), validity.data(), 3u, 2u, Metric::Euclidean(),
                                     DescriptorMissingPolicy::Propagate);
    ASSERT_EQ(result.size(), 3u);
    EXPECT_TRUE(std::isnan(result[0]));  // (0, 1)
    EXPECT_DOUBLE_EQ(result[1], 0.0);    // (0, 2)
    EXPECT_TRUE(std::isnan(result[2]));  // (1, 2)
}

TEST(DescriptorNumericCompareTest, IgnoreRescalesTheAccumulatorByTheDimensionRatio) {
    // Two rows, three columns; column 1 is missing in the second row.
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 0.0, 3.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 1u, 1u, 0u, 1u};

    const auto euclidean = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                        Metric::Euclidean(), DescriptorMissingPolicy::Ignore);
    // squared_l2 over the two used columns is 9; rescaled by D/d = 3/2.
    EXPECT_DOUBLE_EQ(euclidean[0], std::sqrt(9.0 * 1.5));

    const auto manhattan = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                        Metric::Manhattan(), DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(manhattan[0], 3.0 * 1.5);

    // Chebyshev, Hamming, and BrayCurtis are not rescaled.
    const auto chebyshev = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                        Metric::Chebyshev(), DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(chebyshev[0], 3.0);
    const auto hamming = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                      Metric::Hamming(), DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(hamming[0], 0.5);  // one unequal of the two used
    const auto braycurtis = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                         Metric::BrayCurtis(), DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(braycurtis[0], 3.0 / 11.0);  // l1 3 over sum_abs (1+4) + (3+3)
}

TEST(DescriptorNumericCompareTest, IgnoreYieldsNaNWhenNoDimensionIsPresentInBoth) {
    const std::vector<double> values = {1.0, 0.0, 0.0, 4.0};
    const std::vector<std::uint8_t> validity = {1u, 0u, 0u, 1u};
    const auto result = PDistNumeric(values.data(), validity.data(), 2u, 2u, Metric::Euclidean(),
                                     DescriptorMissingPolicy::Ignore);
    EXPECT_TRUE(std::isnan(result[0]));
}

TEST(DescriptorNumericCompareTest, ZeroSelectedColumnsYieldNaN) {
    // `values` is empty, so `values.data()` is permitted to be null. That is deliberate: it
    // is exactly what a caller gets from an empty std::vector, and the kernel has to accept
    // it. Nothing dereferences the pointer, because the per-column loop runs zero times, and
    // `values + row * 0` is well defined on a null pointer under C++20 [expr.add]/4.
    const std::vector<double> values;
    for (const auto missing :
         {DescriptorMissingPolicy::Propagate, DescriptorMissingPolicy::Ignore}) {
        const auto pairwise =
            PDistNumeric(values.data(), nullptr, 2u, 0u, Metric::Euclidean(), missing);
        ASSERT_EQ(pairwise.size(), 1u);
        EXPECT_TRUE(std::isnan(pairwise[0]));

        const auto cross = CDistNumeric(values.data(), nullptr, 2u, values.data(), nullptr, 3u,
                                        0u, Metric::Euclidean(), missing);
        ASSERT_EQ(cross.size(), 6u);
        for (const auto value : cross) {
            EXPECT_TRUE(std::isnan(value));
        }
    }
}

TEST(DescriptorNumericCompareTest, EmptyAndSingleRowInputsProduceNoPairs) {
    EXPECT_TRUE(PDistNumeric(nullptr, nullptr, 0u, 3u, Metric::Euclidean()).empty());
    EXPECT_TRUE(PDistNumeric(kValues.data(), nullptr, 1u, 3u, Metric::Euclidean()).empty());
}

TEST(DescriptorNumericCompareTest, IntoFormsValidateTheirOutputLength) {
    std::vector<double> output(2u, 0.0);
    EXPECT_THROW(PDistNumericInto(kValues.data(), nullptr, 2u, 3u, Metric::Euclidean(),
                                  DescriptorMissingPolicy::Propagate, output.data(),
                                  output.size()),
                 std::invalid_argument);
    EXPECT_THROW(CDistNumericInto(kValues.data(), nullptr, 2u, kValues.data(), nullptr, 2u, 3u,
                                  Metric::Euclidean(), DescriptorMissingPolicy::Propagate,
                                  output.data(), output.size()),
                 std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, RejectsMetricsOutsideTheNumericAllowList) {
    EXPECT_THROW(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Jaccard()),
                 std::invalid_argument);
    EXPECT_THROW(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Tanimoto()),
                 std::invalid_argument);
    EXPECT_THROW(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Haversine()),
                 std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, MultiChunkPDistAgainstIndependentOracle) {
    // 20 rows, 3 columns - produces 190 pairs.
    const std::size_t rows = 20u;
    const std::size_t columns = 3u;
    std::vector<double> values(rows * columns);

    // Fill with distinct values so every pair distance is unique. Each element gets a
    // unique value that mixes row and column in a way that breaks distance-translation
    // symmetry.
    for (std::size_t i = 0u; i < rows; ++i) {
        for (std::size_t j = 0u; j < columns; ++j) {
            values[i * columns + j] = static_cast<double>((i * columns + j) * (i + j + 1));
        }
    }

    // Compute with small chunk size and multiple threads.
    BatchKernelOptions kernel;
    kernel.chunk_size = 2u;
    kernel.num_threads = 4u;

    const auto result = PDistNumeric(values.data(), nullptr, rows, columns,
                                     Metric::Euclidean(),
                                     DescriptorMissingPolicy::Propagate, kernel);

    // Independent oracle: nested loop in condensed order.
    const std::size_t expected_pairs = rows * (rows - 1u) / 2u;
    ASSERT_EQ(result.size(), expected_pairs);

    std::vector<double> oracle;
    oracle.reserve(expected_pairs);

    for (std::size_t i = 0u; i < rows; ++i) {
        for (std::size_t j = i + 1u; j < rows; ++j) {
            double sum_sq = 0.0;
            for (std::size_t col = 0u; col < columns; ++col) {
                const double diff = values[i * columns + col] - values[j * columns + col];
                sum_sq += diff * diff;
            }
            oracle.push_back(std::sqrt(sum_sq));
        }
    }

    ASSERT_EQ(oracle.size(), expected_pairs);

    // Verify all values are distinct.
    std::vector<double> sorted_oracle = oracle;
    std::sort(sorted_oracle.begin(), sorted_oracle.end());
    const auto unique_end = std::unique(sorted_oracle.begin(), sorted_oracle.end());
    const std::size_t unique_count = std::distance(sorted_oracle.begin(), unique_end);
    ASSERT_EQ(unique_count, expected_pairs) << "Oracle values are not distinct";

    // Compare kernel output against oracle.
    for (std::size_t k = 0u; k < expected_pairs; ++k) {
        EXPECT_DOUBLE_EQ(result[k], oracle[k]) << "Mismatch at pair " << k;
    }
}

TEST(DescriptorNumericCompareTest, MultiChunkCDistAgainstIndependentOracle) {
    // 18 x 22 produces 396 pairs.
    const std::size_t a_rows = 18u;
    const std::size_t b_rows = 22u;
    const std::size_t columns = 4u;

    std::vector<double> a_values(a_rows * columns);
    std::vector<double> b_values(b_rows * columns);

    // Fill with distinct values.
    for (std::size_t i = 0u; i < a_rows; ++i) {
        for (std::size_t j = 0u; j < columns; ++j) {
            a_values[i * columns + j] = static_cast<double>(i * 11 + j * 17);
        }
    }
    for (std::size_t i = 0u; i < b_rows; ++i) {
        for (std::size_t j = 0u; j < columns; ++j) {
            b_values[i * columns + j] = static_cast<double>(i * 23 + j * 29 + 1000);
        }
    }

    BatchKernelOptions kernel;
    kernel.chunk_size = 1u;
    kernel.num_threads = 4u;

    const auto result = CDistNumeric(a_values.data(), nullptr, a_rows,
                                     b_values.data(), nullptr, b_rows,
                                     columns, Metric::Euclidean(),
                                     DescriptorMissingPolicy::Propagate, kernel);

    const std::size_t expected_pairs = a_rows * b_rows;
    ASSERT_EQ(result.size(), expected_pairs);

    // Independent oracle.
    std::vector<double> oracle;
    oracle.reserve(expected_pairs);

    for (std::size_t i = 0u; i < a_rows; ++i) {
        for (std::size_t j = 0u; j < b_rows; ++j) {
            double sum_sq = 0.0;
            for (std::size_t col = 0u; col < columns; ++col) {
                const double diff = a_values[i * columns + col] - b_values[j * columns + col];
                sum_sq += diff * diff;
            }
            oracle.push_back(std::sqrt(sum_sq));
        }
    }

    ASSERT_EQ(oracle.size(), expected_pairs);

    // Verify distinctness.
    std::vector<double> sorted_oracle = oracle;
    std::sort(sorted_oracle.begin(), sorted_oracle.end());
    const auto unique_end = std::unique(sorted_oracle.begin(), sorted_oracle.end());
    const std::size_t unique_count = std::distance(sorted_oracle.begin(), unique_end);
    ASSERT_EQ(unique_count, expected_pairs) << "Oracle values are not distinct";

    // Compare.
    for (std::size_t k = 0u; k < expected_pairs; ++k) {
        EXPECT_DOUBLE_EQ(result[k], oracle[k]) << "Mismatch at pair " << k;
    }
}

TEST(DescriptorNumericCompareTest, IntoFormsRejectNullOutputWithNonzeroLength) {
    // Null output with correct nonzero length should throw.
    const std::size_t expected_pdist_length = 1u;  // 2 rows -> 1 pair
    EXPECT_THROW(PDistNumericInto(kValues.data(), nullptr, 2u, 3u, Metric::Euclidean(),
                                  DescriptorMissingPolicy::Propagate, nullptr,
                                  expected_pdist_length),
                 std::invalid_argument);

    const std::size_t expected_cdist_length = 4u;  // 2 x 2 = 4
    EXPECT_THROW(CDistNumericInto(kValues.data(), nullptr, 2u, kValues.data(), nullptr, 2u, 3u,
                                  Metric::Euclidean(), DescriptorMissingPolicy::Propagate,
                                  nullptr, expected_cdist_length),
                 std::invalid_argument);

    // Null output with zero length (empty input) should NOT throw.
    EXPECT_NO_THROW(PDistNumericInto(kValues.data(), nullptr, 0u, 3u, Metric::Euclidean(),
                                     DescriptorMissingPolicy::Propagate, nullptr, 0u));
}

} // namespace test
} // namespace OEFP
