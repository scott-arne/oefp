#include <gtest/gtest.h>

#include "oefp/descriptor_compare.h"
#include "oefp/descriptor.h"
#include "oefp/descriptor_batch.h"
#include "oefp/descriptor_selection.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {
namespace {

// Two rows, three columns, row-major.
const std::vector<double> kValues = {1.0, 2.0, 3.0, 4.0, 6.0, 3.0};

std::shared_ptr<const DescriptorSchema> numeric_schema() {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float, "test:scalar"});
    builder.Add(DescriptorDefinition{"nAtom", DescriptorValueKind::Int, "test:scalar"});
    builder.Add(DescriptorDefinition{"Lipinski", DescriptorValueKind::Bool, "test:scalar"});
    builder.Add(DescriptorDefinition{"Class", DescriptorValueKind::String, "test:scalar"});
    return builder.Build();
}

DescriptorBatch two_row_batch() {
    const auto schema = numeric_schema();

    DescriptorSetBuilder first(schema);
    first.Set("MW", DescriptorValue::Float(1.0));
    first.Set("nAtom", DescriptorValue::Int(2));
    first.Set("Lipinski", DescriptorValue::Bool(true));

    DescriptorSetBuilder second(schema);
    second.Set("MW", DescriptorValue::Float(4.0));
    second.Set("nAtom", DescriptorValue::Int(6));
    second.Set("Lipinski", DescriptorValue::Bool(false));

    return DescriptorBatch::FromDescriptorSets({first.Build("a"), second.Build("b")});
}

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
    ASSERT_EQ(ignore.size(), 2u);
    EXPECT_DOUBLE_EQ(ignore[0], std::sqrt(18.0));
    EXPECT_DOUBLE_EQ(ignore[1], 5.0);

    // Mirrored Ignore: mask on the first argument. Euclidean is symmetric, so the two
    // results must agree exactly.
    const auto mirrored_ignore =
        CDistNumeric(b.data(), b_validity.data(), 2u, a.data(), nullptr, 1u, 2u,
                     Metric::Euclidean(), DescriptorMissingPolicy::Ignore);
    ASSERT_EQ(mirrored_ignore.size(), 2u);
    EXPECT_DOUBLE_EQ(mirrored_ignore[0], std::sqrt(18.0));
    EXPECT_DOUBLE_EQ(mirrored_ignore[1], 5.0);
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
    const auto canberra = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                       Metric::Canberra(), DescriptorMissingPolicy::Ignore);
    // Canberra over the two used columns: 3/5 + 0/6, rescaled by 3/2.
    EXPECT_DOUBLE_EQ(canberra[0], (3.0 / 5.0) * 1.5);
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

    // CDist with empty sides also produces empty results.
    EXPECT_TRUE(CDistNumeric(nullptr, nullptr, 0u, kValues.data(), nullptr, 2u, 3u,
                             Metric::Euclidean()).empty());
    EXPECT_TRUE(CDistNumeric(kValues.data(), nullptr, 2u, nullptr, nullptr, 0u, 3u,
                             Metric::Euclidean()).empty());
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

TEST(DescriptorNumericCompareTest, AnOversizedChunkSizeStillComputesEveryEntry) {
    // A chunk size near SIZE_MAX used to overflow the ceiling division in ParallelFor,
    // producing zero chunks, zero workers, and an output nobody ever wrote to.
    BatchKernelOptions kernel;
    kernel.chunk_size = std::numeric_limits<std::size_t>::max();

    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 6.0, 3.0, 0.0, 0.0, 0.0};
    const auto pairwise = PDistNumeric(values.data(), nullptr, 3u, 3u, Metric::Euclidean(),
                                       DescriptorMissingPolicy::Propagate, kernel);
    ASSERT_EQ(pairwise.size(), 3u);
    EXPECT_DOUBLE_EQ(pairwise[0], 5.0);              // (0, 1): sqrt(9 + 16 + 0)
    EXPECT_DOUBLE_EQ(pairwise[1], std::sqrt(14.0));  // (0, 2): sqrt(1 + 4 + 9)
    EXPECT_DOUBLE_EQ(pairwise[2], std::sqrt(61.0));  // (1, 2): sqrt(16 + 36 + 9)

    const std::vector<double> origin = {0.0, 0.0, 0.0};
    const auto cross = CDistNumeric(values.data(), nullptr, 3u, origin.data(), nullptr, 1u,
                                    3u, Metric::Euclidean(),
                                    DescriptorMissingPolicy::Propagate, kernel);
    ASSERT_EQ(cross.size(), 3u);
    EXPECT_DOUBLE_EQ(cross[0], std::sqrt(14.0));
    EXPECT_DOUBLE_EQ(cross[1], std::sqrt(61.0));
    EXPECT_DOUBLE_EQ(cross[2], 0.0);
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

TEST(DescriptorNumericCompareTest, CDistCoversAllSixMetrics) {
    // Small 3x4 with 2 columns. Test all six metrics against independent oracles.
    const std::vector<double> a_values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::vector<double> b_values = {2.0, 3.0, 0.0, 0.0, 1.0, 1.0, 4.0, 5.0};

    const std::size_t a_rows = 3u;
    const std::size_t b_rows = 4u;
    const std::size_t columns = 2u;
    const std::size_t expected_size = a_rows * b_rows;

    // Euclidean: sqrt(sum((a - b)^2))
    {
        const auto result = CDistNumeric(a_values.data(), nullptr, a_rows,
                                         b_values.data(), nullptr, b_rows,
                                         columns, Metric::Euclidean());
        ASSERT_EQ(result.size(), expected_size);
        // a[0] = {1, 2}, b[0] = {2, 3}: sqrt(1 + 1) = sqrt(2)
        EXPECT_DOUBLE_EQ(result[0], std::sqrt(2.0));
        // a[0] = {1, 2}, b[1] = {0, 0}: sqrt(1 + 4) = sqrt(5)
        EXPECT_DOUBLE_EQ(result[1], std::sqrt(5.0));
        // a[0] = {1, 2}, b[2] = {1, 1}: sqrt(0 + 1) = 1
        EXPECT_DOUBLE_EQ(result[2], 1.0);
        // a[0] = {1, 2}, b[3] = {4, 5}: sqrt(9 + 9) = sqrt(18)
        EXPECT_DOUBLE_EQ(result[3], std::sqrt(18.0));
        // a[1] = {3, 4}, b[0] = {2, 3}: sqrt(1 + 1) = sqrt(2)
        EXPECT_DOUBLE_EQ(result[4], std::sqrt(2.0));
        // a[1] = {3, 4}, b[1] = {0, 0}: sqrt(9 + 16) = 5
        EXPECT_DOUBLE_EQ(result[5], 5.0);
        // a[1] = {3, 4}, b[2] = {1, 1}: sqrt(4 + 9) = sqrt(13)
        EXPECT_DOUBLE_EQ(result[6], std::sqrt(13.0));
        // a[1] = {3, 4}, b[3] = {4, 5}: sqrt(1 + 1) = sqrt(2)
        EXPECT_DOUBLE_EQ(result[7], std::sqrt(2.0));
        // a[2] = {5, 6}, b[0] = {2, 3}: sqrt(9 + 9) = sqrt(18)
        EXPECT_DOUBLE_EQ(result[8], std::sqrt(18.0));
        // a[2] = {5, 6}, b[1] = {0, 0}: sqrt(25 + 36) = sqrt(61)
        EXPECT_DOUBLE_EQ(result[9], std::sqrt(61.0));
        // a[2] = {5, 6}, b[2] = {1, 1}: sqrt(16 + 25) = sqrt(41)
        EXPECT_DOUBLE_EQ(result[10], std::sqrt(41.0));
        // a[2] = {5, 6}, b[3] = {4, 5}: sqrt(1 + 1) = sqrt(2)
        EXPECT_DOUBLE_EQ(result[11], std::sqrt(2.0));
    }

    // Manhattan: sum(|a - b|)
    {
        const auto result = CDistNumeric(a_values.data(), nullptr, a_rows,
                                         b_values.data(), nullptr, b_rows,
                                         columns, Metric::Manhattan());
        ASSERT_EQ(result.size(), expected_size);
        // a[0] = {1, 2}, b[0] = {2, 3}: |1-2| + |2-3| = 2
        EXPECT_DOUBLE_EQ(result[0], 2.0);
        // a[0] = {1, 2}, b[1] = {0, 0}: 1 + 2 = 3
        EXPECT_DOUBLE_EQ(result[1], 3.0);
        // a[0] = {1, 2}, b[2] = {1, 1}: 0 + 1 = 1
        EXPECT_DOUBLE_EQ(result[2], 1.0);
        // a[0] = {1, 2}, b[3] = {4, 5}: 3 + 3 = 6
        EXPECT_DOUBLE_EQ(result[3], 6.0);
    }

    // Chebyshev: max(|a - b|)
    {
        const auto result = CDistNumeric(a_values.data(), nullptr, a_rows,
                                         b_values.data(), nullptr, b_rows,
                                         columns, Metric::Chebyshev());
        ASSERT_EQ(result.size(), expected_size);
        // a[0] = {1, 2}, b[0] = {2, 3}: max(1, 1) = 1
        EXPECT_DOUBLE_EQ(result[0], 1.0);
        // a[0] = {1, 2}, b[1] = {0, 0}: max(1, 2) = 2
        EXPECT_DOUBLE_EQ(result[1], 2.0);
        // a[1] = {3, 4}, b[1] = {0, 0}: max(3, 4) = 4
        EXPECT_DOUBLE_EQ(result[5], 4.0);
    }

    // Hamming: (number of different dimensions) / (total dimensions)
    {
        const auto result = CDistNumeric(a_values.data(), nullptr, a_rows,
                                         b_values.data(), nullptr, b_rows,
                                         columns, Metric::Hamming());
        ASSERT_EQ(result.size(), expected_size);
        // a[0] = {1, 2}, b[0] = {2, 3}: 2 different / 2 = 1.0
        EXPECT_DOUBLE_EQ(result[0], 1.0);
        // a[0] = {1, 2}, b[2] = {1, 1}: 1 different / 2 = 0.5
        EXPECT_DOUBLE_EQ(result[2], 0.5);
    }

    // Canberra: sum(|a - b| / (|a| + |b|)) for each dimension
    {
        const auto result = CDistNumeric(a_values.data(), nullptr, a_rows,
                                         b_values.data(), nullptr, b_rows,
                                         columns, Metric::Canberra());
        ASSERT_EQ(result.size(), expected_size);
        // a[0] = {1, 2}, b[0] = {2, 3}: |1-2|/(1+2) + |2-3|/(2+3) = 1/3 + 1/5
        EXPECT_DOUBLE_EQ(result[0], 1.0 / 3.0 + 1.0 / 5.0);
        // a[0] = {1, 2}, b[1] = {0, 0}: 1/(1+0) + 2/(2+0) = 1 + 1 = 2
        EXPECT_DOUBLE_EQ(result[1], 2.0);
    }

    // BrayCurtis: sum(|a - b|) / sum(|a| + |b|)
    {
        const auto result = CDistNumeric(a_values.data(), nullptr, a_rows,
                                         b_values.data(), nullptr, b_rows,
                                         columns, Metric::BrayCurtis());
        ASSERT_EQ(result.size(), expected_size);
        // a[0] = {1, 2}, b[0] = {2, 3}: (1 + 1) / (3 + 5) = 2 / 8 = 0.25
        EXPECT_DOUBLE_EQ(result[0], 0.25);
        // a[0] = {1, 2}, b[1] = {0, 0}: (1 + 2) / (1 + 2) = 3 / 3 = 1.0
        EXPECT_DOUBLE_EQ(result[1], 1.0);
        // a[1] = {3, 4}, b[2] = {1, 1}: (2 + 3) / (4 + 5) = 5 / 9
        EXPECT_DOUBLE_EQ(result[6], 5.0 / 9.0);
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

    // CDist with empty sides also accepts null output with zero length.
    EXPECT_NO_THROW(CDistNumericInto(kValues.data(), nullptr, 0u, kValues.data(), nullptr, 2u, 3u,
                                     Metric::Euclidean(), DescriptorMissingPolicy::Propagate,
                                     nullptr, 0u));
    EXPECT_NO_THROW(CDistNumericInto(kValues.data(), nullptr, 2u, kValues.data(), nullptr, 0u, 3u,
                                     Metric::Euclidean(), DescriptorMissingPolicy::Propagate,
                                     nullptr, 0u));
}

TEST(DescriptorNumericCompareTest, UnweightedMinkowskiMatchesHandComputation) {
    // Differences are {-3, -4, 0}; p = 3 gives (27 + 64) ^ (1/3).
    const auto result = PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Minkowski(3.0));
    EXPECT_DOUBLE_EQ(result[0], std::pow(91.0, 1.0 / 3.0));
}

TEST(DescriptorNumericCompareTest, UnweightedMinkowskiAgreesWithItsSpecialCases) {
    EXPECT_DOUBLE_EQ(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Minkowski(1.0))[0],
                     7.0);
    EXPECT_DOUBLE_EQ(PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Minkowski(2.0))[0],
                     5.0);

    // Under Ignore with a dropped column, the p=1 and p=2 fast paths must still rescale.
    // Column 1 is missing in the second row, so d = 2 (columns 0 and 2).
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 0.0, 3.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 1u, 1u, 0u, 1u};

    const auto manhattan = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                        Metric::Manhattan(), DescriptorMissingPolicy::Ignore);
    const auto minkowski_p1 = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                           Metric::Minkowski(1.0), DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(minkowski_p1[0], manhattan[0]);

    const auto euclidean = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                        Metric::Euclidean(), DescriptorMissingPolicy::Ignore);
    const auto minkowski_p2 = PDistNumeric(values.data(), validity.data(), 2u, 3u,
                                           Metric::Minkowski(2.0), DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(minkowski_p2[0], euclidean[0]);
}

TEST(DescriptorNumericCompareTest, WeightedMinkowskiMatchesHandComputation) {
    const auto metric = Metric::Minkowski(2.0, {1.0, 2.0, 4.0});
    const auto result = PDistNumeric(kValues.data(), nullptr, 2u, 3u, metric);
    EXPECT_DOUBLE_EQ(result[0], std::sqrt(1.0 * 9.0 + 2.0 * 16.0 + 4.0 * 0.0));
}

TEST(DescriptorNumericCompareTest, WeightedMinkowskiValidatesAgainstTheColumnCount) {
    const auto metric = Metric::Minkowski(2.0, {1.0, 2.0});
    EXPECT_THROW(PDistNumeric(kValues.data(), nullptr, 2u, 3u, metric), std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, IgnoreUsesWeightMassNotDimensionCountForWeightedMinkowski) {
    // Column 2 is missing in the second row. D = 3, d = 2.
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 6.0, 0.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 1u, 1u, 1u, 0u};
    const auto metric = Metric::Minkowski(2.0, {1.0, 2.0, 7.0});

    const auto result = PDistNumeric(values.data(), validity.data(), 2u, 3u, metric,
                                     DescriptorMissingPolicy::Ignore);

    // power_sum over the used columns is 1*9 + 2*16 = 41.
    // Weight-mass factor is 10 / 3; the count factor 3 / 2 would give a different answer.
    EXPECT_DOUBLE_EQ(result[0], std::sqrt(41.0 * (10.0 / 3.0)));
    EXPECT_NE(result[0], std::sqrt(41.0 * 1.5));
}

TEST(DescriptorNumericCompareTest, IgnoreKeepsWeightsBoundToTheirOriginalColumnIndex) {
    // Column 0 is missing in the second row, so the used columns are 1 and 2.
    const std::vector<double> values = {1.0, 2.0, 3.0, 0.0, 6.0, 3.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 1u, 0u, 1u, 1u};
    const auto metric = Metric::Minkowski(1.0, {1.0, 10.0, 3.0});

    const auto result = PDistNumeric(values.data(), validity.data(), 2u, 3u, metric,
                                     DescriptorMissingPolicy::Ignore);

    // Correct: column 1 keeps weight 10 and column 2 keeps weight 3.
    // power_sum = 10 * 4 + 3 * 0 = 40; factor = 14 / 13.
    EXPECT_DOUBLE_EQ(result[0], 40.0 * (14.0 / 13.0));
    // If the weights had shifted down to fill the gap the answer would be 4 * 14 / 11.
    EXPECT_NE(result[0], 4.0 * (14.0 / 11.0));
}

TEST(DescriptorNumericCompareTest, IgnoreYieldsNaNWhenTheUsedWeightMassIsZero) {
    // Column 2 is missing, so only the two zero-weight columns are used.
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 6.0, 0.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 1u, 1u, 1u, 0u};
    const auto metric = Metric::Minkowski(2.0, {0.0, 0.0, 5.0});

    const auto result = PDistNumeric(values.data(), validity.data(), 2u, 3u, metric,
                                     DescriptorMissingPolicy::Ignore);
    EXPECT_TRUE(std::isnan(result[0]));
}

TEST(DescriptorNumericCompareTest, PropagateYieldsNaNForMinkowskiWithMissingValue) {
    // Exercises <Propagate, true, true>: powered path with missing values and NaN result.
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 0.0, 3.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 1u, 1u, 0u, 1u};
    const auto result = PDistNumeric(values.data(), validity.data(), 2u, 3u, Metric::Minkowski(3.0),
                                     DescriptorMissingPolicy::Propagate);
    EXPECT_TRUE(std::isnan(result[0]));
}

TEST(DescriptorNumericCompareTest, IgnoreRescalesUnweightedMinkowskiWithDroppedColumn) {
    // Exercises <Ignore, true, true>: powered unweighted Minkowski with rescale and mask.
    // Column 1 is missing in the second row, so d = 2 (columns 0 and 2).
    const std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 0.0, 3.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 1u, 1u, 0u, 1u};

    const auto result = PDistNumeric(values.data(), validity.data(), 2u, 3u, Metric::Minkowski(3.0),
                                     DescriptorMissingPolicy::Ignore);

    // power_sum over the two used columns: 3^3 + 0^3 = 27; factor = D/d = 3/2.
    EXPECT_DOUBLE_EQ(result[0], std::pow(27.0 * 1.5, 1.0 / 3.0));
}

TEST(DescriptorNumericCompareTest, IgnoreWithFullyPresentUnweightedMinkowski) {
    // Exercises <Ignore, false, true>: powered unweighted Minkowski, no mask.
    // With no mask, nothing is dropped so the factor is 1.0 and the result is the plain
    // unweighted Minkowski, but this path must still be instantiated correctly.
    const auto result = PDistNumeric(kValues.data(), nullptr, 2u, 3u, Metric::Minkowski(3.0),
                                     DescriptorMissingPolicy::Ignore);

    // Differences are {-3, -4, 0}; (27 + 64) ^ (1/3), no rescale.
    EXPECT_DOUBLE_EQ(result[0], std::pow(91.0, 1.0 / 3.0));
}

TEST(DescriptorNumericCompareTest, CDistUnweightedMinkowskiMatchesHandComputation) {
    // Exercises Minkowski on the CDist path.
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {4.0, 6.0, 3.0};
    const auto result = CDistNumeric(a.data(), nullptr, 1u, b.data(), nullptr, 1u, 3u,
                                     Metric::Minkowski(3.0));
    ASSERT_EQ(result.size(), 1u);
    // Differences are {-3, -4, 0}; (27 + 64) ^ (1/3).
    EXPECT_DOUBLE_EQ(result[0], std::pow(91.0, 1.0 / 3.0));
}

TEST(DescriptorNumericCompareTest, CDistWeightedMinkowskiUnderIgnoreReadsWeightMass) {
    // Exercises CDist's total_weight_mass plumbing: weighted Minkowski under Ignore with a mask.
    // Row a[0] is {1, 2, 3}, row b[0] is {4, 6, 0} with column 2 missing.
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {4.0, 6.0, 0.0};
    const std::vector<std::uint8_t> b_validity = {1u, 1u, 0u};
    const auto metric = Metric::Minkowski(2.0, {1.0, 2.0, 7.0});

    const auto result = CDistNumeric(a.data(), nullptr, 1u, b.data(), b_validity.data(), 1u, 3u,
                                     metric, DescriptorMissingPolicy::Ignore);
    ASSERT_EQ(result.size(), 1u);

    // Column 2 is missing in b[0], so only columns 0 and 1 are used.
    // power_sum = 1 * (4-1)^2 + 2 * (6-2)^2 = 1*9 + 2*16 = 41.
    // total_weight_mass = 1 + 2 + 7 = 10; used_weight_mass = 1 + 2 = 3; factor = 10/3.
    EXPECT_DOUBLE_EQ(result[0], std::sqrt(41.0 * (10.0 / 3.0)));
}

TEST(DescriptorNumericCompareTest, IgnoreAndPropagateAgreeOnAllPresentWithZeroWeights) {
    // G1: when nothing is missing, the two policies must agree even when every weight is zero.
    const auto metric = Metric::Minkowski(2.0, {0.0, 0.0, 0.0});

    // Null mask: both policies return 0.0.
    const auto propagate_null = PDistNumeric(kValues.data(), nullptr, 2u, 3u, metric,
                                             DescriptorMissingPolicy::Propagate);
    const auto ignore_null = PDistNumeric(kValues.data(), nullptr, 2u, 3u, metric,
                                          DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(propagate_null[0], 0.0);
    EXPECT_DOUBLE_EQ(ignore_null[0], 0.0);
    EXPECT_DOUBLE_EQ(ignore_null[0], propagate_null[0]);

    // Non-null all-ones mask: both policies return 0.0.
    const std::vector<std::uint8_t> all_present = {1u, 1u, 1u, 1u, 1u, 1u};
    const auto propagate_mask = PDistNumeric(kValues.data(), all_present.data(), 2u, 3u, metric,
                                             DescriptorMissingPolicy::Propagate);
    const auto ignore_mask = PDistNumeric(kValues.data(), all_present.data(), 2u, 3u, metric,
                                          DescriptorMissingPolicy::Ignore);
    EXPECT_DOUBLE_EQ(propagate_mask[0], 0.0);
    EXPECT_DOUBLE_EQ(ignore_mask[0], 0.0);
    EXPECT_DOUBLE_EQ(ignore_mask[0], propagate_mask[0]);
}

TEST(DescriptorNumericCompareTest, PropagateWithAllPresentMaskAndPoweredMinkowski) {
    // G2: <Propagate, true, true> must run the powered accumulation to completion, not only
    // reach the early NaN return.
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {4.0, 6.0, 7.0};
    const std::vector<std::uint8_t> all_present = {1u, 1u, 1u};

    const auto result = CDistNumeric(a.data(), all_present.data(), 1u, b.data(),
                                     all_present.data(), 1u, 3u,
                                     Metric::Minkowski(3.0), DescriptorMissingPolicy::Propagate);
    ASSERT_EQ(result.size(), 1u);
    // Differences: {-3, -4, -4}; 3^3 + 4^3 + 4^3 = 27 + 64 + 64 = 155.
    EXPECT_DOUBLE_EQ(result[0], std::pow(155.0, 1.0 / 3.0));
}

TEST(DescriptorNumericCompareTest, WeightedPropagateWithAllPresentMaskAndNonzeroFinalColumn) {
    // G3: weighted powered accumulation with final column contributing.
    const std::vector<double> a = {1.0, 2.0, 3.0};
    const std::vector<double> b = {4.0, 6.0, 7.0};
    const std::vector<std::uint8_t> all_present = {1u, 1u, 1u};

    const auto result = CDistNumeric(a.data(), all_present.data(), 1u, b.data(),
                                     all_present.data(), 1u, 3u,
                                     Metric::Minkowski(2.0, {1.0, 2.0, 3.0}),
                                     DescriptorMissingPolicy::Propagate);
    ASSERT_EQ(result.size(), 1u);
    // power_sum = 1*9 + 2*16 + 3*16 = 9 + 32 + 48 = 89.
    EXPECT_DOUBLE_EQ(result[0], std::sqrt(89.0));
}

TEST(DescriptorNumericCompareTest, CDistValidatesWeightedMinkowskiAndRejectsNonNumericMetrics) {
    // G4: CDist must validate weighted Minkowski's weights length and reject non-numeric metrics.
    const auto metric = Metric::Minkowski(2.0, {1.0, 2.0});

    // CDist validation (WeightedMinkowskiValidatesAgainstTheColumnCount already covers PDist).
    EXPECT_THROW(CDistNumeric(kValues.data(), nullptr, 2u, kValues.data(), nullptr, 2u, 3u,
                              metric),
                 std::invalid_argument);

    // Also verify CDist rejects non-numeric metrics (RejectsMetricsOutsideTheNumericAllowList
    // only tests PDist).
    EXPECT_THROW(CDistNumeric(kValues.data(), nullptr, 2u, kValues.data(), nullptr, 2u, 3u,
                              Metric::Jaccard()),
                 std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, StandardizedEuclideanWithUnitVariancesEqualsEuclidean) {
    const auto metric = Metric::StandardizedEuclidean({1.0, 1.0, 1.0});
    const auto result = PDistNumeric(kValues.data(), nullptr, 2u, 3u, metric);
    EXPECT_DOUBLE_EQ(result[0], 5.0);
}

TEST(DescriptorNumericCompareTest, StandardizedEuclideanScalesByTheSuppliedVariances) {
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    const auto metric = Metric::StandardizedEuclidean({4.0, 9.0});
    const auto result = PDistNumeric(values.data(), nullptr, 2u, 2u, metric);
    EXPECT_DOUBLE_EQ(result[0], std::sqrt(9.0 / 4.0 + 16.0 / 9.0));
}

TEST(DescriptorNumericCompareTest, MahalanobisWithIdentityInverseCovarianceEqualsEuclidean) {
    const auto metric = Metric::Mahalanobis({1.0, 0.0, 0.0, 1.0});
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    const auto result = PDistNumeric(values.data(), nullptr, 2u, 2u, metric);
    EXPECT_DOUBLE_EQ(result[0], 5.0);
}

TEST(DescriptorNumericCompareTest, MahalanobisMatchesADirectQuadraticForm) {
    // d = {-3, -4}; d' * VI * d = 2*9 + 2*(0.5*12) + 1*16 = 46.
    const auto metric = Metric::Mahalanobis({2.0, 0.5, 0.5, 1.0});
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    const auto result = PDistNumeric(values.data(), nullptr, 2u, 2u, metric);
    EXPECT_NEAR(result[0], std::sqrt(46.0), 1.0e-12);
}

TEST(DescriptorNumericCompareTest, PreTransformPropagatesMissingValuesAtRowGranularity) {
    // Three rows, two columns; row 1 is missing its first column.
    const std::vector<double> values = {1.0, 2.0, 0.0, 5.0, 1.0, 2.0};
    const std::vector<std::uint8_t> validity = {1u, 1u, 0u, 1u, 1u, 1u};

    for (const auto& metric : {Metric::StandardizedEuclidean({1.0, 1.0}),
                               Metric::Mahalanobis({1.0, 0.0, 0.0, 1.0})}) {
        const auto result = PDistNumeric(values.data(), validity.data(), 3u, 2u, metric);
        ASSERT_EQ(result.size(), 3u);
        EXPECT_TRUE(std::isnan(result[0]));  // (0, 1)
        EXPECT_DOUBLE_EQ(result[1], 0.0);    // (0, 2)
        EXPECT_TRUE(std::isnan(result[2]));  // (1, 2)
    }
}

TEST(DescriptorNumericCompareTest, StandardizedEuclideanRejectsEveryBadVarianceKind) {
    const auto infinity = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto bad : {0.0, -1.0, infinity, nan}) {
        const auto metric = Metric::StandardizedEuclidean({1.0, bad});
        const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
        EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u, metric),
                     std::invalid_argument);
    }
}

TEST(DescriptorNumericCompareTest, PreTransformMetricsValidateTheirParameterSizes) {
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u,
                              Metric::StandardizedEuclidean({1.0})),
                 std::invalid_argument);
    EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u,
                              Metric::Mahalanobis({1.0, 0.0, 0.0})),
                 std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, MahalanobisRejectsANonPositiveSemidefiniteMatrix) {
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    // The easy case: a negative diagonal entry, which is also a negative eigenvalue.
    EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u,
                              Metric::Mahalanobis({1.0, 0.0, 0.0, -1.0})),
                 std::invalid_argument);
    // Indefinite with a strictly positive diagonal: the eigenvalues are 3 and -1. A guard that
    // scanned the raw diagonal would accept this and clamp the negative direction to zero,
    // silently substituting a degenerate metric for the one the caller asked for.
    EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u,
                              Metric::Mahalanobis({1.0, 2.0, 2.0, 1.0})),
                 std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, PreTransformMetricsRejectTheIgnorePolicy) {
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u,
                              Metric::StandardizedEuclidean({1.0, 1.0}),
                              DescriptorMissingPolicy::Ignore),
                 std::invalid_argument);
    EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u,
                              Metric::Mahalanobis({1.0, 0.0, 0.0, 1.0}),
                              DescriptorMissingPolicy::Ignore),
                 std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, CDistPreTransformMetricsRejectTheIgnorePolicy) {
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    EXPECT_THROW(CDistNumeric(values.data(), nullptr, 1u, values.data(), nullptr, 1u, 2u,
                              Metric::StandardizedEuclidean({1.0, 1.0}),
                              DescriptorMissingPolicy::Ignore),
                 std::invalid_argument);
    EXPECT_THROW(CDistNumeric(values.data(), nullptr, 1u, values.data(), nullptr, 1u, 2u,
                              Metric::Mahalanobis({1.0, 0.0, 0.0, 1.0}),
                              DescriptorMissingPolicy::Ignore),
                 std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, CDistMahalanobisWithDistinctMatrices) {
    // Catches mutations: whitening only one matrix, or falling through to Euclidean.
    const auto metric = Metric::Mahalanobis({2.0, 0.5, 0.5, 1.0});
    const std::vector<double> a_values = {1.0, 2.0};
    const std::vector<double> b_values = {4.0, 6.0, 1.0, 2.0};
    const auto result = CDistNumeric(a_values.data(), nullptr, 1u,
                                     b_values.data(), nullptr, 2u, 2u, metric);
    ASSERT_EQ(result.size(), 2u);
    // (0, 0): d = {-3, -4}; d' * VI * d = 2*9 + 2*(0.5*12) + 1*16 = 46.
    EXPECT_NEAR(result[0], std::sqrt(46.0), 1.0e-12);
    // (0, 1): d = {0, 0}; distance is 0.
    EXPECT_DOUBLE_EQ(result[1], 0.0);
}

TEST(DescriptorNumericCompareTest, CDistPreTransformPropagatesMissingValuesFromBothSides) {
    // Catches both halves of the guard independently: a[1] is incomplete and b[1] is incomplete,
    // so result[1] pins the B side and result[2] pins the A side.
    const std::vector<double> a_values = {1.0, 2.0, 1.0, 2.0};
    const std::vector<double> b_values = {1.0, 2.0, 0.0, 5.0};
    const std::vector<std::uint8_t> a_validity = {1u, 1u, 0u, 1u};
    const std::vector<std::uint8_t> b_validity = {1u, 1u, 0u, 1u};

    for (const auto& metric : {Metric::StandardizedEuclidean({1.0, 1.0}),
                               Metric::Mahalanobis({1.0, 0.0, 0.0, 1.0})}) {
        const auto result = CDistNumeric(a_values.data(), a_validity.data(), 2u,
                                         b_values.data(), b_validity.data(), 2u, 2u, metric);
        ASSERT_EQ(result.size(), 4u);
        // a[0] vs b[0]: both complete, distance is 0.
        EXPECT_DOUBLE_EQ(result[0], 0.0);
        // a[0] vs b[1]: b[1] incomplete, NaN.
        EXPECT_TRUE(std::isnan(result[1]));
        // a[1] vs b[0]: a[1] incomplete, NaN.
        EXPECT_TRUE(std::isnan(result[2]));
        // a[1] vs b[1]: both incomplete, NaN.
        EXPECT_TRUE(std::isnan(result[3]));
    }
}

TEST(DescriptorNumericCompareTest, MahalanobisToleratesARankDeficientInverseCovariance) {
    // VI = v v' for v = {1, 2, 3} is rank one, and the solver returns its zero eigenvalues as
    // {0, -2.4e-16}. The negative one sits inside the tolerance band, so it is clamped to zero
    // rather than rejected, and the distance reduces to the absolute value of v . d.
    const auto metric = Metric::Mahalanobis({1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0});
    const std::vector<double> values = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    const auto result = PDistNumeric(values.data(), nullptr, 2u, 3u, metric);
    EXPECT_NEAR(result[0], 6.0, 1.0e-12);
}

TEST(DescriptorNumericCompareTest, MahalanobisRejectsAColumnCountWhoseSquareOverflows) {
    // columns = 2^32 wraps columns * columns to exactly zero on a 64-bit size_t, so an empty
    // inverse covariance would satisfy the size check and the eigensolver would write identity
    // entries into a zero-length buffer. rows = 1 keeps the condensed length at zero, so no
    // descriptor value is ever read and no large allocation is needed to reach the guard.
    if (sizeof(std::size_t) < 8u) {
        GTEST_SKIP() << "The wrap requires a 64-bit size_t.";
    }
    const std::size_t columns = std::size_t{1} << 32;
    const double value = 0.0;
    EXPECT_THROW(PDistNumeric(&value, nullptr, 1u, columns, Metric::Mahalanobis({})),
                 std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, MahalanobisRejectsNonFiniteInverseCovarianceEntries) {
    // A NaN off-diagonal never raises the solver's max off-diagonal, so the sweep converges
    // immediately and the input diagonal comes back with an identity eigenbasis — a finite,
    // plausible, wrong answer. An infinite entry instead makes the tolerance band infinite,
    // which disables the positive-semidefinite rejection entirely.
    const auto infinity = std::numeric_limits<double>::infinity();
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    for (const auto bad : {infinity, -infinity, nan}) {
        const auto metric = Metric::Mahalanobis({1.0, bad, bad, 1.0});
        EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u, metric),
                     std::invalid_argument);
    }
}

TEST(DescriptorNumericCompareTest, MahalanobisIsInvariantToTheScaleOfTheInverseCovariance) {
    // The solver's off-diagonal convergence threshold is an absolute 1e-13, so without the
    // power-of-two rescale a uniformly tiny inverse covariance comes back undiagonalized and the
    // off-diagonal term is silently dropped. Scaling s * VI scales the quadratic form by s, so
    // the distance must scale by sqrt(s) exactly.
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    const auto unit = Metric::Mahalanobis({2.0, 0.5, 0.5, 1.0});
    const auto tiny = Metric::Mahalanobis({2.0e-14, 0.5e-14, 0.5e-14, 1.0e-14});
    const auto unit_result = PDistNumeric(values.data(), nullptr, 2u, 2u, unit);
    const auto tiny_result = PDistNumeric(values.data(), nullptr, 2u, 2u, tiny);
    EXPECT_NEAR(unit_result[0], std::sqrt(46.0), 1.0e-12);
    const auto expected = unit_result[0] * std::sqrt(1.0e-14);
    EXPECT_NEAR(tiny_result[0], expected, std::abs(expected) * 1.0e-12);
}

TEST(DescriptorNumericCompareTest, CDistStandardizedEuclideanScalesByTheSuppliedVariances) {
    // Standardized Euclidean reaches CDist elsewhere only with unit variances and zero
    // distances, so a cdist-only mutation substituting an identity factor survives the suite.
    const auto metric = Metric::StandardizedEuclidean({4.0, 9.0});
    const std::vector<double> a_values = {1.0, 2.0};
    const std::vector<double> b_values = {4.0, 6.0, 1.0, 2.0};
    const auto result = CDistNumeric(a_values.data(), nullptr, 1u,
                                     b_values.data(), nullptr, 2u, 2u, metric);
    ASSERT_EQ(result.size(), 2u);
    // (0, 0): d = {-3, -4}; sqrt(9 / 4 + 16 / 9).
    EXPECT_NEAR(result[0], std::sqrt(9.0 / 4.0 + 16.0 / 9.0), 1.0e-12);
    // (0, 1): d = {0, 0}; distance is 0.
    EXPECT_DOUBLE_EQ(result[1], 0.0);
}

TEST(DescriptorNumericCompareTest, MahalanobisRejectsAUniformlyNegativeMatrixAtExtremeScale) {
    // Restoring the eigenvalue scale before the sign test overflows the negative eigenvalue to
    // -infinity, which makes the tolerance infinite and reduces the rejection to -inf < -inf.
    // Without the scaled-domain check this matrix yields a zero whitening factor and a distance
    // of exactly 0.0 instead of throwing.
    const auto huge = std::numeric_limits<double>::max();
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    const auto metric = Metric::Mahalanobis({-huge, -huge, -huge, -huge});
    EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u, metric), std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, MahalanobisRejectsAnIndefiniteMatrixAtExtremeScale) {
    // The overflow does not have to land on the negative direction. Here it is the positive
    // eigenvalue that overflows, and an infinite tolerance then exempts the finite negative one
    // as well, so a plainly indefinite matrix is accepted.
    const auto huge = std::numeric_limits<double>::max();
    const std::vector<double> values = {1.0, 2.0, 4.0, 6.0};
    const auto metric = Metric::Mahalanobis({huge, 0.0, 0.0, -huge});
    EXPECT_THROW(PDistNumeric(values.data(), nullptr, 2u, 2u, metric), std::invalid_argument);
}

TEST(DescriptorNumericCompareTest, MahalanobisKeepsIdenticalRowsAtZeroForAnExtremeScalePsdMatrix) {
    // DBL_MAX * ones(2, 2) is finite, symmetric and positive semidefinite, with eigenvalues
    // {0, 2 * DBL_MAX}. Restoring the scale before taking the root overflows the intermediate to
    // infinity, and the resulting infinite factor turns a zero-length difference into 0.0 * inf,
    // i.e. NaN. The distance from a row to itself is 0.0 for any finite inverse covariance.
    const auto huge = std::numeric_limits<double>::max();
    const std::vector<double> zeros = {0.0, 0.0, 0.0, 0.0};
    const auto metric = Metric::Mahalanobis({huge, huge, huge, huge});
    const auto result = PDistNumeric(zeros.data(), nullptr, 2u, 2u, metric);
    EXPECT_DOUBLE_EQ(result[0], 0.0);

    const std::vector<double> repeated = {1.0, 0.0, 1.0, 0.0};
    const auto same = PDistNumeric(repeated.data(), nullptr, 2u, 2u, metric);
    EXPECT_DOUBLE_EQ(same[0], 0.0);
}

TEST(DescriptorNumericBatchTest, PDistMatchesTheBufferKernelOverTheSameMatrix) {
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW", "nAtom"}),
                                           DescriptorMissingPolicy::Propagate};

    const auto result = PDist(batch, Metric::Euclidean(), options);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_DOUBLE_EQ(result[0], 5.0);  // sqrt(9 + 16)
}

TEST(DescriptorNumericBatchTest, BoolColumnsMapToZeroAndOne) {
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Names({"Lipinski"}),
                                           DescriptorMissingPolicy::Propagate};
    EXPECT_DOUBLE_EQ(PDist(batch, Metric::Manhattan(), options)[0], 1.0);
}

TEST(DescriptorNumericBatchTest, CDistProducesARectangularMatrix) {
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW", "nAtom"}),
                                           DescriptorMissingPolicy::Propagate};
    const auto result = CDist(batch, batch, Metric::Euclidean(), options);
    ASSERT_EQ(result.size(), 4u);
    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 5.0);
    EXPECT_DOUBLE_EQ(result[2], 5.0);
    EXPECT_DOUBLE_EQ(result[3], 0.0);
}

TEST(DescriptorNumericBatchTest, IntoFormsValidateTheirOutputLength) {
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW"}),
                                           DescriptorMissingPolicy::Propagate};
    std::vector<double> output(3u, 0.0);
    EXPECT_THROW(PDistInto(batch, Metric::Euclidean(), options, output.data(), output.size()),
                 std::invalid_argument);

    // CDistInto over batches with different row counts (2 and 3) expects 6 entries. Test both
    // too-short and too-long to ensure the wrapper forwards the caller's output_length rather
    // than recomputing the product from the matrix dimensions.
    const auto schema = numeric_schema();
    DescriptorSetBuilder row1(schema);
    row1.Set("MW", DescriptorValue::Float(1.0));
    DescriptorSetBuilder row2(schema);
    row2.Set("MW", DescriptorValue::Float(2.0));
    DescriptorSetBuilder row3(schema);
    row3.Set("MW", DescriptorValue::Float(3.0));
    const auto a_batch = DescriptorBatch::FromDescriptorSets({row1.Build("a1"), row2.Build("a2")});
    const auto b_batch = DescriptorBatch::FromDescriptorSets(
        {row1.Build("b1"), row2.Build("b2"), row3.Build("b3")});

    std::vector<double> too_short(5u, 0.0);
    EXPECT_THROW(CDistInto(a_batch, b_batch, Metric::Euclidean(), options, too_short.data(),
                           too_short.size()),
                 std::invalid_argument);

    std::vector<double> too_long(7u, 0.0);
    EXPECT_THROW(CDistInto(a_batch, b_batch, Metric::Euclidean(), options, too_long.data(),
                           too_long.size()),
                 std::invalid_argument);
}

TEST(DescriptorNumericBatchTest, RejectsStringColumnsAndLegacyBatches) {
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Names({"Class"}),
                                           DescriptorMissingPolicy::Propagate};
    EXPECT_THROW(PDist(batch, Metric::Euclidean(), options), std::invalid_argument);

    DescriptorBatch legacy;
    EXPECT_THROW(PDist(legacy, Metric::Euclidean(), options), std::invalid_argument);
}

TEST(DescriptorNumericBatchTest, CDistRejectsMismatchedSchemas) {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"MW", DescriptorValueKind::Float, "test:scalar"});
    const auto other_schema = builder.Build();

    DescriptorSetBuilder row(other_schema);
    row.Set("MW", DescriptorValue::Float(1.0));
    const auto other = DescriptorBatch::FromDescriptorSets({row.Build("c")});

    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW"}),
                                           DescriptorMissingPolicy::Propagate};
    EXPECT_THROW(CDist(two_row_batch(), other, Metric::Euclidean(), options),
                 std::invalid_argument);
}

TEST(DescriptorNumericBatchTest, AnEmptySelectionYieldsNaNThroughEveryOverload) {
    // DescriptorSelection::Group resolves an unknown group to zero indices rather than
    // throwing (src/descriptor_schema.cpp:201-207), so a zero-width matrix is reachable
    // through the batch overloads and not only through the raw-buffer kernel. The whole
    // chain -- ToNumericMatrix, the impl helper, the kernel -- has to survive it.
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Group("no-such-group"),
                                           DescriptorMissingPolicy::Propagate};

    const auto pairwise = PDist(batch, Metric::Euclidean(), options);
    ASSERT_EQ(pairwise.size(), 1u);
    EXPECT_TRUE(std::isnan(pairwise[0]));

    const auto cross = CDist(batch, batch, Metric::Euclidean(), options);
    ASSERT_EQ(cross.size(), 4u);
    for (const auto value : cross) {
        EXPECT_TRUE(std::isnan(value));
    }
}

TEST(DescriptorNumericBatchTest, BadVarianceMessageNamesTheColumn) {
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW", "nAtom"}),
                                           DescriptorMissingPolicy::Propagate};
    const auto metric = Metric::StandardizedEuclidean({1.0, 0.0});

    try {
        PDist(batch, metric, options);
        FAIL() << "Expected a rejection for the zero variance.";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(std::string(error.what()).find("'nAtom'"), std::string::npos);
    }
}

TEST(DescriptorNumericBatchTest, CDistOverDifferentBatchesUsesDistinctValues) {
    const auto schema = numeric_schema();

    DescriptorSetBuilder a_builder(schema);
    a_builder.Set("MW", DescriptorValue::Float(0.0));
    a_builder.Set("nAtom", DescriptorValue::Int(0));

    DescriptorSetBuilder b1_builder(schema);
    b1_builder.Set("MW", DescriptorValue::Float(3.0));
    b1_builder.Set("nAtom", DescriptorValue::Int(4));

    DescriptorSetBuilder b2_builder(schema);
    b2_builder.Set("MW", DescriptorValue::Float(6.0));
    b2_builder.Set("nAtom", DescriptorValue::Int(8));

    const auto a_batch = DescriptorBatch::FromDescriptorSets({a_builder.Build("a")});
    const auto b_batch = DescriptorBatch::FromDescriptorSets(
        {b1_builder.Build("b1"), b2_builder.Build("b2")});

    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW", "nAtom"}),
                                           DescriptorMissingPolicy::Propagate};
    const auto result = CDist(a_batch, b_batch, Metric::Euclidean(), options);
    ASSERT_EQ(result.size(), 2u);
    EXPECT_DOUBLE_EQ(result[0], 5.0);   // sqrt(9 + 16)
    EXPECT_DOUBLE_EQ(result[1], 10.0);  // sqrt(36 + 64)
}

TEST(DescriptorNumericBatchTest, PDistIntoWritesCorrectValues) {
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW", "nAtom"}),
                                           DescriptorMissingPolicy::Propagate};
    std::vector<double> output(1u, 999.0);
    PDistInto(batch, Metric::Euclidean(), options, output.data(), output.size());
    EXPECT_DOUBLE_EQ(output[0], 5.0);
}

TEST(DescriptorNumericBatchTest, CDistIntoWritesCorrectValues) {
    const auto batch = two_row_batch();
    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW", "nAtom"}),
                                           DescriptorMissingPolicy::Propagate};
    std::vector<double> output(4u, 999.0);
    CDistInto(batch, batch, Metric::Euclidean(), options, output.data(), output.size());
    ASSERT_EQ(output.size(), 4u);
    EXPECT_DOUBLE_EQ(output[0], 0.0);
    EXPECT_DOUBLE_EQ(output[1], 5.0);
    EXPECT_DOUBLE_EQ(output[2], 5.0);
    EXPECT_DOUBLE_EQ(output[3], 0.0);
}

TEST(DescriptorNumericBatchTest, MissingPolicyIsForwarded) {
    const auto schema = numeric_schema();

    DescriptorSetBuilder first(schema);
    first.Set("MW", DescriptorValue::Float(1.0));
    // nAtom is missing

    DescriptorSetBuilder second(schema);
    second.Set("MW", DescriptorValue::Float(4.0));
    second.Set("nAtom", DescriptorValue::Int(6));

    const auto batch = DescriptorBatch::FromDescriptorSets({first.Build("a"), second.Build("b")});

    const DescriptorNumericOptions propagate_options{
        DescriptorSelection::Names({"MW", "nAtom"}), DescriptorMissingPolicy::Propagate};
    const auto propagate_result = PDist(batch, Metric::Euclidean(), propagate_options);
    ASSERT_EQ(propagate_result.size(), 1u);
    EXPECT_TRUE(std::isnan(propagate_result[0]));

    const DescriptorNumericOptions ignore_options{
        DescriptorSelection::Names({"MW", "nAtom"}), DescriptorMissingPolicy::Ignore};
    const auto ignore_result = PDist(batch, Metric::Euclidean(), ignore_options);
    ASSERT_EQ(ignore_result.size(), 1u);
    EXPECT_DOUBLE_EQ(ignore_result[0], std::sqrt(18.0));  // sqrt((4-1)^2 * 2) for the rescale
}

TEST(DescriptorNumericBatchTest, CDistOrientsTheOutputMatrixRowsByColumns) {
    // Transposing the two matrices in the cdist_numeric_impl call transposes the output. That
    // survives a 1-by-n fixture, whose transpose flattens identically, and a batch compared
    // against itself, whose output matrix is symmetric. It takes an off-diagonal pair that
    // differs to pin the layout, and metric symmetry is irrelevant to it.
    const auto schema = numeric_schema();
    DescriptorSetBuilder a0(schema);
    a0.Set("MW", DescriptorValue::Float(0.0));
    a0.Set("nAtom", DescriptorValue::Int(0));
    DescriptorSetBuilder a1(schema);
    a1.Set("MW", DescriptorValue::Float(10.0));
    a1.Set("nAtom", DescriptorValue::Int(0));
    DescriptorSetBuilder b0(schema);
    b0.Set("MW", DescriptorValue::Float(1.0));
    b0.Set("nAtom", DescriptorValue::Int(0));
    DescriptorSetBuilder b1(schema);
    b1.Set("MW", DescriptorValue::Float(100.0));
    b1.Set("nAtom", DescriptorValue::Int(0));

    const auto a_batch = DescriptorBatch::FromDescriptorSets({a0.Build("a0"), a1.Build("a1")});
    const auto b_batch = DescriptorBatch::FromDescriptorSets({b0.Build("b0"), b1.Build("b1")});
    const DescriptorNumericOptions options{DescriptorSelection::Names({"MW", "nAtom"}),
                                           DescriptorMissingPolicy::Propagate};
    const auto result = CDist(a_batch, b_batch, Metric::Euclidean(), options);
    ASSERT_EQ(result.size(), 4u);
    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 100.0);
    EXPECT_DOUBLE_EQ(result[2], 9.0);
    EXPECT_DOUBLE_EQ(result[3], 90.0);
}

TEST(DescriptorNumericBatchTest, AnUnresolvableSelectionThrowsOutOfRange) {
    // The wrappers propagate the selection's own exception type rather than normalizing it, which
    // is what ToNumericMatrix documents and what the rest of include/oefp does.
    const auto batch = two_row_batch();
    const DescriptorNumericOptions by_name{DescriptorSelection::Names({"NoSuchColumn"}),
                                           DescriptorMissingPolicy::Propagate};
    EXPECT_THROW(PDist(batch, Metric::Euclidean(), by_name), std::out_of_range);

    const DescriptorNumericOptions by_index{DescriptorSelection::Indices({99u}),
                                            DescriptorMissingPolicy::Propagate};
    EXPECT_THROW(CDist(batch, batch, Metric::Euclidean(), by_index), std::out_of_range);
}

TEST(DescriptorNumericBatchTest, CDistForwardsMissingValuePolicyForBothSides) {
    // Missing value on the b side specifically tests that b_matrix.validity is forwarded rather
    // than aliased to a_matrix.validity, which all other CDist tests miss.
    const auto schema = numeric_schema();
    DescriptorSetBuilder a_builder(schema);
    a_builder.Set("MW", DescriptorValue::Float(1.0));
    a_builder.Set("nAtom", DescriptorValue::Int(2));

    DescriptorSetBuilder b_builder(schema);
    b_builder.Set("MW", DescriptorValue::Float(4.0));
    // nAtom is missing

    const auto a_batch = DescriptorBatch::FromDescriptorSets({a_builder.Build("a")});
    const auto b_batch = DescriptorBatch::FromDescriptorSets({b_builder.Build("b")});

    const DescriptorNumericOptions propagate_options{
        DescriptorSelection::Names({"MW", "nAtom"}), DescriptorMissingPolicy::Propagate};
    const auto propagate_result = CDist(a_batch, b_batch, Metric::Euclidean(), propagate_options);
    ASSERT_EQ(propagate_result.size(), 1u);
    EXPECT_TRUE(std::isnan(propagate_result[0]));

    const DescriptorNumericOptions ignore_options{
        DescriptorSelection::Names({"MW", "nAtom"}), DescriptorMissingPolicy::Ignore};
    const auto ignore_result = CDist(a_batch, b_batch, Metric::Euclidean(), ignore_options);
    ASSERT_EQ(ignore_result.size(), 1u);
    EXPECT_DOUBLE_EQ(ignore_result[0], std::sqrt(18.0));  // sqrt((4-1)^2 * 2) for the rescale

    // Drive the same fixture through CDistInto to cover both wrapper forms.
    std::vector<double> propagate_into(1u, 999.0);
    CDistInto(a_batch, b_batch, Metric::Euclidean(), propagate_options, propagate_into.data(),
              propagate_into.size());
    EXPECT_TRUE(std::isnan(propagate_into[0]));

    std::vector<double> ignore_into(1u, 999.0);
    CDistInto(a_batch, b_batch, Metric::Euclidean(), ignore_options, ignore_into.data(),
              ignore_into.size());
    EXPECT_DOUBLE_EQ(ignore_into[0], std::sqrt(18.0));
}

} // namespace test
} // namespace OEFP
