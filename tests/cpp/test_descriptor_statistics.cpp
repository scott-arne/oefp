#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "oefp/descriptor.h"
#include "oefp/descriptor_batch.h"
#include "oefp/descriptor_compare.h"
#include "oefp/descriptor_schema.h"
#include "oefp/descriptor_selection.h"
#include "oefp/descriptor_statistics.h"
#include "oefp/metric.h"

namespace OEFP {
namespace {

std::shared_ptr<const DescriptorSchema> statistics_schema() {
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{"A", DescriptorValueKind::Float, "test:scalar"});
    builder.Add(DescriptorDefinition{"B", DescriptorValueKind::Float, "test:scalar"});
    return builder.Build();
}

// Rows: (1, 10), (2, missing), (3, 30).
DescriptorBatch gapped_batch() {
    const auto schema = statistics_schema();

    DescriptorSetBuilder first(schema);
    first.Set("A", DescriptorValue::Float(1.0));
    first.Set("B", DescriptorValue::Float(10.0));

    DescriptorSetBuilder second(schema);
    second.Set("A", DescriptorValue::Float(2.0));

    DescriptorSetBuilder third(schema);
    third.Set("A", DescriptorValue::Float(3.0));
    third.Set("B", DescriptorValue::Float(30.0));

    return DescriptorBatch::FromDescriptorSets(
        {first.Build("r0"), second.Build("r1"), third.Build("r2")});
}

// Rows: (-1, -2), (1, 2), (-1, 2), (1, -2). No missing values, and the two columns are
// uncorrelated with variances 4/3 and 16/3, which are therefore also the two covariance
// eigenvalues. That spread is what makes an explicit rcond able to drop exactly one of them.
DescriptorBatch full_rank_batch() {
    const auto schema = statistics_schema();

    const double rows[4][2] = {{-1.0, -2.0}, {1.0, 2.0}, {-1.0, 2.0}, {1.0, -2.0}};
    std::vector<DescriptorSet> sets;
    sets.reserve(4u);
    for (std::size_t row = 0u; row < 4u; ++row) {
        DescriptorSetBuilder builder(schema);
        builder.Set("A", DescriptorValue::Float(rows[row][0]));
        builder.Set("B", DescriptorValue::Float(rows[row][1]));
        sets.push_back(builder.Build("r" + std::to_string(row)));
    }
    return DescriptorBatch::FromDescriptorSets(sets);
}

TEST(DescriptorStatisticsTest, ColumnStatisticsUsePerColumnPresentValues) {
    const auto statistics =
        ColumnStatistics(gapped_batch(), DescriptorSelection::Names({"A", "B"}));

    ASSERT_EQ(statistics.names.size(), 2u);
    EXPECT_EQ(statistics.names[0], "A");
    EXPECT_EQ(statistics.names[1], "B");
    EXPECT_EQ(statistics.present_count[0], 3u);
    EXPECT_EQ(statistics.present_count[1], 2u);
    EXPECT_DOUBLE_EQ(statistics.mean[0], 2.0);
    EXPECT_DOUBLE_EQ(statistics.mean[1], 20.0);
    EXPECT_DOUBLE_EQ(statistics.variance[0], 1.0);
    EXPECT_DOUBLE_EQ(statistics.variance[1], 200.0);
    EXPECT_DOUBLE_EQ(statistics.minimum[0], 1.0);
    EXPECT_DOUBLE_EQ(statistics.maximum[0], 3.0);
    EXPECT_DOUBLE_EQ(statistics.minimum[1], 10.0);
    EXPECT_DOUBLE_EQ(statistics.maximum[1], 30.0);
}

TEST(DescriptorStatisticsTest, ResultsFollowSelectionOrderNotSchemaOrder) {
    // "B" precedes "A" here but follows it in the schema, so every assertion below is
    // asymmetric on purpose: returning schema order would swap all of them.
    const auto batch = gapped_batch();
    const auto selection = DescriptorSelection::Names({"B", "A"});

    const auto statistics = ColumnStatistics(batch, selection);
    ASSERT_EQ(statistics.names.size(), 2u);
    EXPECT_EQ(statistics.names[0], "B");
    EXPECT_EQ(statistics.names[1], "A");
    EXPECT_EQ(statistics.present_count[0], 2u);
    EXPECT_EQ(statistics.present_count[1], 3u);
    EXPECT_DOUBLE_EQ(statistics.mean[0], 20.0);
    EXPECT_DOUBLE_EQ(statistics.mean[1], 2.0);

    const auto covariance = CovarianceMatrix(batch, selection);
    ASSERT_EQ(covariance.matrix.size(), 4u);
    EXPECT_DOUBLE_EQ(covariance.matrix[0], 200.0);
    EXPECT_DOUBLE_EQ(covariance.matrix[3], 2.0);
}

TEST(DescriptorStatisticsTest, EmptyAndSingletonColumnsReportNaN) {
    const std::vector<double> values{1.0, 5.0};
    const std::vector<std::uint8_t> validity{1u, 0u};
    const auto statistics = ColumnStatistics(values.data(), validity.data(), 2u, 1u);

    EXPECT_EQ(statistics.present_count[0], 1u);
    EXPECT_DOUBLE_EQ(statistics.mean[0], 1.0);
    EXPECT_TRUE(std::isnan(statistics.variance[0]));
    EXPECT_TRUE(statistics.names.empty());

    const std::vector<std::uint8_t> none{0u, 0u};
    const auto empty = ColumnStatistics(values.data(), none.data(), 2u, 1u);
    EXPECT_EQ(empty.present_count[0], 0u);
    EXPECT_TRUE(std::isnan(empty.mean[0]));
    // A guard relaxed from ">= 2" to ">= 1" still yields NaN at one present value, from
    // 0.0 / 0.0, so only this zero-present case pins it: here the unsigned n - 1 wraps and the
    // variance would come back as a plain 0.0, which reads as a real answer.
    EXPECT_TRUE(std::isnan(empty.variance[0]));
    EXPECT_TRUE(std::isnan(empty.minimum[0]));
    EXPECT_TRUE(std::isnan(empty.maximum[0]));
}

TEST(DescriptorStatisticsTest, AZeroRowMatrixYieldsAllNaN) {
    // The header allows a null buffer when rows is zero, because that is what an empty batch's
    // std::vector::data() is allowed to hand back.
    const auto statistics = ColumnStatistics(nullptr, nullptr, 0u, 2u);

    ASSERT_EQ(statistics.mean.size(), 2u);
    ASSERT_EQ(statistics.variance.size(), 2u);
    ASSERT_EQ(statistics.minimum.size(), 2u);
    ASSERT_EQ(statistics.maximum.size(), 2u);
    ASSERT_EQ(statistics.present_count.size(), 2u);

    for (std::size_t column = 0u; column < 2u; ++column) {
        EXPECT_EQ(statistics.present_count[column], 0u);
        EXPECT_TRUE(std::isnan(statistics.mean[column]));
        EXPECT_TRUE(std::isnan(statistics.variance[column]));
        EXPECT_TRUE(std::isnan(statistics.minimum[column]));
        EXPECT_TRUE(std::isnan(statistics.maximum[column]));
    }
}

TEST(DescriptorStatisticsTest, CovarianceUsesListwiseDeletion) {
    const auto covariance =
        CovarianceMatrix(gapped_batch(), DescriptorSelection::Names({"A", "B"}));

    // Only rows r0 and r2 are complete: A = (1, 3), B = (10, 30).
    EXPECT_EQ(covariance.row_count, 2u);
    ASSERT_EQ(covariance.matrix.size(), 4u);
    EXPECT_DOUBLE_EQ(covariance.matrix[0], 2.0);
    EXPECT_DOUBLE_EQ(covariance.matrix[1], 20.0);
    EXPECT_DOUBLE_EQ(covariance.matrix[2], 20.0);
    EXPECT_DOUBLE_EQ(covariance.matrix[3], 200.0);
}

TEST(DescriptorStatisticsTest, CovarianceFillsTheWholeUpperTriangleOfALargerMatrix) {
    // Every other covariance fixture here is 1x1 or 2x2, where the upper triangle holds a
    // single off-diagonal and an index error in the (i, j) loops cannot show itself. Five
    // rows, three columns, all three column means exactly zero, and all six upper-triangle
    // entries distinct and exactly representable:
    //   var  A = 34/4 = 8.5    cov(A, B) = -22/4 = -5.5    cov(A, C) = -13/4 = -3.25
    //   var  B = 20/4 = 5.0    cov(B, C) =   8/4 =  2.0
    //   var  C = 14/4 = 3.5
    const std::vector<double> values{
        -4.0,  3.0,  1.0,
        -1.0, -1.0,  2.0,
         0.0,  0.0, -2.0,
         1.0,  1.0,  1.0,
         4.0, -3.0, -2.0,
    };
    const auto covariance = CovarianceMatrix(values.data(), nullptr, 5u, 3u);

    EXPECT_EQ(covariance.row_count, 5u);
    ASSERT_EQ(covariance.matrix.size(), 9u);
    EXPECT_DOUBLE_EQ(covariance.matrix[0], 8.5);
    EXPECT_DOUBLE_EQ(covariance.matrix[1], -5.5);
    EXPECT_DOUBLE_EQ(covariance.matrix[2], -3.25);
    EXPECT_DOUBLE_EQ(covariance.matrix[4], 5.0);
    EXPECT_DOUBLE_EQ(covariance.matrix[5], 2.0);
    EXPECT_DOUBLE_EQ(covariance.matrix[8], 3.5);

    // The mirror is written from the same double, so it is bitwise equal, not merely close.
    EXPECT_DOUBLE_EQ(covariance.matrix[3], covariance.matrix[1]);
    EXPECT_DOUBLE_EQ(covariance.matrix[6], covariance.matrix[2]);
    EXPECT_DOUBLE_EQ(covariance.matrix[7], covariance.matrix[5]);
}

TEST(DescriptorStatisticsTest, CovarianceRejectsFewerThanTwoCompleteRows) {
    const std::vector<double> values{1.0, 2.0};
    const std::vector<std::uint8_t> validity{1u, 0u};
    EXPECT_THROW(CovarianceMatrix(values.data(), validity.data(), 2u, 1u),
                 std::invalid_argument);
}

TEST(DescriptorStatisticsTest, InverseCovarianceInvertsAFullRankMatrix) {
    // Four rows, both column means zero. Column A is (-1, 1, -1, 1) with sum of squares 4, so
    // its variance is 4/3; column B is (-2, 2, 2, -2) with sum of squares 16, so its variance
    // is 16/3. The A-B products are 2, 2, -2, -2 and sum to zero, so the columns are
    // uncorrelated on this fixture and both the covariance and its inverse are diagonal.
    const std::vector<double> values{-1.0, -2.0, 1.0, 2.0, -1.0, 2.0, 1.0, -2.0};
    const auto inverse = InverseCovarianceMatrix(values.data(), nullptr, 4u, 2u);

    EXPECT_EQ(inverse.row_count, 4u);
    EXPECT_EQ(inverse.rank, 2u);
    ASSERT_EQ(inverse.matrix.size(), 4u);
    EXPECT_NEAR(inverse.matrix[0], 1.0 / (4.0 / 3.0), 1.0e-12);
    EXPECT_NEAR(inverse.matrix[1], 0.0, 1.0e-12);
    EXPECT_NEAR(inverse.matrix[2], 0.0, 1.0e-12);
    EXPECT_NEAR(inverse.matrix[3], 1.0 / (16.0 / 3.0), 1.0e-12);
}

TEST(DescriptorStatisticsTest, InverseCovarianceTruncatesARankDeficientMatrix) {
    // Column B is exactly 2 * column A, so the covariance has rank 1.
    const std::vector<double> values{1.0, 2.0, 2.0, 4.0, 3.0, 6.0};
    const auto inverse = InverseCovarianceMatrix(values.data(), nullptr, 3u, 2u);

    EXPECT_EQ(inverse.rank, 1u);
    // The pseudo-inverse of a rank-1 symmetric matrix is symmetric with the same null space.
    EXPECT_NEAR(inverse.matrix[1], inverse.matrix[2], 1.0e-12);
}

TEST(DescriptorStatisticsTest, SmallMagnitudeColumnsInvertCorrectly) {
    // The rank-deficient fixture above, scaled by 1e-7. The covariance is
    // 1e-14 * [[1, 2], [2, 4]], whose 2e-14 off-diagonal is below the absolute 1.0e-13
    // threshold the Jacobi sweep converges against, so before pseudo_inverse_symmetric
    // normalized its input this came back as rank 2 with a diagonal "inverse" — a silently
    // wrong Mahalanobis VI, since a positive diagonal is positive definite. The trigger is
    // just "every selected column has small variance", which fractional or already
    // standardised descriptor columns produce routinely.
    const std::vector<double> values{1.0e-7, 2.0e-7, 2.0e-7, 4.0e-7, 3.0e-7, 6.0e-7};
    const auto inverse = InverseCovarianceMatrix(values.data(), nullptr, 3u, 2u);

    EXPECT_EQ(inverse.rank, 1u);
    ASSERT_EQ(inverse.matrix.size(), 4u);
    EXPECT_NE(inverse.matrix[1], 0.0);
    EXPECT_DOUBLE_EQ(inverse.matrix[1], inverse.matrix[2]);
    EXPECT_NEAR(inverse.matrix[0], 4.0e12, 1.0e1);
    EXPECT_NEAR(inverse.matrix[1], 8.0e12, 1.0e1);
    EXPECT_NEAR(inverse.matrix[3], 1.6e13, 1.0e1);
}

TEST(DescriptorStatisticsTest, InverseCovarianceRejectsAZeroMatrix) {
    // This pins that a constant column really does produce a zero covariance at this layer.
    const std::vector<double> values{7.0, 7.0, 7.0};
    EXPECT_THROW(InverseCovarianceMatrix(values.data(), nullptr, 3u, 1u),
                 std::invalid_argument);
}

TEST(DescriptorStatisticsTest, CovarianceDiagonalDiffersFromPerColumnVariance) {
    // This pins the deliberate two-row-set behaviour of spec section 4.3 rather than
    // leaving it to be discovered. Column A has three present values (1, 2, 3), so its
    // per-column variance is 1.0. Listwise deletion drops row r1 for the covariance,
    // leaving A = (1, 3) with variance 2.0.
    const auto batch = gapped_batch();
    const auto selection = DescriptorSelection::Names({"A", "B"});

    const auto statistics = ColumnStatistics(batch, selection);
    const auto covariance = CovarianceMatrix(batch, selection);

    EXPECT_DOUBLE_EQ(statistics.variance[0], 1.0);
    EXPECT_DOUBLE_EQ(covariance.matrix[0], 2.0);
    EXPECT_NE(statistics.variance[0], covariance.matrix[0]);

    // Column B has no extra present values beyond the complete rows, so it agrees. The two
    // sides come from different accumulators — ColumnStatistics runs Welford, CovarianceMatrix
    // takes a mean pass then a centred-products pass — and it is EXPECT_DOUBLE_EQ's four-ULP
    // band, not exact equality, that makes comparing them legitimate. On {10.0, 30.0} both
    // routes happen to land on exactly 200.0, but nothing requires them to.
    EXPECT_DOUBLE_EQ(statistics.variance[1], covariance.matrix[3]);
}

TEST(DescriptorStatisticsTest, StatisticsFedStraightIntoAMetricAreStillValidated) {
    // Spec section 4.2: ColumnStatistics output is not unconditionally safe to feed
    // forward. This is the real path by which a caller reaches a NaN variance, and the
    // Task 8 validation is what has to catch it.
    const std::vector<double> values{1.0, 5.0, 2.0, 7.0};
    const std::vector<std::uint8_t> validity{1u, 1u, 0u, 1u};
    const auto statistics = ColumnStatistics(values.data(), validity.data(), 2u, 2u);

    ASSERT_EQ(statistics.present_count[0], 1u);
    ASSERT_TRUE(std::isnan(statistics.variance[0]));

    const auto metric = Metric::StandardizedEuclidean(statistics.variance);
    std::vector<double> output(1u, 0.0);
    EXPECT_THROW(PDistNumericInto(values.data(), validity.data(), 2u, 2u, metric,
                                  DescriptorMissingPolicy::Propagate, output.data(),
                                  output.size()),
                 std::invalid_argument);
}

TEST(DescriptorStatisticsTest, PseudoInverseSatisfiesTheMoorePenroseRoundTrip) {
    // This test pins the plumbing: that InverseCovarianceMatrix inverts the same matrix
    // CovarianceMatrix returns, and not a differently-built one. This fixture is exactly
    // collinear: the covariance is [[1, 2], [2, 4]] with eigenvalues 0 and 5.
    const std::vector<double> values{1.0, 2.0, 2.0, 4.0, 3.0, 6.0};
    const auto covariance = CovarianceMatrix(values.data(), nullptr, 3u, 2u);
    const auto inverse = InverseCovarianceMatrix(values.data(), nullptr, 3u, 2u);

    constexpr std::size_t DIMENSION = 2u;
    std::vector<double> product(DIMENSION * DIMENSION, 0.0);
    for (std::size_t i = 0u; i < DIMENSION; ++i) {
        for (std::size_t j = 0u; j < DIMENSION; ++j) {
            double total = 0.0;
            for (std::size_t k = 0u; k < DIMENSION; ++k) {
                total += covariance.matrix[(i * DIMENSION) + k] *
                         inverse.matrix[(k * DIMENSION) + j];
            }
            product[(i * DIMENSION) + j] = total;
        }
    }

    for (std::size_t i = 0u; i < DIMENSION; ++i) {
        for (std::size_t j = 0u; j < DIMENSION; ++j) {
            double total = 0.0;
            for (std::size_t k = 0u; k < DIMENSION; ++k) {
                total += product[(i * DIMENSION) + k] * covariance.matrix[(k * DIMENSION) + j];
            }
            EXPECT_NEAR(total, covariance.matrix[(i * DIMENSION) + j], 1.0e-9);
        }
    }
}

TEST(DescriptorStatisticsTest, ARankDeficientInverseCovarianceIsAcceptedByTheMahalanobisMetric) {
    // whitening_factor's tolerance comment claims a matrix this library produced always passes
    // its positive-semidefinite check. The two bands are the same expression, but the
    // pseudo-inverse drops its null directions and reconstruction returns them as small,
    // possibly negative residues, so whether they land inside the band is empirical.
    //
    // The fixture has to earn that. Any exactly-collinear pair of columns cancels in exact
    // arithmetic and leaves a null eigenvalue of precisely 0.0, which clears a band of any
    // width including zero and demonstrates nothing. Here the dependency is
    // c3 = 0.1*c0 + 0.3*c1 + 0.7*c2, whose coefficients are not dyadic, so the residue in the
    // dropped direction is genuinely non-zero and the band is what decides the verdict.
    const std::vector<double> values{
        1.0, 2.0, 3.0, 2.8,
        2.0, 1.0, 5.0, 4.0,
        4.0, 3.0, 1.0, 2.0,
        3.0, 5.0, 2.0, 3.2,
        5.0, 4.0, 4.0, 4.5,
        2.0, 6.0, 1.0, 2.7,
    };
    const auto inverse = InverseCovarianceMatrix(values.data(), nullptr, 6u, 4u);
    ASSERT_EQ(inverse.rank, 3u);

    const auto metric = Metric::Mahalanobis(inverse.matrix);
    EXPECT_NO_THROW(PDistNumeric(values.data(), nullptr, 6u, 4u, metric));
}

TEST(DescriptorStatisticsTest, AFullRankInverseCovarianceIsAcceptedByTheMahalanobisMetric) {
    // This pins the full-rank companion to the rank-deficient round trip.
    const std::vector<double> values{-1.0, -2.0, 1.0, 2.0, -1.0, 2.0, 1.0, -2.0};
    const auto inverse = InverseCovarianceMatrix(values.data(), nullptr, 4u, 2u);
    ASSERT_EQ(inverse.rank, 2u);

    const auto metric = Metric::Mahalanobis(inverse.matrix);
    EXPECT_NO_THROW(PDistNumeric(values.data(), nullptr, 4u, 2u, metric));
}

TEST(DescriptorStatisticsTest, InverseCovarianceOfABatchUsesTheSelectedColumns) {
    // The batch overload is the entry point Task 11's address forms and Task 12's Python
    // bindings will be built on, and nothing else in the suite calls it. gapped_batch drops
    // row r1 to listwise deletion, leaving A = (1, 3) and B = (10, 30) and the covariance
    // [[2, 20], [20, 200]], which is exactly singular: 2 * 200 - 20 * 20 == 0.
    const auto inverse =
        InverseCovarianceMatrix(gapped_batch(), DescriptorSelection::Names({"A", "B"}));

    EXPECT_EQ(inverse.row_count, 2u);
    EXPECT_EQ(inverse.rank, 1u);
    ASSERT_EQ(inverse.matrix.size(), 4u);
    EXPECT_DOUBLE_EQ(inverse.matrix[1], inverse.matrix[2]);
    // The pseudo-inverse of a rank-1 matrix keeps its null space, so the off-diagonal survives.
    EXPECT_NE(inverse.matrix[1], 0.0);
}

TEST(DescriptorStatisticsTest, AnExplicitRcondChangesTheRetainedRank) {
    // The covariance eigenvalues are 4/3 and 16/3, so an rcond of 0.5 puts the cutoff at 8/3
    // and drops exactly the smaller one. Both overloads have to forward the caller's value:
    // substituting the 0.0 default in either would leave the rank at 2.
    const std::vector<double> values{-1.0, -2.0, 1.0, 2.0, -1.0, 2.0, 1.0, -2.0};
    EXPECT_EQ(InverseCovarianceMatrix(values.data(), nullptr, 4u, 2u).rank, 2u);
    EXPECT_EQ(InverseCovarianceMatrix(values.data(), nullptr, 4u, 2u, 0.5).rank, 1u);

    const auto batch = full_rank_batch();
    const auto selection = DescriptorSelection::Names({"A", "B"});
    EXPECT_EQ(InverseCovarianceMatrix(batch, selection).rank, 2u);
    EXPECT_EQ(InverseCovarianceMatrix(batch, selection, 0.5).rank, 1u);
}

TEST(DescriptorStatisticsTest, ANegativeRcondIsRejectedThroughBothOverloads) {
    // Task 4's rcond guards are tested against pseudo_inverse_symmetric directly, so this is
    // the only thing pinning that Task 10 hands the caller's value down at all.
    const std::vector<double> values{-1.0, -2.0, 1.0, 2.0, -1.0, 2.0, 1.0, -2.0};
    EXPECT_THROW(InverseCovarianceMatrix(values.data(), nullptr, 4u, 2u, -1.0),
                 std::invalid_argument);
    EXPECT_THROW(
        InverseCovarianceMatrix(full_rank_batch(), DescriptorSelection::Names({"A", "B"}), -1.0),
        std::invalid_argument);
}

TEST(DescriptorStatisticsTest, NullValidityMeansEveryValueIsPresent) {
    const std::vector<double> values{1.0, 2.0, 3.0};
    const auto statistics = ColumnStatistics(values.data(), nullptr, 3u, 1u);
    EXPECT_EQ(statistics.present_count[0], 3u);
    EXPECT_DOUBLE_EQ(statistics.mean[0], 2.0);
}

TEST(DescriptorStatisticsTest, NullValuesPointerIsRejected) {
    // All three, not just ColumnStatistics: InverseCovarianceMatrix performs no validation of
    // its own, so CovarianceMatrix's guard is the only thing between a null buffer and a null
    // dereference in the mean loop.
    EXPECT_THROW(ColumnStatistics(nullptr, nullptr, 3u, 1u), std::invalid_argument);
    EXPECT_THROW(CovarianceMatrix(nullptr, nullptr, 3u, 1u), std::invalid_argument);
    EXPECT_THROW(InverseCovarianceMatrix(nullptr, nullptr, 3u, 1u), std::invalid_argument);
}

}  // namespace
}  // namespace OEFP
