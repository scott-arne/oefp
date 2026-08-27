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

TEST(DescriptorStatisticsTest, ColumnStatisticsUsePerColumnPresentValues) {
    const auto statistics =
        ColumnStatistics(gapped_batch(), DescriptorSelection::Names({"A", "B"}));

    ASSERT_EQ(statistics.names.size(), 2u);
    EXPECT_EQ(statistics.names[0], "A");
    EXPECT_EQ(statistics.present_count[0], 3u);
    EXPECT_EQ(statistics.present_count[1], 2u);
    EXPECT_DOUBLE_EQ(statistics.mean[0], 2.0);
    EXPECT_DOUBLE_EQ(statistics.mean[1], 20.0);
    EXPECT_DOUBLE_EQ(statistics.variance[0], 1.0);
    EXPECT_DOUBLE_EQ(statistics.variance[1], 200.0);
    EXPECT_DOUBLE_EQ(statistics.minimum[1], 10.0);
    EXPECT_DOUBLE_EQ(statistics.maximum[1], 30.0);
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
    EXPECT_TRUE(std::isnan(empty.minimum[0]));
    EXPECT_TRUE(std::isnan(empty.maximum[0]));
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

TEST(DescriptorStatisticsTest, CovarianceRejectsFewerThanTwoCompleteRows) {
    const std::vector<double> values{1.0, 2.0};
    const std::vector<std::uint8_t> validity{1u, 0u};
    EXPECT_THROW(CovarianceMatrix(values.data(), validity.data(), 2u, 1u),
                 std::invalid_argument);
}

TEST(DescriptorStatisticsTest, InverseCovarianceInvertsAFullRankMatrix) {
    // Two independent columns with variances 2 and 8.
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

    // Column B has no extra present values beyond the complete rows, so it agrees.
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
    // possibly negative residues, so whether they land inside the band is empirical. This is the
    // round trip that claim rests on.
    const std::vector<double> values{1.0, 2.0, 2.0, 4.0, 3.0, 6.0};
    const auto inverse = InverseCovarianceMatrix(values.data(), nullptr, 3u, 2u);
    ASSERT_EQ(inverse.rank, 1u);

    const auto metric = Metric::Mahalanobis(inverse.matrix);
    EXPECT_NO_THROW(PDistNumeric(values.data(), nullptr, 3u, 2u, metric));
}

TEST(DescriptorStatisticsTest, AFullRankInverseCovarianceIsAcceptedByTheMahalanobisMetric) {
    // This pins the full-rank companion to the rank-deficient round trip.
    const std::vector<double> values{-1.0, -2.0, 1.0, 2.0, -1.0, 2.0, 1.0, -2.0};
    const auto inverse = InverseCovarianceMatrix(values.data(), nullptr, 4u, 2u);
    ASSERT_EQ(inverse.rank, 2u);

    const auto metric = Metric::Mahalanobis(inverse.matrix);
    EXPECT_NO_THROW(PDistNumeric(values.data(), nullptr, 4u, 2u, metric));
}

TEST(DescriptorStatisticsTest, NullValidityMeansEveryValueIsPresent) {
    const std::vector<double> values{1.0, 2.0, 3.0};
    const auto statistics = ColumnStatistics(values.data(), nullptr, 3u, 1u);
    EXPECT_EQ(statistics.present_count[0], 3u);
    EXPECT_DOUBLE_EQ(statistics.mean[0], 2.0);
}

TEST(DescriptorStatisticsTest, NullValuesPointerIsRejected) {
    EXPECT_THROW(ColumnStatistics(nullptr, nullptr, 3u, 1u), std::invalid_argument);
}

}  // namespace
}  // namespace OEFP
