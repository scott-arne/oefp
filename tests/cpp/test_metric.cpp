#include <gtest/gtest.h>

#include "oefp/metric.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {

TEST(MetricTest, DistanceFactoriesSetNameTypeAndSpace) {
    const auto euclidean = Metric::Euclidean();
    EXPECT_EQ(euclidean.Name(), MetricName::Euclidean);
    EXPECT_EQ(euclidean.Type(), MetricType::Distance);
    EXPECT_EQ(euclidean.Space(), MetricSpace::Real);

    const auto hamming = Metric::Hamming();
    EXPECT_EQ(hamming.Name(), MetricName::Hamming);
    EXPECT_EQ(hamming.Type(), MetricType::Distance);
    EXPECT_EQ(hamming.Space(), MetricSpace::Integer);

    const auto jaccard = Metric::Jaccard();
    EXPECT_EQ(jaccard.Name(), MetricName::Jaccard);
    EXPECT_EQ(jaccard.Type(), MetricType::Distance);
    EXPECT_EQ(jaccard.Space(), MetricSpace::Boolean);
}

TEST(MetricTest, SimilarityFactoriesSetNameTypeAndSpace) {
    const auto tanimoto = Metric::Tanimoto();
    EXPECT_EQ(tanimoto.Name(), MetricName::Tanimoto);
    EXPECT_EQ(tanimoto.Type(), MetricType::Similarity);
    EXPECT_EQ(tanimoto.Space(), MetricSpace::Boolean);
    EXPECT_EQ(tanimoto.Alpha(), 1.0);
    EXPECT_EQ(tanimoto.Beta(), 1.0);

    const auto tversky = Metric::Tversky(0.2, 0.8);
    EXPECT_EQ(tversky.Name(), MetricName::Tversky);
    EXPECT_EQ(tversky.Type(), MetricType::Similarity);
    EXPECT_EQ(tversky.Space(), MetricSpace::Boolean);
    EXPECT_EQ(tversky.Alpha(), 0.2);
    EXPECT_EQ(tversky.Beta(), 0.8);
}

TEST(MetricTest, ParameterizedDistanceFactoriesStoreParameters) {
    const auto minkowski = Metric::Minkowski(3.0, {1.0, 0.5, 2.0});
    EXPECT_EQ(minkowski.Name(), MetricName::Minkowski);
    EXPECT_EQ(minkowski.Type(), MetricType::Distance);
    EXPECT_EQ(minkowski.Space(), MetricSpace::Real);
    EXPECT_EQ(minkowski.P(), 3.0);
    EXPECT_EQ(minkowski.Weights(), std::vector<double>({1.0, 0.5, 2.0}));

    const auto standardized = Metric::StandardizedEuclidean({1.0, 2.0, 4.0});
    EXPECT_EQ(standardized.Name(), MetricName::StandardizedEuclidean);
    EXPECT_EQ(standardized.Variances(), std::vector<double>({1.0, 2.0, 4.0}));

    const auto mahalanobis = Metric::Mahalanobis({1.0, 0.0, 0.0, 2.0});
    EXPECT_EQ(mahalanobis.Name(), MetricName::Mahalanobis);
    EXPECT_EQ(mahalanobis.InverseCovariance(), std::vector<double>({1.0, 0.0, 0.0, 2.0}));
}

TEST(MetricTest, RejectsInvalidMetricParameters) {
    EXPECT_THROW(Metric::Minkowski(0.0), std::invalid_argument);
    EXPECT_THROW(Metric::Minkowski(-1.0), std::invalid_argument);
    EXPECT_THROW(Metric::Minkowski(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_THROW(Metric::Minkowski(2.0, {1.0, -1.0}), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(-0.1, 0.5), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(1.1, 0.5), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(0.5, -0.1), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(0.5, 1.1), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(std::numeric_limits<double>::quiet_NaN(), 0.5), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(0.5, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}

TEST(MetricTest, ValidateForPDistRejectsAsymmetricTverskyOnly) {
    EXPECT_NO_THROW(Metric::Euclidean().ValidateForPDist());
    EXPECT_NO_THROW(Metric::Jaccard().ValidateForPDist());
    EXPECT_NO_THROW(Metric::Tanimoto().ValidateForPDist());
    EXPECT_NO_THROW(Metric::Tversky(0.3, 0.3).ValidateForPDist());
    EXPECT_THROW(Metric::Tversky(0.2, 0.8).ValidateForPDist(), std::invalid_argument);
}

TEST(MetricTest, HasZeroSelfDistanceReportsEveryMetric) {
    // Kulsinski and Russell-Rao measure a self-distance of (dimensions - jointly set) /
    // dimensions, and the similarity metrics return 1.0 rather than 0.0 for identical inputs.
    EXPECT_FALSE(Metric::Kulsinski().HasZeroSelfDistance());
    EXPECT_FALSE(Metric::RussellRao().HasZeroSelfDistance());
    EXPECT_FALSE(Metric::Tanimoto().HasZeroSelfDistance());
    EXPECT_FALSE(Metric::Tversky(0.5, 0.5).HasZeroSelfDistance());

    EXPECT_TRUE(Metric::Euclidean().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Manhattan().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Chebyshev().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Minkowski(2.0).HasZeroSelfDistance());
    EXPECT_TRUE(Metric::StandardizedEuclidean({1.0, 1.0}).HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Mahalanobis({1.0, 0.0, 0.0, 1.0}).HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Haversine().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Hamming().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Canberra().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::BrayCurtis().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Jaccard().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Matching().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Dice().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::RogersTanimoto().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::SokalMichener().HasZeroSelfDistance());
    EXPECT_TRUE(Metric::SokalSneath().HasZeroSelfDistance());
}

TEST(MetricTest, SatisfiesTriangleInequalityReportsEveryMetric) {
    EXPECT_FALSE(Metric::Dice().SatisfiesTriangleInequality());
    EXPECT_FALSE(Metric::BrayCurtis().SatisfiesTriangleInequality());
    EXPECT_FALSE(Metric::Tanimoto().SatisfiesTriangleInequality());
    EXPECT_FALSE(Metric::Tversky(0.5, 0.5).SatisfiesTriangleInequality());

    EXPECT_TRUE(Metric::Euclidean().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Manhattan().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Chebyshev().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::StandardizedEuclidean({1.0, 1.0}).SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Mahalanobis({1.0, 0.0, 0.0, 1.0}).SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Haversine().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Hamming().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Canberra().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Jaccard().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Matching().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Kulsinski().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::RogersTanimoto().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::RussellRao().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::SokalMichener().SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::SokalSneath().SatisfiesTriangleInequality());
}

TEST(MetricTest, MinkowskiTriangleInequalityFollowsTheExponent) {
    EXPECT_FALSE(Metric::Minkowski(0.5).SatisfiesTriangleInequality());
    EXPECT_FALSE(Metric::Minkowski(0.999).SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Minkowski(1.0).SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Minkowski(1.5).SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Minkowski(3.0).SatisfiesTriangleInequality());

    // Weights scale each term and change neither answer.
    EXPECT_FALSE(Metric::Minkowski(0.5, {1.0, 2.0}).SatisfiesTriangleInequality());
    EXPECT_TRUE(Metric::Minkowski(2.0, {1.0, 2.0}).SatisfiesTriangleInequality());

    // The exponent never affects self-distance.
    EXPECT_TRUE(Metric::Minkowski(0.5).HasZeroSelfDistance());
    EXPECT_TRUE(Metric::Minkowski(3.0).HasZeroSelfDistance());
}

TEST(MetricTest, ValidateAsDistanceMetricAcceptsOnlyTrueMetrics) {
    EXPECT_NO_THROW(Metric::Euclidean().ValidateAsDistanceMetric());
    EXPECT_NO_THROW(Metric::Manhattan().ValidateAsDistanceMetric());
    EXPECT_NO_THROW(Metric::Chebyshev().ValidateAsDistanceMetric());
    EXPECT_NO_THROW(Metric::Minkowski(1.0).ValidateAsDistanceMetric());
    EXPECT_NO_THROW(Metric::Hamming().ValidateAsDistanceMetric());
    EXPECT_NO_THROW(Metric::Canberra().ValidateAsDistanceMetric());
    EXPECT_NO_THROW(Metric::Jaccard().ValidateAsDistanceMetric());
    EXPECT_NO_THROW(Metric::Matching().ValidateAsDistanceMetric());
    EXPECT_NO_THROW(Metric::SokalSneath().ValidateAsDistanceMetric());

    EXPECT_THROW(Metric::Tanimoto().ValidateAsDistanceMetric(), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(0.5, 0.5).ValidateAsDistanceMetric(), std::invalid_argument);
    EXPECT_THROW(Metric::Kulsinski().ValidateAsDistanceMetric(), std::invalid_argument);
    EXPECT_THROW(Metric::RussellRao().ValidateAsDistanceMetric(), std::invalid_argument);
    EXPECT_THROW(Metric::Dice().ValidateAsDistanceMetric(), std::invalid_argument);
    EXPECT_THROW(Metric::BrayCurtis().ValidateAsDistanceMetric(), std::invalid_argument);
    EXPECT_THROW(Metric::Minkowski(0.5).ValidateAsDistanceMetric(), std::invalid_argument);
}

TEST(MetricTest, ValidateAsDistanceMetricNamesTheFailedProperty) {
    const auto message_for = [](const Metric& metric) {
        try {
            metric.ValidateAsDistanceMetric();
        } catch (const std::invalid_argument& error) {
            return std::string(error.what());
        }
        return std::string("no throw");
    };

    EXPECT_EQ(message_for(Metric::Tanimoto()), "Similarity metrics are not valid distance metrics.");
    EXPECT_EQ(
        message_for(Metric::Kulsinski()),
        "Metrics without zero self-distance are not valid distance metrics.");
    EXPECT_EQ(
        message_for(Metric::Dice()),
        "Metrics that violate the triangle inequality are not valid distance metrics.");
}

} // namespace test
} // namespace OEFP
