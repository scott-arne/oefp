#include <gtest/gtest.h>

#include "oefp/metric.h"

#include <limits>
#include <stdexcept>
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

} // namespace test
} // namespace OEFP
