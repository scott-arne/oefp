#include <gtest/gtest.h>

#include "oefp/metric.h"

#include <limits>
#include <stdexcept>

namespace OEFP {
namespace test {

TEST(MetricTest, FactoriesSetKindModeAndDefaultParameters) {
    const auto tanimoto = Metric::Tanimoto();
    EXPECT_EQ(tanimoto.Kind(), MetricKind::Tanimoto);
    EXPECT_EQ(tanimoto.Mode(), MetricMode::Similarity);
    EXPECT_EQ(tanimoto.Alpha(), 1.0);
    EXPECT_EQ(tanimoto.Beta(), 1.0);
    EXPECT_TRUE(tanimoto.IsSymmetric());

    const auto jaccard = Metric::Jaccard(MetricMode::Distance);
    EXPECT_EQ(jaccard.Kind(), MetricKind::Jaccard);
    EXPECT_EQ(jaccard.Mode(), MetricMode::Distance);
    EXPECT_TRUE(jaccard.IsSymmetric());

    const auto dice = Metric::Dice(MetricMode::Distance);
    EXPECT_EQ(dice.Kind(), MetricKind::Dice);
    EXPECT_EQ(dice.Mode(), MetricMode::Distance);
    EXPECT_TRUE(dice.IsSymmetric());

    const auto cosine = Metric::Cosine();
    EXPECT_EQ(cosine.Kind(), MetricKind::Cosine);
    EXPECT_EQ(cosine.Mode(), MetricMode::Similarity);
    EXPECT_TRUE(cosine.IsSymmetric());

    const auto manhattan = Metric::Manhattan();
    EXPECT_EQ(manhattan.Kind(), MetricKind::Manhattan);
    EXPECT_EQ(manhattan.Mode(), MetricMode::Distance);
    EXPECT_TRUE(manhattan.IsSymmetric());
}

TEST(MetricTest, TverskyStoresAlphaBetaModeAndSymmetry) {
    const auto symmetric = Metric::Tversky(0.5, 0.5, MetricMode::Distance);
    EXPECT_EQ(symmetric.Kind(), MetricKind::Tversky);
    EXPECT_EQ(symmetric.Mode(), MetricMode::Distance);
    EXPECT_EQ(symmetric.Alpha(), 0.5);
    EXPECT_EQ(symmetric.Beta(), 0.5);
    EXPECT_TRUE(symmetric.IsSymmetric());

    const auto asymmetric = Metric::Tversky(0.2, 0.8);
    EXPECT_EQ(asymmetric.Mode(), MetricMode::Similarity);
    EXPECT_EQ(asymmetric.Alpha(), 0.2);
    EXPECT_EQ(asymmetric.Beta(), 0.8);
    EXPECT_FALSE(asymmetric.IsSymmetric());
}

TEST(MetricTest, TverskyRejectsAlphaOrBetaOutsideUnitInterval) {
    EXPECT_THROW(Metric::Tversky(-0.1, 0.5), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(1.1, 0.5), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(0.5, -0.1), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(0.5, 1.1), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(std::numeric_limits<double>::quiet_NaN(), 0.5), std::invalid_argument);
    EXPECT_THROW(Metric::Tversky(0.5, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
    EXPECT_NO_THROW(Metric::Tversky(0.0, 1.0));
}

TEST(MetricTest, ManhattanRejectsSimilarityMode) {
    EXPECT_THROW(Metric::Manhattan(MetricMode::Similarity), std::invalid_argument);
}

TEST(MetricTest, ValidateForPDistAcceptsSymmetricMetrics) {
    EXPECT_NO_THROW(Metric::Tanimoto().ValidateForPDist());
    EXPECT_NO_THROW(Metric::Jaccard().ValidateForPDist());
    EXPECT_NO_THROW(Metric::Tversky(0.3, 0.3).ValidateForPDist());
    EXPECT_NO_THROW(Metric::Dice().ValidateForPDist());
    EXPECT_NO_THROW(Metric::Cosine().ValidateForPDist());
    EXPECT_NO_THROW(Metric::Manhattan().ValidateForPDist());
}

TEST(MetricTest, ValidateForPDistRejectsAsymmetricMetrics) {
    EXPECT_THROW(Metric::Tversky(0.2, 0.8).ValidateForPDist(), std::invalid_argument);
}

} // namespace test
} // namespace OEFP
