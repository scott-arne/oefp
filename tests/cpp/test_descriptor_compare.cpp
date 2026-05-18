#include <gtest/gtest.h>

#include "oefp/compare.h"
#include "oefp/descriptor.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {
namespace {

DescriptorSpec string_spec() {
    DescriptorSpec spec;
    spec.value_type = DescriptorValueType::String;
    spec.source_name = "unit-test";
    spec.source_type = "descriptor";
    spec.source_version = "1";
    spec.parameters = "kind=string";
    return spec;
}

DescriptorSpec integer_spec() {
    auto spec = string_spec();
    spec.value_type = DescriptorValueType::Integer;
    spec.parameters = "kind=integer";
    return spec;
}

DescriptorSpec float_spec() {
    auto spec = string_spec();
    spec.value_type = DescriptorValueType::Float;
    spec.parameters = "kind=float";
    return spec;
}

} // namespace

TEST(DescriptorCompareTest, CountOverlapTanimotoUsesMinMaxCounts) {
    const auto a = DescriptorSet::FromStrings(
        string_spec(),
        {"alpha", "alpha", "alpha", "beta"});
    const auto b = DescriptorSet::FromStrings(
        string_spec(),
        {"alpha", "gamma", "gamma"});

    EXPECT_NEAR(Compare(a, b, Metric::Tanimoto()), 1.0 / 6.0, 1.0e-12);
}

TEST(DescriptorCompareTest, CountOverlapBooleanDimensionsUseWeightedUnion) {
    const auto a = DescriptorSet::FromStrings(string_spec(), {"alpha", "alpha", "beta"});
    const auto b = DescriptorSet::FromStrings(string_spec(), {"alpha", "beta", "beta", "gamma"});

    EXPECT_NEAR(Compare(a, b, Metric::Matching()), 3.0 / 5.0, 1.0e-12);
}

TEST(DescriptorCompareTest, PresenceTanimotoIgnoresCounts) {
    const auto a = DescriptorSet::FromStrings(
        string_spec(),
        {"alpha", "alpha", "alpha", "beta"});
    const auto b = DescriptorSet::FromStrings(
        string_spec(),
        {"alpha", "gamma", "gamma"});

    EXPECT_NEAR(
        Compare(a, b, Metric::Tanimoto(), DescriptorComparisonMode::Presence),
        1.0 / 3.0,
        1.0e-12);
}

TEST(DescriptorCompareTest, ExactCountTanimotoRequiresSameKeyAndCount) {
    const DescriptorSet a(
        string_spec(),
        std::vector<std::string>{"alpha", "beta"},
        std::vector<std::uint32_t>{2u, 1u});
    const DescriptorSet b(
        string_spec(),
        std::vector<std::string>{"alpha", "beta"},
        std::vector<std::uint32_t>{3u, 1u});

    EXPECT_NEAR(
        Compare(a, b, Metric::Tanimoto(), DescriptorComparisonMode::ExactCount),
        1.0 / 3.0,
        1.0e-12);
}

TEST(DescriptorCompareTest, CountOverlapNumericDistancesUseCountVectors) {
    const DescriptorSet a(
        string_spec(),
        std::vector<std::string>{"alpha", "beta"},
        std::vector<std::uint32_t>{3u, 1u});
    const DescriptorSet b(
        string_spec(),
        std::vector<std::string>{"alpha", "gamma"},
        std::vector<std::uint32_t>{1u, 2u});

    EXPECT_EQ(Compare(a, b, Metric::Manhattan()), 5.0);
    EXPECT_NEAR(Compare(a, b, Metric::Euclidean()), 3.0, 1.0e-12);
}

TEST(DescriptorCompareTest, RejectsUnsupportedDensePositionalMetrics) {
    const auto a = DescriptorSet::FromStrings(string_spec(), {"alpha"});
    const auto b = DescriptorSet::FromStrings(string_spec(), {"beta"});

    EXPECT_THROW(
        Compare(a, b, Metric::StandardizedEuclidean({1.0, 1.0})),
        std::invalid_argument);
    EXPECT_THROW(
        Compare(a, b, Metric::Mahalanobis({1.0, 0.0, 0.0, 1.0})),
        std::invalid_argument);
    EXPECT_THROW(Compare(a, b, Metric::Haversine()), std::invalid_argument);
}

TEST(DescriptorCompareTest, SupportsIntegerAndFloatKeys) {
    const auto integers_a = DescriptorSet::FromIntegers(integer_spec(), {-3, -3, 7});
    const auto integers_b = DescriptorSet::FromIntegers(integer_spec(), {-3, 11});
    const auto floats_a = DescriptorSet::FromFloats(float_spec(), {1.5, 2.5, 2.5});
    const auto floats_b = DescriptorSet::FromFloats(float_spec(), {2.5, 3.5});

    EXPECT_NEAR(Compare(integers_a, integers_b, Metric::Tanimoto()), 1.0 / 4.0, 1.0e-12);
    EXPECT_NEAR(Compare(floats_a, floats_b, Metric::Tanimoto()), 1.0 / 4.0, 1.0e-12);
}

TEST(DescriptorCompareTest, QueryToBatchComparisonUsesDescriptorMode) {
    const auto query = DescriptorSet::FromStrings(string_spec(), {"alpha", "alpha", "beta"});
    const auto first = DescriptorSet::FromStrings(string_spec(), {"alpha", "alpha", "beta"});
    const auto second = DescriptorSet::FromStrings(
        string_spec(),
        {"alpha", "beta", "beta", "gamma"});
    const auto batch = DescriptorBatch::FromDescriptorSets({first, second});

    const auto values = Compare(
        query,
        batch,
        Metric::Tanimoto(),
        DescriptorComparisonMode::Presence);

    ASSERT_EQ(values.size(), 2u);
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_NEAR(values[1], 2.0 / 3.0, 1.0e-12);
}

TEST(DescriptorCompareTest, CDistAndPDistSupportDescriptorBatches) {
    const auto first = DescriptorSet::FromStrings(string_spec(), {"alpha"});
    const auto second = DescriptorSet::FromStrings(string_spec(), {"beta"});
    const auto mixed = DescriptorSet::FromStrings(string_spec(), {"alpha", "beta"});
    const auto a = DescriptorBatch::FromDescriptorSets({first, second});
    const auto b = DescriptorBatch::FromDescriptorSets({first, mixed});

    const auto cdist = CDist(a, b, Metric::Jaccard(), DescriptorComparisonMode::Presence);
    const auto pdist = PDist(b, Metric::Jaccard(), DescriptorComparisonMode::Presence);

    ASSERT_EQ(cdist.size(), 4u);
    EXPECT_DOUBLE_EQ(cdist[0], 0.0);
    EXPECT_DOUBLE_EQ(cdist[1], 0.5);
    EXPECT_DOUBLE_EQ(cdist[2], 1.0);
    EXPECT_DOUBLE_EQ(cdist[3], 0.5);
    ASSERT_EQ(pdist.size(), 1u);
    EXPECT_DOUBLE_EQ(pdist[0], 0.5);
}

TEST(DescriptorCompareTest, PDistRejectsAsymmetricMetric) {
    const auto batch = DescriptorBatch::FromDescriptorSets({
        DescriptorSet::FromStrings(string_spec(), {"alpha"}),
        DescriptorSet::FromStrings(string_spec(), {"beta"}),
    });

    EXPECT_THROW(
        PDist(batch, Metric::Tversky(0.2, 0.8)),
        std::invalid_argument);
}

TEST(DescriptorCompareTest, IntoFunctionsRejectIncorrectOutputLengths) {
    const auto query = DescriptorSet::FromStrings(string_spec(), {"alpha"});
    const auto batch = DescriptorBatch::FromDescriptorSets({
        DescriptorSet::FromStrings(string_spec(), {"alpha"}),
        DescriptorSet::FromStrings(string_spec(), {"beta"}),
    });
    std::vector<double> output(1, 0.0);

    EXPECT_THROW(
        CompareInto(
            query,
            batch,
            Metric::Tanimoto(),
            DescriptorComparisonMode::Presence,
            output.data(),
            output.size()),
        std::invalid_argument);
    EXPECT_THROW(
        CDistInto(
            batch,
            batch,
            Metric::Tanimoto(),
            DescriptorComparisonMode::Presence,
            output.data(),
            output.size()),
        std::invalid_argument);
    EXPECT_THROW(
        PDistInto(
            batch,
            Metric::Tanimoto(),
            DescriptorComparisonMode::Presence,
            nullptr,
            0),
        std::invalid_argument);
}

} // namespace test
} // namespace OEFP
