#include <gtest/gtest.h>

#include "oefp/compare.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace OEFP {
namespace test {
namespace {

FingerprintSpec binary_spec(std::uint64_t size_bits) {
    FingerprintSpec spec;
    spec.size_bits = size_bits;
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "unit-test";
    spec.source_type = "dense";
    spec.source_version = "1";
    spec.parameters = "size=" + std::to_string(size_bits);
    return spec;
}

OEFP fingerprint_with_bits(std::uint64_t size_bits, std::initializer_list<std::uint64_t> bits) {
    OEFP fp(binary_spec(size_bits));
    for (const auto bit : bits) {
        fp.SetBit(bit);
    }
    return fp;
}

} // namespace

TEST(CompareTest, BatchKernelOptionsDefaultsAreStable) {
    const BatchKernelOptions options;

    EXPECT_EQ(options.num_threads, 0u);
    EXPECT_EQ(options.chunk_size, 256u);
}

TEST(CompareTest, ComputesTanimotoAndJaccardSimilarityAndDistance) {
    const auto a = fingerprint_with_bits(128, {0, 1, 64, 100});
    const auto b = fingerprint_with_bits(128, {1, 64, 65, 100, 127});

    EXPECT_NEAR(Compare(a, b, Metric::Tanimoto()), 3.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(Compare(a, b, Metric::Tanimoto(MetricMode::Distance)), 1.0 - 3.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(Compare(a, b, Metric::Jaccard()), 3.0 / 6.0, 1.0e-12);
    EXPECT_NEAR(Compare(a, b, Metric::Jaccard(MetricMode::Distance)), 1.0 - 3.0 / 6.0, 1.0e-12);
}

TEST(CompareTest, ComputesDiceCosineAndTversky) {
    const auto a = fingerprint_with_bits(130, {0, 1, 64, 129});
    const auto b = fingerprint_with_bits(130, {1, 64, 100});

    EXPECT_NEAR(Compare(a, b, Metric::Dice()), 4.0 / 7.0, 1.0e-12);
    EXPECT_NEAR(Compare(a, b, Metric::Dice(MetricMode::Distance)), 1.0 - 4.0 / 7.0, 1.0e-12);
    EXPECT_NEAR(Compare(a, b, Metric::Cosine()), 2.0 / std::sqrt(12.0), 1.0e-12);
    EXPECT_NEAR(Compare(a, b, Metric::Cosine(MetricMode::Distance)), 1.0 - 2.0 / std::sqrt(12.0), 1.0e-12);
    EXPECT_NEAR(Compare(a, b, Metric::Tversky(0.25, 0.75)), 2.0 / 3.25, 1.0e-12);
    EXPECT_NEAR(
        Compare(a, b, Metric::Tversky(0.25, 0.75, MetricMode::Distance)),
        1.0 - 2.0 / 3.25,
        1.0e-12);
}

TEST(CompareTest, ComputesManhattanDistance) {
    const auto a = fingerprint_with_bits(128, {0, 1, 64, 100});
    const auto b = fingerprint_with_bits(128, {1, 64, 65, 100, 127});

    EXPECT_EQ(Compare(a, b, Metric::Manhattan()), 3.0);
}

TEST(CompareTest, RejectsMismatchedSpecs) {
    const auto a = fingerprint_with_bits(64, {1, 3});
    const auto different_size = fingerprint_with_bits(65, {1, 3});

    auto metadata_spec = binary_spec(64);
    metadata_spec.source_version = "2";
    OEFP different_metadata(metadata_spec);
    different_metadata.SetBit(1);
    different_metadata.SetBit(3);

    EXPECT_THROW(Compare(a, different_size, Metric::Tanimoto()), std::invalid_argument);
    EXPECT_THROW(Compare(a, different_metadata, Metric::Tanimoto()), std::invalid_argument);
}

TEST(CompareTest, UsesZeroSimilarityForEmptyDenominators) {
    const auto a = fingerprint_with_bits(64, {});
    const auto b = fingerprint_with_bits(64, {});

    EXPECT_EQ(Compare(a, b, Metric::Tanimoto()), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Tanimoto(MetricMode::Distance)), 1.0);
    EXPECT_EQ(Compare(a, b, Metric::Jaccard()), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Dice()), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Dice(MetricMode::Distance)), 1.0);
    EXPECT_EQ(Compare(a, b, Metric::Cosine()), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Cosine(MetricMode::Distance)), 1.0);
    EXPECT_EQ(Compare(a, b, Metric::Tversky(0.5, 0.5)), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Tversky(0.5, 0.5, MetricMode::Distance)), 1.0);
    EXPECT_EQ(Compare(a, b, Metric::Manhattan()), 0.0);
}

TEST(CompareTest, HandlesZeroWidthFingerprintsWithEmptyDenominatorRule) {
    OEFP a(binary_spec(0));
    OEFP b(binary_spec(0));

    EXPECT_EQ(Compare(a, b, Metric::Tanimoto()), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Tanimoto(MetricMode::Distance)), 1.0);
    EXPECT_EQ(Compare(a, b, Metric::Dice()), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Dice(MetricMode::Distance)), 1.0);
    EXPECT_EQ(Compare(a, b, Metric::Cosine()), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Cosine(MetricMode::Distance)), 1.0);
    EXPECT_EQ(Compare(a, b, Metric::Tversky(1.0, 0.0)), 0.0);
    EXPECT_EQ(Compare(a, b, Metric::Tversky(1.0, 0.0, MetricMode::Distance)), 1.0);
    EXPECT_EQ(Compare(a, b, Metric::Manhattan()), 0.0);
}

} // namespace test
} // namespace OEFP
