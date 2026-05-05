#include <gtest/gtest.h>

#include "oefp/compare.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

TEST(CompareTest, AsymmetricTverskyDependsOnDirection) {
    const auto a = fingerprint_with_bits(64, {0, 1, 2, 3});
    const auto b = fingerprint_with_bits(64, {0, 4});
    const auto metric = Metric::Tversky(0.25, 0.75);

    EXPECT_NEAR(Compare(a, b, metric), 1.0 / 2.5, 1.0e-12);
    EXPECT_NEAR(Compare(b, a, metric), 1.0 / 3.5, 1.0e-12);
    EXPECT_NE(Compare(a, b, metric), Compare(b, a, metric));
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

TEST(CompareBatchTest, QueryToBatchMatchesScalarComparison) {
    const auto query = fingerprint_with_bits(128, {0, 1, 64});
    const auto first = fingerprint_with_bits(128, {0, 1, 64});
    const auto second = fingerprint_with_bits(128, {1, 2});
    const auto third = fingerprint_with_bits(128, {70});
    const auto batch = OEFPBatch::FromFingerprints({first, second, third});

    const auto values = Compare(query, batch, Metric::Tanimoto(MetricMode::Similarity));

    ASSERT_EQ(values.size(), batch.Size());
    EXPECT_DOUBLE_EQ(values[0], Compare(query, first, Metric::Tanimoto()));
    EXPECT_DOUBLE_EQ(values[1], Compare(query, second, Metric::Tanimoto()));
    EXPECT_DOUBLE_EQ(values[2], Compare(query, third, Metric::Tanimoto()));
}

TEST(CompareBatchTest, CDistReturnsRowMajorValues) {
    const auto a = OEFPBatch::FromFingerprints({
        fingerprint_with_bits(64, {0, 1}),
        fingerprint_with_bits(64, {2, 3}),
    });
    const auto b = OEFPBatch::FromFingerprints({
        fingerprint_with_bits(64, {0, 1}),
        fingerprint_with_bits(64, {3}),
        fingerprint_with_bits(64, {9}),
    });

    const auto values = CDist(a, b, Metric::Tanimoto(MetricMode::Similarity));

    ASSERT_EQ(values.size(), 6u);
    EXPECT_DOUBLE_EQ(values[0], 1.0);
    EXPECT_DOUBLE_EQ(values[1], 0.0);
    EXPECT_DOUBLE_EQ(values[2], 0.0);
    EXPECT_DOUBLE_EQ(values[3], 0.0);
    EXPECT_DOUBLE_EQ(values[4], 0.5);
    EXPECT_DOUBLE_EQ(values[5], 0.0);
}

TEST(CompareBatchTest, PDistReturnsCondensedValues) {
    const auto batch = OEFPBatch::FromFingerprints({
        fingerprint_with_bits(64, {0, 1}),
        fingerprint_with_bits(64, {0}),
        fingerprint_with_bits(64, {2}),
    });

    const auto values = PDist(batch, Metric::Tanimoto(MetricMode::Similarity));

    ASSERT_EQ(values.size(), 3u);
    EXPECT_DOUBLE_EQ(values[0], 0.5);
    EXPECT_DOUBLE_EQ(values[1], 0.0);
    EXPECT_DOUBLE_EQ(values[2], 0.0);
}

TEST(CompareBatchTest, PDistRejectsAsymmetricMetric) {
    const auto batch = OEFPBatch::FromFingerprints({
        fingerprint_with_bits(64, {0, 1}),
        fingerprint_with_bits(64, {0}),
    });

    EXPECT_THROW(PDist(batch, Metric::Tversky(0.9, 0.1)), std::invalid_argument);
}

TEST(CompareBatchTest, IntoFunctionsRejectIncorrectOutputLengths) {
    const auto query = fingerprint_with_bits(64, {0, 1});
    const auto batch = OEFPBatch::FromFingerprints({
        fingerprint_with_bits(64, {0, 1}),
        fingerprint_with_bits(64, {2}),
    });
    std::vector<double> output(1, 0.0);

    EXPECT_THROW(
        CompareInto(query, batch, Metric::Tanimoto(), output.data(), output.size()),
        std::invalid_argument);
    EXPECT_THROW(
        CDistInto(batch, batch, Metric::Tanimoto(), output.data(), output.size()),
        std::invalid_argument);
    EXPECT_THROW(
        PDistInto(batch, Metric::Tanimoto(), nullptr, 0),
        std::invalid_argument);
}

TEST(CompareBatchTest, ThreadedPDistMatchesSingleThreadedOutput) {
    std::vector<OEFP> fingerprints;
    fingerprints.reserve(32);
    for (std::uint64_t i = 0; i < 32; ++i) {
        auto fp = fingerprint_with_bits(128, {i, (i * 7) % 128, (i * 13) % 128});
        fingerprints.push_back(fp);
    }
    const auto batch = OEFPBatch::FromFingerprints(fingerprints);

    BatchKernelOptions single_thread;
    single_thread.num_threads = 1;
    single_thread.chunk_size = 3;
    BatchKernelOptions multi_thread;
    multi_thread.num_threads = 4;
    multi_thread.chunk_size = 3;

    const auto expected = PDist(batch, Metric::Tanimoto(), single_thread);
    const auto actual = PDist(batch, Metric::Tanimoto(), multi_thread);

    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_DOUBLE_EQ(actual[i], expected[i]);
    }
}

TEST(CompareBatchTest, AddressHelpersWriteIntoCallerOwnedOutput) {
    const auto query = fingerprint_with_bits(64, {0, 1});
    const auto batch = OEFPBatch::FromFingerprints({
        fingerprint_with_bits(64, {0, 1}),
        fingerprint_with_bits(64, {0}),
    });
    std::vector<double> query_output(2, -1.0);
    std::vector<double> cdist_output(4, -1.0);
    std::vector<double> pdist_output(1, -1.0);

    CompareIntoAddress(
        query,
        batch,
        Metric::Tanimoto(),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(query_output.data())),
        query_output.size());
    CDistIntoAddress(
        batch,
        batch,
        Metric::Tanimoto(),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(cdist_output.data())),
        cdist_output.size());
    PDistIntoAddress(
        batch,
        Metric::Tanimoto(),
        static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(pdist_output.data())),
        pdist_output.size());

    EXPECT_DOUBLE_EQ(query_output[0], 1.0);
    EXPECT_DOUBLE_EQ(query_output[1], 0.5);
    EXPECT_DOUBLE_EQ(cdist_output[0], 1.0);
    EXPECT_DOUBLE_EQ(cdist_output[1], 0.5);
    EXPECT_DOUBLE_EQ(cdist_output[2], 0.5);
    EXPECT_DOUBLE_EQ(cdist_output[3], 1.0);
    EXPECT_DOUBLE_EQ(pdist_output[0], 0.5);
}

} // namespace test
} // namespace OEFP
