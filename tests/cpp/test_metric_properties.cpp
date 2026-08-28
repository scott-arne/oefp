#include <gtest/gtest.h>

#include "oefp/compare.h"
#include "oefp/count.h"
#include "oefp/fingerprint.h"
#include "oefp/metric.h"

#include <cstdint>
#include <string>
#include <vector>

// The capability predicates on Metric are a hand-maintained table of claims about the metric
// formulas. These tests measure the claims against the formulas themselves, so a change to a
// formula that breaks a claimed property fails the build rather than silently making the
// predicate lie.

namespace OEFP {
namespace test {
namespace {

constexpr double TRIANGLE_TOLERANCE = 1.0e-12;

struct LabelledMetric {
    const char* label;
    Metric metric;
};

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

FingerprintSpec count_spec(std::uint64_t size_bits) {
    auto spec = binary_spec(size_bits);
    spec.value_type = FingerprintValueType::Counted;
    spec.source_type = "count";
    return spec;
}

/// \brief Build a fingerprint whose set bits are the set bits of mask.
OEFP fingerprint_from_mask(std::uint64_t size_bits, std::uint64_t mask) {
    OEFP fingerprint(binary_spec(size_bits));
    for (std::uint64_t bit = 0; bit < 64u && (mask >> bit) != 0u; ++bit) {
        if (((mask >> bit) & 1u) != 0u) {
            fingerprint.SetBit(bit);
        }
    }
    return fingerprint;
}

OEFPCount count_from_values(const std::vector<std::uint32_t>& values) {
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> counts;
    for (std::uint32_t index = 0; index < values.size(); ++index) {
        if (values[index] != 0u) {
            indices.push_back(index);
            counts.push_back(values[index]);
        }
    }
    return OEFPCount(count_spec(values.size()), std::move(indices), std::move(counts));
}

std::vector<LabelledMetric> boolean_metrics() {
    return {
        {"jaccard", Metric::Jaccard()},
        {"matching", Metric::Matching()},
        {"dice", Metric::Dice()},
        {"kulsinski", Metric::Kulsinski()},
        {"rogers_tanimoto", Metric::RogersTanimoto()},
        {"russell_rao", Metric::RussellRao()},
        {"sokal_michener", Metric::SokalMichener()},
        {"sokal_sneath", Metric::SokalSneath()},
        {"tanimoto", Metric::Tanimoto()},
        {"tversky", Metric::Tversky(0.5, 0.5)},
    };
}

/// \brief Numeric metrics evaluated over count fingerprints of width three.
///
/// Haversine is excluded: its inputs are radian coordinates rather than descriptor values, so
/// sweeping it over arbitrary count vectors would test the formula outside its domain. The
/// parameterized metrics are given the parameters that satisfy their documented preconditions.
std::vector<LabelledMetric> numeric_metrics() {
    return {
        {"euclidean", Metric::Euclidean()},
        {"manhattan", Metric::Manhattan()},
        {"chebyshev", Metric::Chebyshev()},
        {"minkowski(0.5)", Metric::Minkowski(0.5)},
        {"minkowski(1.0)", Metric::Minkowski(1.0)},
        {"minkowski(3.0)", Metric::Minkowski(3.0)},
        {"hamming", Metric::Hamming()},
        {"canberra", Metric::Canberra()},
        {"bray_curtis", Metric::BrayCurtis()},
        {"standardized_euclidean", Metric::StandardizedEuclidean({1.0, 2.0, 4.0})},
        {"mahalanobis",
         Metric::Mahalanobis({
             2.0, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 3.0,
         })},
    };
}

/// \brief Report whether every pair in the cached matrix satisfies d(a, c) <= d(a, b) + d(b, c).
///
/// On violation, witness receives the offending triple.
bool measure_triangle_inequality(
    const std::vector<double>& distances,
    std::size_t count,
    std::string& witness) {
    for (std::size_t a = 0; a < count; ++a) {
        for (std::size_t b = 0; b < count; ++b) {
            const auto first_leg = distances[a * count + b];
            for (std::size_t c = 0; c < count; ++c) {
                const auto direct = distances[a * count + c];
                const auto detour = first_leg + distances[b * count + c];
                if (direct > detour + TRIANGLE_TOLERANCE) {
                    witness = "triple (" + std::to_string(a) + ", " + std::to_string(b) + ", "
                        + std::to_string(c) + "): " + std::to_string(direct) + " > "
                        + std::to_string(detour);
                    return false;
                }
            }
        }
    }
    return true;
}

bool measure_zero_self_distance(const std::vector<double>& distances, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index) {
        if (distances[index * count + index] != 0.0) {
            return false;
        }
    }
    return true;
}

} // namespace

TEST(MetricPropertyTest, BooleanMetricsMatchTheirPredicatesOverEveryFiveBitVector) {
    // Exhaustive rather than sampled. Dice's violation needs nested sets, and random vectors
    // do not produce them: sweeps at 64 and 2048 bits found zero Dice violations.
    constexpr std::uint64_t WIDTH = 5;
    constexpr std::size_t COUNT = std::size_t{1} << WIDTH;

    std::vector<OEFP> vectors;
    vectors.reserve(COUNT);
    for (std::uint64_t mask = 0; mask < COUNT; ++mask) {
        vectors.push_back(fingerprint_from_mask(WIDTH, mask));
    }

    for (const auto& entry : boolean_metrics()) {
        std::vector<double> distances(COUNT * COUNT);
        for (std::size_t a = 0; a < COUNT; ++a) {
            for (std::size_t b = 0; b < COUNT; ++b) {
                distances[a * COUNT + b] = Compare(vectors[a], vectors[b], entry.metric);
            }
        }

        EXPECT_EQ(measure_zero_self_distance(distances, COUNT), entry.metric.HasZeroSelfDistance())
            << entry.label;

        std::string witness;
        EXPECT_EQ(
            measure_triangle_inequality(distances, COUNT, witness),
            entry.metric.SatisfiesTriangleInequality())
            << entry.label << " " << witness;
    }
}

TEST(MetricPropertyTest, NumericMetricsMatchTheirPredicatesOverEveryThreeByThreeCountVector) {
    constexpr std::uint32_t MAX_COUNT = 2;
    constexpr std::size_t COUNT = 27; // Three dimensions, counts drawn from {0, 1, 2}.

    std::vector<OEFPCount> vectors;
    vectors.reserve(COUNT);
    for (std::uint32_t x = 0; x <= MAX_COUNT; ++x) {
        for (std::uint32_t y = 0; y <= MAX_COUNT; ++y) {
            for (std::uint32_t z = 0; z <= MAX_COUNT; ++z) {
                vectors.push_back(count_from_values({x, y, z}));
            }
        }
    }
    ASSERT_EQ(vectors.size(), COUNT);

    for (const auto& entry : numeric_metrics()) {
        std::vector<double> distances(COUNT * COUNT);
        for (std::size_t a = 0; a < COUNT; ++a) {
            for (std::size_t b = 0; b < COUNT; ++b) {
                distances[a * COUNT + b] = Compare(vectors[a], vectors[b], entry.metric);
            }
        }

        EXPECT_EQ(measure_zero_self_distance(distances, COUNT), entry.metric.HasZeroSelfDistance())
            << entry.label;

        std::string witness;
        EXPECT_EQ(
            measure_triangle_inequality(distances, COUNT, witness),
            entry.metric.SatisfiesTriangleInequality())
            << entry.label << " " << witness;
    }
}

TEST(MetricPropertyTest, NestedSetsViolateTheTriangleInequalityAtEveryWidth) {
    // The needle the exhaustive sweep above finds at five bits, pinned at widths a random sweep
    // would realistically use. It is dimension-independent: the three vectors occupy the same
    // two bits however wide the fingerprint is, so the distances do not change with width.
    const auto violators = std::vector<LabelledMetric>{
        {"dice", Metric::Dice()},
        {"bray_curtis", Metric::BrayCurtis()},
        {"minkowski(0.5)", Metric::Minkowski(0.5)},
    };

    for (const std::uint64_t width : std::vector<std::uint64_t>{2, 8, 64, 2048}) {
        const auto a = fingerprint_from_mask(width, 0b01);
        const auto b = fingerprint_from_mask(width, 0b11);
        const auto c = fingerprint_from_mask(width, 0b10);

        for (const auto& entry : violators) {
            const auto direct = Compare(a, c, entry.metric);
            const auto detour = Compare(a, b, entry.metric) + Compare(b, c, entry.metric);

            EXPECT_GT(direct, detour + TRIANGLE_TOLERANCE)
                << entry.label << " at width " << width;
            EXPECT_FALSE(entry.metric.SatisfiesTriangleInequality()) << entry.label;
        }
    }
}

} // namespace test
} // namespace OEFP
