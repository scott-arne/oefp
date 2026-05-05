#include "oefp/compare.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace OEFP {
namespace {

std::uint64_t count_bits(std::uint64_t word) {
#if defined(_MSC_VER)
    return static_cast<std::uint64_t>(__popcnt64(word));
#elif defined(__clang__) || defined(__GNUC__)
    return static_cast<std::uint64_t>(
        __builtin_popcountll(static_cast<unsigned long long>(word)));
#else
    std::uint64_t count = 0;
    while (word != 0) {
        word &= word - 1;
        ++count;
    }
    return count;
#endif
}

double zero_safe_divide(double numerator, double denominator) {
    if (denominator == 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}

double apply_mode(double similarity, MetricMode mode) {
    if (mode == MetricMode::Distance) {
        return 1.0 - similarity;
    }
    return similarity;
}

} // namespace

double Compare(const OEFP& a, const OEFP& b, const Metric& metric) {
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Fingerprint specifications must match for comparison.");
    }

    std::uint64_t intersection = 0;
    std::uint64_t only_a = 0;
    std::uint64_t only_b = 0;
    std::uint64_t xor_count = 0;

    const auto& a_words = a.Words();
    const auto& b_words = b.Words();
    for (std::size_t i = 0; i < a_words.size(); ++i) {
        const auto aw = a_words[i];
        const auto bw = b_words[i];
        intersection += count_bits(aw & bw);
        only_a += count_bits(aw & ~bw);
        only_b += count_bits(~aw & bw);
        xor_count += count_bits(aw ^ bw);
    }

    const auto count_a = intersection + only_a;
    const auto count_b = intersection + only_b;
    const auto union_count = intersection + only_a + only_b;

    switch (metric.Kind()) {
    case MetricKind::Tanimoto:
    case MetricKind::Jaccard:
        return apply_mode(
            zero_safe_divide(static_cast<double>(intersection), static_cast<double>(union_count)),
            metric.Mode());
    case MetricKind::Tversky:
        return apply_mode(
            zero_safe_divide(
                static_cast<double>(intersection),
                static_cast<double>(intersection)
                    + metric.Alpha() * static_cast<double>(only_a)
                    + metric.Beta() * static_cast<double>(only_b)),
            metric.Mode());
    case MetricKind::Dice:
        return apply_mode(
            zero_safe_divide(
                2.0 * static_cast<double>(intersection),
                static_cast<double>(count_a + count_b)),
            metric.Mode());
    case MetricKind::Cosine:
        return apply_mode(
            zero_safe_divide(
                static_cast<double>(intersection),
                std::sqrt(static_cast<double>(count_a) * static_cast<double>(count_b))),
            metric.Mode());
    case MetricKind::Manhattan:
        return static_cast<double>(xor_count);
    }

    throw std::invalid_argument("Unsupported fingerprint metric.");
}

} // namespace OEFP
