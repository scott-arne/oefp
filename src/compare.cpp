#include "oefp/compare.h"

#include "thread_pool.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

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

struct DenseCounts {
    std::uint64_t a = 0;
    std::uint64_t b = 0;
    std::uint64_t intersection = 0;
    std::uint64_t xor_count = 0;
};

std::size_t checked_product(std::size_t a, std::size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::invalid_argument(label);
    }
    return a * b;
}

std::size_t condensed_size(std::size_t n) {
    if (n < 2) {
        return 0;
    }

    auto left = n;
    auto right = n - 1;
    if (left % 2 == 0) {
        left /= 2;
    } else {
        right /= 2;
    }
    return checked_product(left, right, "Pairwise output size is too large.");
}

std::size_t condensed_index(std::size_t n, std::size_t i, std::size_t j) {
    return n * i - i * (i + 1) / 2 + j - i - 1;
}

double zero_safe_divide(double numerator, double denominator) {
    if (denominator == 0.0) {
        return 0.0;
    }
    return numerator / denominator;
}

double apply_mode(double similarity, MetricMode mode) {
    switch (mode) {
    case MetricMode::Similarity:
        return similarity;
    case MetricMode::Distance:
        return 1.0 - similarity;
    }

    throw std::invalid_argument("Metric mode is invalid.");
}

DenseCounts count_dense_pair(
    const std::uint64_t* a_words,
    const std::uint64_t* b_words,
    std::size_t word_count,
    std::uint64_t a_popcount,
    std::uint64_t b_popcount) {
    DenseCounts counts;
    counts.a = a_popcount;
    counts.b = b_popcount;

    for (std::size_t i = 0; i < word_count; ++i) {
        const auto aw = a_words[i];
        const auto bw = b_words[i];
        counts.intersection += count_bits(aw & bw);
        counts.xor_count += count_bits(aw ^ bw);
    }
    return counts;
}

double evaluate_metric(const DenseCounts& counts, const Metric& metric) {
    const auto only_a = counts.a - counts.intersection;
    const auto only_b = counts.b - counts.intersection;
    const auto union_count = counts.intersection + only_a + only_b;

    switch (metric.Kind()) {
    case MetricKind::Tanimoto:
    case MetricKind::Jaccard:
        return apply_mode(
            zero_safe_divide(static_cast<double>(counts.intersection), static_cast<double>(union_count)),
            metric.Mode());
    case MetricKind::Tversky:
        return apply_mode(
            zero_safe_divide(
                static_cast<double>(counts.intersection),
                static_cast<double>(counts.intersection)
                    + metric.Alpha() * static_cast<double>(only_a)
                    + metric.Beta() * static_cast<double>(only_b)),
            metric.Mode());
    case MetricKind::Dice:
        return apply_mode(
            zero_safe_divide(
                2.0 * static_cast<double>(counts.intersection),
                static_cast<double>(counts.a + counts.b)),
            metric.Mode());
    case MetricKind::Cosine:
        return apply_mode(
            zero_safe_divide(
                static_cast<double>(counts.intersection),
                std::sqrt(static_cast<double>(counts.a) * static_cast<double>(counts.b))),
            metric.Mode());
    case MetricKind::Manhattan:
        return static_cast<double>(counts.xor_count);
    }

    throw std::invalid_argument("Unsupported fingerprint metric.");
}

void validate_output(double* output, std::size_t output_length, std::size_t expected_length) {
    if (output_length != expected_length) {
        throw std::invalid_argument("Output length does not match requested comparison shape.");
    }
    if (expected_length != 0 && output == nullptr) {
        throw std::invalid_argument("Output pointer cannot be null for non-empty comparison output.");
    }
}

void validate_fingerprint_batch_compatibility(const OEFP& query, const OEFPBatch& library) {
    if (library.Size() == 0) {
        return;
    }
    if (query.Spec() != library.Spec()) {
        throw std::invalid_argument("Fingerprint specification must match batch specification.");
    }
    if (query.WordCount() != library.WordsPerFingerprint()) {
        throw std::invalid_argument("Fingerprint word count must match batch row width.");
    }
}

void validate_batch_compatibility(const OEFPBatch& a, const OEFPBatch& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return;
    }
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Batch fingerprint specifications must match.");
    }
    if (a.WordsPerFingerprint() != b.WordsPerFingerprint()) {
        throw std::invalid_argument("Batch row widths must match.");
    }
}

double* address_to_output(std::uint64_t output_address) {
    return reinterpret_cast<double*>(static_cast<std::uintptr_t>(output_address));
}

} // namespace

double Compare(const OEFP& a, const OEFP& b, const Metric& metric) {
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Fingerprint specifications must match for comparison.");
    }

    if (a.WordCount() != b.WordCount()) {
        throw std::invalid_argument("Fingerprint word counts must match for comparison.");
    }

    const auto& a_words = a.Words();
    const auto& b_words = b.Words();
    const auto counts = count_dense_pair(
        a_words.data(),
        b_words.data(),
        a_words.size(),
        a.CountOnBits(),
        b.CountOnBits());
    return evaluate_metric(counts, metric);
}

std::vector<double> Compare(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(library.Size(), 0.0);
    CompareInto(query, library, metric, output.data(), output.size(), options);
    return output;
}

void CompareInto(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    validate_output(output, output_length, library.Size());
    validate_fingerprint_batch_compatibility(query, library);

    const auto& query_words = query.Words();
    const auto query_popcount = query.CountOnBits();
    const auto word_count = query.WordCount();
    detail::ParallelFor(0, library.Size(), options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            const auto counts = count_dense_pair(
                query_words.data(),
                library.RowWords(row),
                word_count,
                query_popcount,
                library.PopCount(row));
            output[row] = evaluate_metric(counts, metric);
        }
    });
}

std::vector<double> CDist(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(checked_product(a.Size(), b.Size(), "CDist output size is too large."), 0.0);
    CDistInto(a, b, metric, output.data(), output.size(), options);
    return output;
}

void CDistInto(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    const auto expected_length = checked_product(a.Size(), b.Size(), "CDist output size is too large.");
    validate_output(output, output_length, expected_length);
    validate_batch_compatibility(a, b);

    const auto word_count = a.WordsPerFingerprint();
    detail::ParallelFor(0, a.Size(), options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row_a = begin; row_a < end; ++row_a) {
            const auto* a_words = a.RowWords(row_a);
            for (std::size_t row_b = 0; row_b < b.Size(); ++row_b) {
                const auto counts = count_dense_pair(
                    a_words,
                    b.RowWords(row_b),
                    word_count,
                    a.PopCount(row_a),
                    b.PopCount(row_b));
                output[row_a * b.Size() + row_b] = evaluate_metric(counts, metric);
            }
        }
    });
}

std::vector<double> PDist(
    const OEFPBatch& batch,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(condensed_size(batch.Size()), 0.0);
    PDistInto(batch, metric, output.data(), output.size(), options);
    return output;
}

void PDistInto(
    const OEFPBatch& batch,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    metric.ValidateForPDist();
    const auto expected_length = condensed_size(batch.Size());
    validate_output(output, output_length, expected_length);

    const auto word_count = batch.WordsPerFingerprint();
    detail::ParallelFor(0, batch.Size(), options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row_a = begin; row_a < end; ++row_a) {
            const auto* a_words = batch.RowWords(row_a);
            for (std::size_t row_b = row_a + 1; row_b < batch.Size(); ++row_b) {
                const auto counts = count_dense_pair(
                    a_words,
                    batch.RowWords(row_b),
                    word_count,
                    batch.PopCount(row_a),
                    batch.PopCount(row_b));
                output[condensed_index(batch.Size(), row_a, row_b)] = evaluate_metric(counts, metric);
            }
        }
    });
}

void CompareIntoAddress(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CompareInto(query, library, metric, address_to_output(output_address), output_length, options);
}

void CDistIntoAddress(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    CDistInto(a, b, metric, address_to_output(output_address), output_length, options);
}

void PDistIntoAddress(
    const OEFPBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    PDistInto(batch, metric, address_to_output(output_address), output_length, options);
}

} // namespace OEFP
