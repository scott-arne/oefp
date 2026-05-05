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

struct SparseCountStats {
    double a = 0.0;
    double b = 0.0;
    double overlap = 0.0;
    double union_count = 0.0;
    double dot = 0.0;
    double square_product = 0.0;
    double l1 = 0.0;
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

void condensed_pair_from_index(
    std::size_t index,
    std::size_t n,
    std::size_t& i,
    std::size_t& j) {
    const auto n_double = static_cast<double>(n);
    const auto index_double = static_cast<double>(index);
    const auto row = n_double - 2.0
        - std::floor(
            std::sqrt(-8.0 * index_double + 4.0 * n_double * (n_double - 1.0) - 7.0) / 2.0 - 0.5);
    i = static_cast<std::size_t>(row);
    j = index + i + 1 - n * (n - 1) / 2 + (n - i) * ((n - i) - 1) / 2;
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

SparseCountStats count_sparse_rows(
    const std::uint32_t* a_indices,
    const std::uint32_t* a_counts,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    const std::uint32_t* b_counts,
    std::size_t b_size) {
    SparseCountStats stats;
    double a_square = 0.0;
    double b_square = 0.0;
    std::size_t a_row = 0;
    std::size_t b_row = 0;

    while (a_row < a_size || b_row < b_size) {
        if (b_row == b_size || (a_row < a_size && a_indices[a_row] < b_indices[b_row])) {
            const auto a_count = static_cast<double>(a_counts[a_row]);
            stats.a += a_count;
            stats.union_count += a_count;
            stats.l1 += a_count;
            a_square += a_count * a_count;
            ++a_row;
        } else if (a_row == a_size || b_indices[b_row] < a_indices[a_row]) {
            const auto b_count = static_cast<double>(b_counts[b_row]);
            stats.b += b_count;
            stats.union_count += b_count;
            stats.l1 += b_count;
            b_square += b_count * b_count;
            ++b_row;
        } else {
            const auto a_count = static_cast<double>(a_counts[a_row]);
            const auto b_count = static_cast<double>(b_counts[b_row]);
            stats.a += a_count;
            stats.b += b_count;
            stats.overlap += a_count < b_count ? a_count : b_count;
            stats.union_count += a_count > b_count ? a_count : b_count;
            stats.dot += a_count * b_count;
            stats.l1 += a_count > b_count ? a_count - b_count : b_count - a_count;
            a_square += a_count * a_count;
            b_square += b_count * b_count;
            ++a_row;
            ++b_row;
        }
    }

    stats.square_product = a_square * b_square;
    return stats;
}

SparseCountStats count_sparse_pair(const OEFPCount& a, const OEFPCount& b) {
    return count_sparse_rows(
        a.IndexData(),
        a.CountData(),
        a.NonzeroCount(),
        b.IndexData(),
        b.CountData(),
        b.NonzeroCount());
}

DenseCounts count_sparse_binary_rows(
    const std::uint32_t* a_indices,
    std::size_t a_size,
    const std::uint32_t* b_indices,
    std::size_t b_size) {
    DenseCounts counts;
    counts.a = static_cast<std::uint64_t>(a_size);
    counts.b = static_cast<std::uint64_t>(b_size);

    std::size_t a_row = 0;
    std::size_t b_row = 0;
    while (a_row < a_size && b_row < b_size) {
        if (a_indices[a_row] < b_indices[b_row]) {
            ++a_row;
        } else if (b_indices[b_row] < a_indices[a_row]) {
            ++b_row;
        } else {
            ++counts.intersection;
            ++a_row;
            ++b_row;
        }
    }

    counts.xor_count = counts.a + counts.b - 2u * counts.intersection;
    return counts;
}

DenseCounts count_sparse_binary_pair(const OEFPSparse& a, const OEFPSparse& b) {
    return count_sparse_binary_rows(
        a.IndexData(),
        a.CountOnBits(),
        b.IndexData(),
        b.CountOnBits());
}

double evaluate_count_metric(const SparseCountStats& stats, const Metric& metric) {
    const auto only_a = stats.a - stats.overlap;
    const auto only_b = stats.b - stats.overlap;

    switch (metric.Kind()) {
    case MetricKind::Tanimoto:
    case MetricKind::Jaccard:
        return apply_mode(zero_safe_divide(stats.overlap, stats.union_count), metric.Mode());
    case MetricKind::Tversky:
        return apply_mode(
            zero_safe_divide(
                stats.overlap,
                stats.overlap + metric.Alpha() * only_a + metric.Beta() * only_b),
            metric.Mode());
    case MetricKind::Dice:
        return apply_mode(zero_safe_divide(2.0 * stats.overlap, stats.a + stats.b), metric.Mode());
    case MetricKind::Cosine:
        return apply_mode(
            zero_safe_divide(stats.dot, std::sqrt(stats.square_product)),
            metric.Mode());
    case MetricKind::Manhattan:
        return stats.l1;
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

void validate_count_fingerprint_batch_compatibility(
    const OEFPCount& query,
    const OEFPCountBatch& library) {
    if (library.Size() == 0) {
        return;
    }
    if (query.Spec() != library.Spec()) {
        throw std::invalid_argument("Count fingerprint specification must match batch specification.");
    }
}

void validate_count_batch_compatibility(const OEFPCountBatch& a, const OEFPCountBatch& b) {
    if (a.Size() == 0 || b.Size() == 0) {
        return;
    }
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Count batch fingerprint specifications must match.");
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

double Compare(const OEFPCount& a, const OEFPCount& b, const Metric& metric) {
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Count fingerprint specifications must match for comparison.");
    }

    return evaluate_count_metric(count_sparse_pair(a, b), metric);
}

double Compare(const OEFPSparse& a, const OEFPSparse& b, const Metric& metric) {
    if (a.Spec() != b.Spec()) {
        throw std::invalid_argument("Sparse fingerprint specifications must match for comparison.");
    }

    return evaluate_metric(count_sparse_binary_pair(a, b), metric);
}

std::vector<double> Compare(
    const OEFPCount& query,
    const OEFPCountBatch& library,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(library.Size(), 0.0);
    CompareInto(query, library, metric, output.data(), output.size(), options);
    return output;
}

void CompareInto(
    const OEFPCount& query,
    const OEFPCountBatch& library,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    validate_output(output, output_length, library.Size());
    validate_count_fingerprint_batch_compatibility(query, library);

    const auto* query_indices = query.IndexData();
    const auto* query_counts = query.CountData();
    const auto query_size = query.NonzeroCount();
    detail::ParallelFor(0, library.Size(), options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t row = begin; row < end; ++row) {
            const auto counts = count_sparse_rows(
                query_indices,
                query_counts,
                query_size,
                library.RowIndices(row),
                library.RowCounts(row),
                library.RowEntryCount(row));
            output[row] = evaluate_count_metric(counts, metric);
        }
    });
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
    const auto b_size = b.Size();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        auto row_a = begin / b_size;
        auto row_b = begin % b_size;
        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint64_t* a_words = nullptr;
        std::uint32_t a_popcount = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_words = a.RowWords(row_a);
                a_popcount = a.PopCount(row_a);
            }
            const auto counts = count_dense_pair(
                a_words,
                b.RowWords(row_b),
                word_count,
                a_popcount,
                b.PopCount(row_b));
            output[output_index] = evaluate_metric(counts, metric);

            ++row_b;
            if (row_b == b_size) {
                row_b = 0;
                ++row_a;
            }
        }
    });
}

std::vector<double> CDist(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(checked_product(a.Size(), b.Size(), "CDist output size is too large."), 0.0);
    CDistInto(a, b, metric, output.data(), output.size(), options);
    return output;
}

void CDistInto(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    const auto expected_length = checked_product(a.Size(), b.Size(), "CDist output size is too large.");
    validate_output(output, output_length, expected_length);
    validate_count_batch_compatibility(a, b);

    const auto b_size = b.Size();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        auto row_a = begin / b_size;
        auto row_b = begin % b_size;
        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint32_t* a_indices = nullptr;
        const std::uint32_t* a_counts = nullptr;
        std::size_t a_count = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_indices = a.RowIndices(row_a);
                a_counts = a.RowCounts(row_a);
                a_count = a.RowEntryCount(row_a);
            }
            const auto counts = count_sparse_rows(
                a_indices,
                a_counts,
                a_count,
                b.RowIndices(row_b),
                b.RowCounts(row_b),
                b.RowEntryCount(row_b));
            output[output_index] = evaluate_count_metric(counts, metric);

            ++row_b;
            if (row_b == b_size) {
                row_b = 0;
                ++row_a;
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

    const auto batch_size = batch.Size();
    const auto word_count = batch.WordsPerFingerprint();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        std::size_t row_a = 0;
        std::size_t row_b = 0;
        condensed_pair_from_index(begin, batch_size, row_a, row_b);

        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint64_t* a_words = nullptr;
        std::uint32_t a_popcount = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_words = batch.RowWords(row_a);
                a_popcount = batch.PopCount(row_a);
            }
            const auto counts = count_dense_pair(
                a_words,
                batch.RowWords(row_b),
                word_count,
                a_popcount,
                batch.PopCount(row_b));
            output[output_index] = evaluate_metric(counts, metric);

            ++row_b;
            if (row_b == batch_size) {
                ++row_a;
                row_b = row_a + 1;
            }
        }
    });
}

std::vector<double> PDist(
    const OEFPCountBatch& batch,
    const Metric& metric,
    const BatchKernelOptions& options) {
    std::vector<double> output(condensed_size(batch.Size()), 0.0);
    PDistInto(batch, metric, output.data(), output.size(), options);
    return output;
}

void PDistInto(
    const OEFPCountBatch& batch,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    metric.ValidateForPDist();
    const auto expected_length = condensed_size(batch.Size());
    validate_output(output, output_length, expected_length);

    const auto batch_size = batch.Size();
    detail::ParallelFor(0, expected_length, options.chunk_size, options.num_threads, [&](std::size_t begin, std::size_t end) {
        std::size_t row_a = 0;
        std::size_t row_b = 0;
        condensed_pair_from_index(begin, batch_size, row_a, row_b);

        std::size_t cached_row_a = std::numeric_limits<std::size_t>::max();
        const std::uint32_t* a_indices = nullptr;
        const std::uint32_t* a_counts = nullptr;
        std::size_t a_count = 0;

        for (std::size_t output_index = begin; output_index < end; ++output_index) {
            if (row_a != cached_row_a) {
                cached_row_a = row_a;
                a_indices = batch.RowIndices(row_a);
                a_counts = batch.RowCounts(row_a);
                a_count = batch.RowEntryCount(row_a);
            }
            const auto counts = count_sparse_rows(
                a_indices,
                a_counts,
                a_count,
                batch.RowIndices(row_b),
                batch.RowCounts(row_b),
                batch.RowEntryCount(row_b));
            output[output_index] = evaluate_count_metric(counts, metric);

            ++row_b;
            if (row_b == batch_size) {
                ++row_a;
                row_b = row_a + 1;
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

void CompareIntoAddress(
    const OEFPCount& query,
    const OEFPCountBatch& library,
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

void CDistIntoAddress(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
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

void PDistIntoAddress(
    const OEFPCountBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options) {
    PDistInto(batch, metric, address_to_output(output_address), output_length, options);
}

} // namespace OEFP
