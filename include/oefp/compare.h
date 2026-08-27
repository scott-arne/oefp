#ifndef OEFP_COMPARE_H
#define OEFP_COMPARE_H

#include "oefp/batch.h"
#include "oefp/batch_kernel_options.h"
#include "oefp/count.h"
#include "oefp/count_batch.h"
#include "oefp/descriptor.h"
#include "oefp/descriptor_batch.h"
#include "oefp/descriptor_compare.h"
#include "oefp/fingerprint.h"
#include "oefp/metric.h"
#include "oefp/sparse.h"
#include "oefp/sparse_batch.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Compare two dense binary fingerprints with the requested metric.
///
/// Fingerprint specifications must compare exactly equal, including provenance
/// metadata. This intentionally matches batch admission rules for dense-binary
/// milestone 1.
///
/// \param a First fingerprint.
/// \param b Second fingerprint.
/// \param metric Metric configuration.
/// \returns Similarity or distance according to metric.Type().
/// \throws std::invalid_argument: When fingerprint specifications or storage
///     widths differ.
double Compare(const OEFP& a, const OEFP& b, const Metric& metric);

/// \brief Compare two sparse counted fingerprints with the requested metric.
///
/// Tanimoto, Jaccard, Dice, and Tversky use weighted count overlap semantics
/// compatible with RDKit count fingerprints. Cosine uses weighted dot-product
/// semantics and Manhattan uses L1 count distance.
///
/// \param a First counted fingerprint.
/// \param b Second counted fingerprint.
/// \param metric Metric configuration.
/// \returns Similarity or distance according to metric.Type().
/// \throws std::invalid_argument: When fingerprint specifications differ.
double Compare(const OEFPCount& a, const OEFPCount& b, const Metric& metric);

/// \brief Compare two sparse binary fingerprints with the requested metric.
///
/// Tanimoto, Jaccard, Dice, Cosine, Tversky, and Manhattan use binary set
/// semantics over sorted sparse on-bit identifiers.
///
/// \param a First sparse binary fingerprint.
/// \param b Second sparse binary fingerprint.
/// \param metric Metric configuration.
/// \returns Similarity or distance according to metric.Type().
/// \throws std::invalid_argument: When fingerprint specifications differ.
double Compare(const OEFPSparse& a, const OEFPSparse& b, const Metric& metric);

/// \brief Compare two scalar descriptor sets with the requested metric.
///
/// Descriptor specifications must compare exactly equal. Count-overlap mode
/// evaluates count-aware descriptor overlap by default, presence mode collapses
/// counts to binary key presence, and exact-count mode treats each key/count
/// pair as its own binary feature.
///
/// \param a First descriptor set.
/// \param b Second descriptor set.
/// \param metric Metric configuration.
/// \param mode Descriptor count interpretation mode.
/// \returns Similarity or distance according to metric.Type().
/// \throws std::invalid_argument: When descriptor specifications differ or the
///     metric is not valid for descriptor comparison.
double Compare(
    const DescriptorSet& a,
    const DescriptorSet& b,
    const Metric& metric,
    DescriptorComparisonMode mode = DescriptorComparisonMode::CountOverlap);

/// \brief Compare one descriptor query against each row in a descriptor batch.
std::vector<double> Compare(
    const DescriptorSet& query,
    const DescriptorBatch& library,
    const Metric& metric,
    DescriptorComparisonMode mode = DescriptorComparisonMode::CountOverlap,
    const BatchKernelOptions& options = {});

/// \brief Fill output with one descriptor query-to-batch comparison per row.
void CompareInto(
    const DescriptorSet& query,
    const DescriptorBatch& library,
    const Metric& metric,
    DescriptorComparisonMode mode,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compare one sparse binary query fingerprint against each row in a sparse batch.
std::vector<double> Compare(
    const OEFPSparse& query,
    const OEFPSparseBatch& library,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with one sparse binary query-to-batch comparison per row.
void CompareInto(
    const OEFPSparse& query,
    const OEFPSparseBatch& library,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compare one counted query fingerprint against each row in a counted batch.
std::vector<double> Compare(
    const OEFPCount& query,
    const OEFPCountBatch& library,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with one counted query-to-batch comparison per row.
void CompareInto(
    const OEFPCount& query,
    const OEFPCountBatch& library,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compare one query fingerprint against each row in a dense batch.
std::vector<double> Compare(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with one query-to-batch comparison value per batch row.
///
/// \throws std::invalid_argument: When output_length is not library.Size(), or
///     when output is null for non-empty output.
void CompareInto(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compute row-major cross-distance/comparison values for two batches.
std::vector<double> CDist(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with row-major cross-distance/comparison values.
void CDistInto(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compute row-major cross-distance/comparison values for counted batches.
std::vector<double> CDist(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with row-major counted cross-distance/comparison values.
void CDistInto(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compute row-major cross-distance/comparison values for sparse binary batches.
std::vector<double> CDist(
    const OEFPSparseBatch& a,
    const OEFPSparseBatch& b,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with row-major sparse binary cross-distance/comparison values.
void CDistInto(
    const OEFPSparseBatch& a,
    const OEFPSparseBatch& b,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compute row-major cross-distance/comparison values for descriptor batches.
std::vector<double> CDist(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    DescriptorComparisonMode mode = DescriptorComparisonMode::CountOverlap,
    const BatchKernelOptions& options = {});

/// \brief Fill output with row-major descriptor cross-distance/comparison values.
void CDistInto(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    DescriptorComparisonMode mode,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compute SciPy-compatible condensed pairwise values for one batch.
std::vector<double> PDist(
    const OEFPBatch& batch,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with SciPy-compatible condensed pairwise values.
void PDistInto(
    const OEFPBatch& batch,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compute SciPy-compatible condensed pairwise values for one counted batch.
std::vector<double> PDist(
    const OEFPCountBatch& batch,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with SciPy-compatible counted pairwise values.
void PDistInto(
    const OEFPCountBatch& batch,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compute SciPy-compatible condensed pairwise values for one sparse binary batch.
std::vector<double> PDist(
    const OEFPSparseBatch& batch,
    const Metric& metric,
    const BatchKernelOptions& options = {});

/// \brief Fill output with SciPy-compatible sparse binary pairwise values.
void PDistInto(
    const OEFPSparseBatch& batch,
    const Metric& metric,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Compute SciPy-compatible condensed pairwise values for one descriptor batch.
std::vector<double> PDist(
    const DescriptorBatch& batch,
    const Metric& metric,
    DescriptorComparisonMode mode = DescriptorComparisonMode::CountOverlap,
    const BatchKernelOptions& options = {});

/// \brief Fill output with SciPy-compatible descriptor pairwise values.
void PDistInto(
    const DescriptorBatch& batch,
    const Metric& metric,
    DescriptorComparisonMode mode,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \cond OEFP_BINDING_DETAIL
/// \brief Address-based query-to-batch output helper for Python bindings.
void CompareIntoAddress(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based counted query-to-batch output helper for Python bindings.
void CompareIntoAddress(
    const OEFPCount& query,
    const OEFPCountBatch& library,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based sparse binary query-to-batch output helper for Python bindings.
void CompareIntoAddress(
    const OEFPSparse& query,
    const OEFPSparseBatch& library,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based descriptor query-to-batch output helper for Python bindings.
void CompareIntoAddress(
    const DescriptorSet& query,
    const DescriptorBatch& library,
    const Metric& metric,
    DescriptorComparisonMode mode,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based cdist output helper for Python bindings.
void CDistIntoAddress(
    const OEFPBatch& a,
    const OEFPBatch& b,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based counted cdist output helper for Python bindings.
void CDistIntoAddress(
    const OEFPCountBatch& a,
    const OEFPCountBatch& b,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based sparse binary cdist output helper for Python bindings.
void CDistIntoAddress(
    const OEFPSparseBatch& a,
    const OEFPSparseBatch& b,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based descriptor cdist output helper for Python bindings.
void CDistIntoAddress(
    const DescriptorBatch& a,
    const DescriptorBatch& b,
    const Metric& metric,
    DescriptorComparisonMode mode,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based pdist output helper for Python bindings.
void PDistIntoAddress(
    const OEFPBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based counted pdist output helper for Python bindings.
void PDistIntoAddress(
    const OEFPCountBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based sparse binary pdist output helper for Python bindings.
void PDistIntoAddress(
    const OEFPSparseBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

/// \brief Address-based descriptor pdist output helper for Python bindings.
void PDistIntoAddress(
    const DescriptorBatch& batch,
    const Metric& metric,
    DescriptorComparisonMode mode,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});
/// \endcond

} // namespace OEFP

#endif // OEFP_COMPARE_H
