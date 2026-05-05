#ifndef OEFP_COMPARE_H
#define OEFP_COMPARE_H

#include "oefp/batch.h"
#include "oefp/fingerprint.h"
#include "oefp/metric.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Execution options reserved for dense batch comparison kernels.
struct BatchKernelOptions {
    std::size_t num_threads = 0;
    std::size_t chunk_size = 256;
};

/// \brief Compare two dense binary fingerprints with the requested metric.
///
/// Fingerprint specifications must compare exactly equal, including provenance
/// metadata. This intentionally matches batch admission rules for dense-binary
/// milestone 1.
///
/// \param a First fingerprint.
/// \param b Second fingerprint.
/// \param metric Metric configuration.
/// \returns Similarity or distance according to metric.Mode().
/// \throws std::invalid_argument: When fingerprint specifications or storage
///     widths differ.
double Compare(const OEFP& a, const OEFP& b, const Metric& metric);

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

/// \brief Address-based query-to-batch output helper for Python bindings.
void CompareIntoAddress(
    const OEFP& query,
    const OEFPBatch& library,
    const Metric& metric,
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

/// \brief Address-based pdist output helper for Python bindings.
void PDistIntoAddress(
    const OEFPBatch& batch,
    const Metric& metric,
    std::uint64_t output_address,
    std::size_t output_length,
    const BatchKernelOptions& options = {});

} // namespace OEFP

#endif // OEFP_COMPARE_H
