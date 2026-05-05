#ifndef OEFP_COMPARE_H
#define OEFP_COMPARE_H

#include "oefp/batch.h"
#include "oefp/fingerprint.h"
#include "oefp/metric.h"

#include <cstddef>

namespace OEFP {

/// \brief Execution options reserved for dense batch comparison kernels.
struct BatchKernelOptions {
    std::size_t num_threads = 0;
    std::size_t chunk_size = 256;
};

/// \brief Compare two dense binary fingerprints with the requested metric.
///
/// \param a First fingerprint.
/// \param b Second fingerprint.
/// \param metric Metric configuration.
/// \returns Similarity or distance according to metric.Mode().
/// \raises std::invalid_argument: When fingerprint specifications differ.
double Compare(const OEFP& a, const OEFP& b, const Metric& metric);

} // namespace OEFP

#endif // OEFP_COMPARE_H
