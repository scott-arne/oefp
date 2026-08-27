#ifndef OEFP_BATCH_KERNEL_OPTIONS_H
#define OEFP_BATCH_KERNEL_OPTIONS_H

#include <cstddef>

namespace OEFP {

/// \brief Execution options reserved for dense batch comparison kernels.
struct BatchKernelOptions {
    std::size_t num_threads = 0;
    std::size_t chunk_size = 256;
};

} // namespace OEFP

#endif // OEFP_BATCH_KERNEL_OPTIONS_H
