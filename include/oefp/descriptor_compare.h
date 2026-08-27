#ifndef OEFP_DESCRIPTOR_COMPARE_H
#define OEFP_DESCRIPTOR_COMPARE_H

#include "oefp/batch_kernel_options.h"
#include "oefp/metric.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace OEFP {

/// \brief Policy for descriptor dimensions missing from either compared row.
enum class DescriptorMissingPolicy {
    Propagate,  ///< Any missing value in either row yields NaN. Default.
    Ignore,     ///< Skip that dimension for the pair, then renormalize the accumulator.
};

/// \brief Condensed pairwise distances over a dense numeric descriptor matrix.
///
/// \param values Row-major \p rows x \p columns buffer of descriptor values.
/// \param validity Row-major \p rows x \p columns presence mask, or \c nullptr when every
///        value is present.
/// \param rows Row count.
/// \param columns Column count.
/// \param metric Comparison metric. Euclidean, Manhattan, Chebyshev, Hamming, Canberra, and
///        BrayCurtis are supported.
/// \param missing Missing-value policy.
/// \param kernel Threading options.
/// \return \c rows * (rows - 1) / 2 distances in condensed upper-triangular order.
/// \throws std::invalid_argument When the metric is not valid for numeric comparison.
std::vector<double> PDistNumeric(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing = DescriptorMissingPolicy::Propagate,
    const BatchKernelOptions& kernel = {});

/// \brief Condensed pairwise distances written into a caller-supplied buffer.
///
/// \param output Destination buffer, caller-owned.
/// \param output_length Destination length; must equal \c rows * (rows - 1) / 2.
/// \throws std::invalid_argument When \p output_length is wrong or the metric is invalid.
void PDistNumericInto(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel = {});

/// \brief Rectangular distances between two dense numeric descriptor matrices.
///
/// \return \c a_rows * \c b_rows distances in row-major order.
/// \throws std::invalid_argument When the metric is not valid for numeric comparison.
std::vector<double> CDistNumeric(
    const double* a_values,
    const std::uint8_t* a_validity,
    std::size_t a_rows,
    const double* b_values,
    const std::uint8_t* b_validity,
    std::size_t b_rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing = DescriptorMissingPolicy::Propagate,
    const BatchKernelOptions& kernel = {});

/// \brief Rectangular distances written into a caller-supplied buffer.
///
/// \throws std::invalid_argument When \p output_length is wrong or the metric is invalid.
void CDistNumericInto(
    const double* a_values,
    const std::uint8_t* a_validity,
    std::size_t a_rows,
    const double* b_values,
    const std::uint8_t* b_validity,
    std::size_t b_rows,
    std::size_t columns,
    const Metric& metric,
    DescriptorMissingPolicy missing,
    double* output,
    std::size_t output_length,
    const BatchKernelOptions& kernel = {});

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_COMPARE_H
