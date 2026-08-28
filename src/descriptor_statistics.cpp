#include "oefp/descriptor_statistics.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "linear_algebra.h"

namespace OEFP {
namespace {

constexpr double NOT_A_NUMBER = std::numeric_limits<double>::quiet_NaN();

// A null buffer is tolerated only for a zero-row matrix: an empty batch reaches here through
// std::vector::data(), which is allowed to return null for an empty vector.
void validate_matrix(const double* values, std::size_t rows, std::size_t columns) {
    if (values == nullptr && rows != 0u) {
        throw std::invalid_argument("Statistics require a non-null value buffer.");
    }
    if (columns == 0u) {
        throw std::invalid_argument("Statistics require at least one column.");
    }
    // CovarianceMatrix forms columns * columns for its own allocation, so it reaches this
    // product before pseudo_inverse_symmetric gets a chance to guard it; see the comment on
    // the matching check in src/linear_algebra.cpp for why the multiplication cannot be left
    // to defeat the size check it feeds.
    if (columns > std::numeric_limits<std::size_t>::max() / columns) {
        throw std::invalid_argument("Statistics column count is too large to index.");
    }
}

bool row_is_complete(const std::uint8_t* validity, std::size_t row, std::size_t columns) {
    if (validity == nullptr) {
        return true;
    }
    const auto* mask = validity + (row * columns);
    for (std::size_t column = 0u; column < columns; ++column) {
        if (mask[column] == 0u) {
            return false;
        }
    }
    return true;
}

const double* values_from_address(std::uint64_t address) {
    if (address == 0u) {
        throw std::invalid_argument("The value buffer address must not be zero.");
    }
    return reinterpret_cast<const double*>(static_cast<std::uintptr_t>(address));
}

const std::uint8_t* validity_from_address(std::uint64_t address) {
    return address == 0u ? nullptr
                         : reinterpret_cast<const std::uint8_t*>(
                               static_cast<std::uintptr_t>(address));
}

}  // namespace

DescriptorColumnStatistics ColumnStatistics(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns) {
    validate_matrix(values, rows, columns);

    DescriptorColumnStatistics statistics;
    statistics.mean.assign(columns, NOT_A_NUMBER);
    statistics.variance.assign(columns, NOT_A_NUMBER);
    statistics.minimum.assign(columns, NOT_A_NUMBER);
    statistics.maximum.assign(columns, NOT_A_NUMBER);
    statistics.present_count.assign(columns, 0u);

    // Welford's online update keeps the variance stable for large, offset-heavy columns.
    std::vector<double> sum_of_squares(columns, 0.0);
    for (std::size_t row = 0u; row < rows; ++row) {
        const auto* row_values = values + (row * columns);
        const auto* row_mask = validity == nullptr ? nullptr : validity + (row * columns);
        for (std::size_t column = 0u; column < columns; ++column) {
            if (row_mask != nullptr && row_mask[column] == 0u) {
                continue;
            }

            const auto value = row_values[column];
            const auto count = static_cast<double>(++statistics.present_count[column]);
            if (statistics.present_count[column] == 1u) {
                statistics.mean[column] = value;
                statistics.minimum[column] = value;
                statistics.maximum[column] = value;
                continue;
            }

            const auto delta = value - statistics.mean[column];
            statistics.mean[column] += delta / count;
            sum_of_squares[column] += delta * (value - statistics.mean[column]);
            // std::min and std::max return their first argument when the comparison is false, so
            // a NaN value arriving after a finite one would be dropped while the same NaN
            // arriving first would stick -- the extrema would depend on row order. A present
            // value can legitimately be NaN (RDKit emits NaN BCUT2D columns for elements without
            // Gasteiger parameters), so propagate it the way mean and variance already do.
            if (std::isnan(value)) {
                statistics.minimum[column] = value;
                statistics.maximum[column] = value;
            } else {
                statistics.minimum[column] = std::min(statistics.minimum[column], value);
                statistics.maximum[column] = std::max(statistics.maximum[column], value);
            }
        }
    }

    for (std::size_t column = 0u; column < columns; ++column) {
        if (statistics.present_count[column] >= 2u) {
            statistics.variance[column] =
                sum_of_squares[column] / static_cast<double>(statistics.present_count[column] - 1u);
        }
    }

    return statistics;
}

DescriptorCovariance CovarianceMatrix(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns) {
    validate_matrix(values, rows, columns);

    std::vector<std::size_t> complete;
    complete.reserve(rows);
    for (std::size_t row = 0u; row < rows; ++row) {
        if (row_is_complete(validity, row, columns)) {
            complete.push_back(row);
        }
    }

    if (complete.size() < 2u) {
        throw std::invalid_argument(
            "Covariance requires at least two rows with every selected column present.");
    }

    // Centre the mean pass on the first complete row instead of summing raw values. A raw sum
    // overflows on input that never does: two rows of DBL_MAX in one column sum to +inf, the
    // mean is +inf, every centred value is -inf, and a column whose true covariance is 0.0
    // comes back as +inf for InverseCovarianceMatrix to reject. Shifting keeps every partial
    // sum on the scale of the column's spread rather than its offset, so a constant column
    // accumulates exact zeros whatever its magnitude. The subtraction is exact when the
    // reference is close to the value, which is the case that motivates it.
    const auto* reference = values + (complete.front() * columns);
    std::vector<double> offset_sum(columns, 0.0);
    for (const auto row : complete) {
        const auto* row_values = values + (row * columns);
        for (std::size_t column = 0u; column < columns; ++column) {
            offset_sum[column] += row_values[column] - reference[column];
        }
    }
    const auto count = static_cast<double>(complete.size());
    std::vector<double> mean(columns, 0.0);
    for (std::size_t column = 0u; column < columns; ++column) {
        mean[column] = reference[column] + (offset_sum[column] / count);
    }

    DescriptorCovariance covariance;
    covariance.row_count = static_cast<std::uint64_t>(complete.size());
    covariance.matrix.assign(columns * columns, 0.0);

    std::vector<double> centered(columns, 0.0);
    for (const auto row : complete) {
        const auto* row_values = values + (row * columns);
        for (std::size_t column = 0u; column < columns; ++column) {
            centered[column] = row_values[column] - mean[column];
        }
        for (std::size_t i = 0u; i < columns; ++i) {
            for (std::size_t j = i; j < columns; ++j) {
                covariance.matrix[(i * columns) + j] += centered[i] * centered[j];
            }
        }
    }

    const auto denominator = count - 1.0;
    for (std::size_t i = 0u; i < columns; ++i) {
        for (std::size_t j = i; j < columns; ++j) {
            const auto value = covariance.matrix[(i * columns) + j] / denominator;
            covariance.matrix[(i * columns) + j] = value;
            covariance.matrix[(j * columns) + i] = value;  // Mirror; keeps it exactly symmetric.
        }
    }

    return covariance;
}

DescriptorInverseCovariance InverseCovarianceMatrix(
    const double* values,
    const std::uint8_t* validity,
    std::size_t rows,
    std::size_t columns,
    double rcond) {
    const auto covariance = CovarianceMatrix(values, validity, rows, columns);
    auto pseudo_inverse = detail::pseudo_inverse_symmetric(covariance.matrix, columns, rcond);

    DescriptorInverseCovariance inverse;
    inverse.matrix = std::move(pseudo_inverse.matrix);
    inverse.rank = pseudo_inverse.rank;
    inverse.row_count = covariance.row_count;
    return inverse;
}

DescriptorColumnStatistics ColumnStatistics(
    const DescriptorBatch& batch,
    const DescriptorSelection& columns) {
    const auto matrix = batch.ToNumericMatrix(columns);
    auto statistics = ColumnStatistics(matrix.values.data(), matrix.validity.data(), matrix.rows,
                                       matrix.columns);
    statistics.names = matrix.names;
    return statistics;
}

DescriptorCovariance CovarianceMatrix(
    const DescriptorBatch& batch,
    const DescriptorSelection& columns) {
    const auto matrix = batch.ToNumericMatrix(columns);
    return CovarianceMatrix(matrix.values.data(), matrix.validity.data(), matrix.rows,
                            matrix.columns);
}

DescriptorInverseCovariance InverseCovarianceMatrix(
    const DescriptorBatch& batch,
    const DescriptorSelection& columns,
    double rcond) {
    const auto matrix = batch.ToNumericMatrix(columns);
    return InverseCovarianceMatrix(matrix.values.data(), matrix.validity.data(), matrix.rows,
                                   matrix.columns, rcond);
}

DescriptorColumnStatistics ColumnStatisticsAddress(
    std::uint64_t values_address,
    std::uint64_t validity_address,
    std::size_t rows,
    std::size_t columns) {
    return ColumnStatistics(values_from_address(values_address),
                            validity_from_address(validity_address), rows, columns);
}

DescriptorCovariance CovarianceMatrixAddress(
    std::uint64_t values_address,
    std::uint64_t validity_address,
    std::size_t rows,
    std::size_t columns) {
    return CovarianceMatrix(values_from_address(values_address),
                            validity_from_address(validity_address), rows, columns);
}

DescriptorInverseCovariance InverseCovarianceMatrixAddress(
    std::uint64_t values_address,
    std::uint64_t validity_address,
    std::size_t rows,
    std::size_t columns,
    double rcond) {
    return InverseCovarianceMatrix(values_from_address(values_address),
                                   validity_from_address(validity_address), rows, columns,
                                   rcond);
}

}  // namespace OEFP
