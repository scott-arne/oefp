#include "oefp/kallisto_geometry.h"
#include "oefp/kallisto_data.h"
#include "oefp/kallisto_descriptors.h"

#include <oechem.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace OEFP {

std::optional<std::vector<double>> solve_dense_linear(
    std::vector<double> a,
    std::vector<double> b,
    std::size_t n
) {
    // Validate dimensions
    if (a.size() != n * n) {
        throw std::invalid_argument("Matrix a size must be n*n");
    }
    if (b.size() != n) {
        throw std::invalid_argument("Vector b size must be n");
    }

    if (n == 0) {
        return std::vector<double>{};
    }

    // Gaussian elimination with partial pivoting (in-place LU decomposition)
    // a is row-major: a[i*n + j] is row i, column j
    constexpr double epsilon = 1e-12;

    // Copy b for solving
    std::vector<double> x = b;

    // Forward elimination
    for (std::size_t col = 0; col < n; ++col) {
        // Find pivot: max |a[row][col]| for row >= col
        std::size_t pivot_row = col;
        double max_val = std::abs(a[col * n + col]);
        for (std::size_t row = col + 1; row < n; ++row) {
            double val = std::abs(a[row * n + col]);
            if (val > max_val) {
                max_val = val;
                pivot_row = row;
            }
        }

        // Check for singularity
        if (max_val < epsilon) {
            return std::nullopt;
        }

        // Swap rows if needed
        if (pivot_row != col) {
            for (std::size_t j = 0; j < n; ++j) {
                std::swap(a[col * n + j], a[pivot_row * n + j]);
            }
            std::swap(x[col], x[pivot_row]);
        }

        // Eliminate column entries below pivot
        for (std::size_t row = col + 1; row < n; ++row) {
            double factor = a[row * n + col] / a[col * n + col];
            for (std::size_t j = col; j < n; ++j) {
                a[row * n + j] -= factor * a[col * n + j];
            }
            x[row] -= factor * x[col];
        }
    }

    // Back substitution
    for (std::size_t i = n; i > 0; --i) {
        std::size_t row = i - 1;
        double sum = 0.0;
        for (std::size_t j = row + 1; j < n; ++j) {
            sum += a[row * n + j] * x[j];
        }
        x[row] = (x[row] - sum) / a[row * n + row];
    }

    return x;
}

KallistoGeometryContext::KallistoGeometryContext(
    const OEChem::OEMolBase& mol,
    std::optional<int> charge
)
    : eligible_(false)
    , atom_count_(0)
    , charge_(0)
{
    atom_count_ = mol.NumAtoms();

    // Resolve charge first: use override if provided, otherwise sum formal charges
    if (charge.has_value()) {
        charge_ = charge.value();
    } else {
        int summed_formal_charge = 0;
        for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
            summed_formal_charge += atom->GetFormalCharge();
        }
        charge_ = summed_formal_charge;
    }

    // Empty molecule is ineligible
    if (atom_count_ == 0) {
        return;
    }

    // Must be 3D
    if (mol.GetDimension() != 3) {
        return;
    }

    // Extract atomic numbers, coordinates, and atom indices
    atomic_numbers_.reserve(atom_count_);
    coords_bohr_.reserve(atom_count_);
    atom_indices_.reserve(atom_count_);

    double coords_angstrom[3];

    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        // Get atomic number (1-based Z)
        int z = atom->GetAtomicNum();

        // Check if Z is in supported range [1, 86]
        if (z < 1 || z > 86) {
            // Unsupported element: mark ineligible and return
            atomic_numbers_.clear();
            coords_bohr_.clear();
            atom_indices_.clear();
            return;
        }

        atomic_numbers_.push_back(z);

        // Get coordinates
        if (!mol.GetCoords(atom, coords_angstrom)) {
            // Failed to get coordinates: mark ineligible and return
            atomic_numbers_.clear();
            coords_bohr_.clear();
            atom_indices_.clear();
            return;
        }

        // Convert Angstrom to Bohr
        std::array<double, 3> coords_bohr_atom = {
            coords_angstrom[0] / kallisto::BOHR_RADIUS_ANGSTROM,
            coords_angstrom[1] / kallisto::BOHR_RADIUS_ANGSTROM,
            coords_angstrom[2] / kallisto::BOHR_RADIUS_ANGSTROM
        };
        coords_bohr_.push_back(coords_bohr_atom);

        // Capture OpenEye atom index
        atom_indices_.push_back(atom->GetIdx());
    }

    // All checks passed: molecule is eligible
    eligible_ = true;
}

bool KallistoGeometryContext::Eligible() const {
    return eligible_;
}

std::size_t KallistoGeometryContext::AtomCount() const {
    return atom_count_;
}

const std::vector<int>& KallistoGeometryContext::AtomicNumbers() const {
    return atomic_numbers_;
}

const std::vector<std::array<double, 3>>& KallistoGeometryContext::CoordsBohr() const {
    return coords_bohr_;
}

int KallistoGeometryContext::Charge() const {
    return charge_;
}

const std::vector<double>& KallistoGeometryContext::CovalentCoordinationNumbers() const {
    if (!covalent_cn_cache_.has_value()) {
        covalent_cn_cache_ = coordination_numbers(*this, "cov");
    }
    return covalent_cn_cache_.value();
}

const std::vector<double>& KallistoGeometryContext::EeqCharges() const {
    if (!eeq_charges_cache_.has_value()) {
        eeq_charges_cache_ = eeq_charges(*this);
    }
    return eeq_charges_cache_.value();
}

const std::vector<std::uint32_t>& KallistoGeometryContext::AtomIndices() const {
    return atom_indices_;
}

} // namespace OEFP
