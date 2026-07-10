// Kallisto geometry context and linear algebra utilities.
//
// Ported algorithms from kallisto (Apache 2.0):
//   AstraZeneca; Caldeweyher, Meli, Pracht
//   https://github.com/AstraZeneca/kallisto

#ifndef OEFP_KALLISTO_GEOMETRY_H
#define OEFP_KALLISTO_GEOMETRY_H

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace OEChem {
class OEMolBase;
}

namespace OEFP {

/// Solve A x = b for a dense (row-major) NxN system via partial-pivot LU.
///
/// :param a: Row-major NxN coefficient matrix (size n*n).
/// :param b: Right-hand side vector (size n).
/// :param n: System dimension.
/// :returns: Solution vector x, or nullopt if the system is singular.
/// :raises std::invalid_argument: If a.size() != n*n or b.size() != n.
///
/// Uses Gaussian elimination with partial pivoting. A pivot is considered
/// singular if its absolute value is below 1e-12.
std::optional<std::vector<double>> solve_dense_linear(
    std::vector<double> a,
    std::vector<double> b,
    std::size_t n
);

/// Per-molecule geometry context for kallisto atom/bond descriptors.
///
/// Extracts and caches 3D coordinates (in Bohr), atomic numbers, and resolved
/// charge. Eligible molecules are non-empty, 3D (GetDimension()==3), have
/// valid coordinates for all atoms, and contain only elements Z in [1,86].
///
/// Caches covalent coordination numbers and EEQ charges on first access (computed
/// by kernels in Tasks 5 and 6).
///
/// Thread-safe for read-only access from multiple threads (one context per
/// molecule per thread). Mutable caches are protected by internal synchronization.
class KallistoGeometryContext {
public:
    /// Construct a geometry context from an OpenEye molecule.
    ///
    /// :param mol: OpenEye molecule (coordinates in Angstrom).
    /// :param charge: Optional molecular charge override. If not provided,
    ///                charge is computed as the sum of formal charges.
    ///
    /// If the molecule is ineligible (2D, missing coordinates, unsupported
    /// elements), the context is still constructed but Eligible() returns false.
    explicit KallistoGeometryContext(
        const OEChem::OEMolBase& mol,
        std::optional<int> charge = std::nullopt
    );

    /// Check if the molecule is eligible for kallisto descriptors.
    ///
    /// :returns: True if non-empty, 3D, all coordinates valid, all Z in [1,86].
    bool Eligible() const;

    /// Number of atoms.
    std::size_t AtomCount() const;

    /// Atomic numbers (1-based Z) for each atom.
    const std::vector<int>& AtomicNumbers() const;

    /// 3D coordinates in Bohr for each atom.
    ///
    /// Coordinates are converted from Angstrom by dividing by BOHR_RADIUS_ANGSTROM.
    const std::vector<std::array<double, 3>>& CoordsBohr() const;

    /// Resolved molecular charge.
    ///
    /// :returns: Charge override if provided, else sum of formal charges.
    int Charge() const;

    /// Covalent coordination numbers (cached, computed on first access).
    ///
    /// :returns: Vector of covalent CN for each atom (cntype "cov").
    /// :raises std::logic_error: Stub in Task 4; kernel lands in Task 5.
    const std::vector<double>& CovalentCoordinationNumbers() const;

    /// EEQ atomic charges (cached, computed on first access).
    ///
    /// :returns: Vector of EEQ charges for each atom.
    /// :raises std::logic_error: Stub in Task 4; kernel lands in Task 6.
    const std::vector<double>& EeqCharges() const;

private:
    bool eligible_;
    std::size_t atom_count_;
    std::vector<int> atomic_numbers_;
    std::vector<std::array<double, 3>> coords_bohr_;
    int charge_;

    // Lazy caches for computed properties
    mutable std::optional<std::vector<double>> covalent_cn_cache_;
    mutable std::optional<std::vector<double>> eeq_charges_cache_;
};

} // namespace OEFP

#endif // OEFP_KALLISTO_GEOMETRY_H
