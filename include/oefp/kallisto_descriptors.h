// Kallisto atom descriptors — coordination numbers, proximity shells, and charge-related properties.
//
// Ported algorithms from kallisto (Apache 2.0):
//   AstraZeneca; Caldeweyher, Meli, Pracht
//   https://github.com/AstraZeneca/kallisto

#ifndef OEFP_KALLISTO_DESCRIPTORS_H
#define OEFP_KALLISTO_DESCRIPTORS_H

#include "oefp/kallisto_geometry.h"
#include "oefp/descriptor_schema.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace OEFP {

/// Compute coordination numbers for all atoms.
///
/// :param ctx: Geometry context for the molecule.
/// :param cntype: Functional type ("erf", "cov", or "exp").
/// :returns: Per-atom coordination numbers.
///
/// For each ordered pair (i, j) with i ≠ j, computes a damped contribution
/// based on the interatomic distance in Bohr, covalent radii, and functional
/// type. Pairs beyond threshold = 800.0 Bohr² (squared distance) are skipped.
///
/// Functional types:
///   - "exp": Standard damping with k1 = 16.0
///   - "erf": Error function damping with kn = 7.50
///   - "cov": Covalent damping with electronegativity weighting
///            k4 = 4.10451, k5 = 19.08857, k6 = 2*11.28174²
///
/// Mirrors kallisto.methods.getCoordinationNumbers.
std::vector<double> coordination_numbers(
    const KallistoGeometryContext& ctx,
    const std::string& cntype
);

/// Compute proximity shells for all atoms.
///
/// :param ctx: Geometry context for the molecule.
/// :param size: Shell scale pair (default {2, 3}).
/// :returns: Per-atom proximity shell difference (shell at size.second - shell at size.first).
///
/// For each atom, computes a shell sum at two different covalent radius scales,
/// then returns the difference (larger shell - smaller shell). Uses the same
/// electronegativity-weighted damping as covalent coordination numbers.
/// Threshold is forced to 800.0 Bohr² (squared distance cutoff).
///
/// Mirrors kallisto.methods.getProximityShells.
std::vector<double> proximity_shells(
    const KallistoGeometryContext& ctx,
    std::pair<int, int> size = {2, 3}
);

/// Compute EEQ (electronegativity equilibration) atomic partial charges.
///
/// :param ctx: Geometry context for the molecule.
/// :returns: Per-atom EEQ charges (in elementary charge units).
///
/// Builds an (N+1)×(N+1) Lagrange-augmented linear system encoding the
/// electronegativity equilibration condition, then solves for atomic charges
/// constrained to sum to the total molecular charge.
///
/// Diagonal: A[i][i] = eeq_gamm[Zi-1] + sqrt(2/π) / sqrt(alpha_i),
///           where alpha_i = (eeq_alp[Zi-1])²
/// Off-diagonal: A[i][j] = erf(r_ij / sqrt(alpha_i + alpha_j)) / r_ij
/// RHS: X[i] = -eeq_en[Zi-1] + eeq_cnfak[Zi-1] * sqrt(covalent_cn[i])
/// Lagrange row/column all 1, corner 0; X[N] = total charge.
///
/// Returns empty vector if ctx is ineligible or the system is singular.
///
/// Mirrors kallisto.methods.getAtomicPartialCharges.
std::vector<double> eeq_charges(const KallistoGeometryContext& ctx);

/// Compute atomic-charge dependent dynamic polarizabilities (D4 method).
///
/// :param ctx: Geometry context for the molecule.
/// :returns: Per-atom static polarizabilities in Bohr³.
///
/// Computes charge-scaled atomic polarizabilities using the DFT-D4 damping
/// functions and reference polarizabilities. For each atom, builds weighted
/// averages over reference systems using Gaussian weighting of covalent CN,
/// then applies charge scaling via zeta functions on EEQ charges.
///
/// Returns the static (ω=0) polarizability aw[0] for each atom.
///
/// Returns empty vector if ctx is ineligible, EEQ charges are unavailable,
/// or any result value is non-finite.
///
/// Mirrors kallisto.methods.getPolarizabilities.
std::vector<double> polarizabilities(const KallistoGeometryContext& ctx);

/// Compute van der Waals radii using atomic polarizabilities.
///
/// :param ctx: Geometry context for the molecule.
/// :param vdwtype: van der Waals radius type ("rahm" or "truhlar").
/// :returns: Per-atom van der Waals radii in Bohr.
///
/// Computes van der Waals radii from polarizabilities using the formula:
/// vdw[i] = scale * theta_b * 2.54 * aw[i]^(1/7), where theta_b is the
/// element-specific parameter from VDW_RAHM or VDW_TRUHLAR, aw is the
/// atomic polarizability, and scale=1.0 for descriptors (Bohr units).
///
/// Returns empty vector if ctx is ineligible, polarizabilities are unavailable,
/// vdwtype is invalid, or any result value is non-finite.
///
/// Mirrors kallisto.methods.getVanDerWaalsRadii.
std::vector<double> van_der_waals_radii(
    const KallistoGeometryContext& ctx,
    const std::string& vdwtype
);

/// Return the descriptor schema for kallisto atom descriptors.
///
/// :returns: Shared singleton schema instance.
///
/// Defines eight columns: cn_erf, cn_cov, cn_exp, prox, eeq, alp, vdw_rahm, vdw_truhlar.
/// All columns have value_kind Float, group "kallisto", source_name "kallisto",
/// source_type "geometric", source_version "kallisto-1.0.10", and
/// prerequisites kDescriptorPrerequisiteCoordinates3D.
std::shared_ptr<const DescriptorSchema> KallistoAtomDescriptorSchema();

} // namespace OEFP

// Forward declarations for OEFP namespace types needed by the free functions below
namespace OEFP {
class AtomDescriptorSet;
class AtomDescriptorBatch;
}

// Free functions in OEFP namespace for Python binding and internal use
namespace OEFP {

/// Compute kallisto atom descriptors for one molecule.
///
/// :param mol: Molecule to compute descriptors for.
/// :param charge: Optional override for molecular charge (defaults to formal charge).
/// :returns: AtomDescriptorSet with schema KallistoAtomDescriptorSchema(),
///           empty if molecule is ineligible.
///
/// Eligibility: GetDimension()==3, GetCoords succeeds for every atom, and every
/// atom Z is 1..86. Ineligible molecules return AtomDescriptorSet::Empty(...).
AtomDescriptorSet MakeKallistoAtomDescriptors(
    const OEChem::OEMolBase& mol,
    std::optional<int> charge = std::nullopt
);

/// Descriptor source for kallisto atom descriptors.
///
/// Computes cn_erf, cn_cov, cn_exp, and prox for 3D molecules with Z <= 86.
class KallistoAtomDescriptorSource {
public:
    /// Construct a kallisto atom descriptor source.
    ///
    /// :param charge: Optional override for molecular charge.
    explicit KallistoAtomDescriptorSource(std::optional<int> charge = std::nullopt);

    /// Return the descriptor schema.
    std::shared_ptr<const DescriptorSchema> Schema() const;

    /// Compute descriptors for one molecule.
    ///
    /// :param mol: Molecule to compute descriptors for.
    /// :returns: AtomDescriptorSet with schema KallistoAtomDescriptorSchema(),
    ///           empty if molecule is ineligible.
    AtomDescriptorSet Compute(const OEChem::OEMolBase& mol) const;

private:
    std::optional<int> charge_;
};

/// Compute kallisto atom descriptors as a single-segment batch (minimal Python binding helper).
///
/// :param mol: Molecule to compute descriptors for.
/// :param charge: Optional override for molecular charge.
/// :returns: AtomDescriptorBatch with one segment (may be empty if ineligible).
///
/// This helper exists solely to provide Python with zero-copy access to kallisto atom
/// descriptors for a single molecule via the batch's column data/validity address APIs.
/// Task 10 will add threaded batching.
AtomDescriptorBatch MakeKallistoAtomDescriptorBatch(
    const OEChem::OEMolBase& mol,
    std::optional<int> charge = std::nullopt
);

} // namespace OEFP

#endif // OEFP_KALLISTO_DESCRIPTORS_H
