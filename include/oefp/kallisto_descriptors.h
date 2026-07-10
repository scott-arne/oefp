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

/// Return the descriptor schema for kallisto atom descriptors.
///
/// :returns: Shared singleton schema instance.
///
/// For Task 5, defines four columns: cn_erf, cn_cov, cn_exp, prox.
/// Later tasks (6-9) will append additional columns (eeq, alp, vdw_rahm, vdw_truhlar).
/// All columns have value_kind Float, group "kallisto", source_name "kallisto",
/// source_type "geometric", source_version "kallisto-1.0.10", and
/// prerequisites kDescriptorPrerequisiteCoordinates3D.
std::shared_ptr<const DescriptorSchema> KallistoAtomDescriptorSchema();

} // namespace OEFP

#endif // OEFP_KALLISTO_DESCRIPTORS_H
