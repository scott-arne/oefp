#ifndef OEFP_MORGAN_H
#define OEFP_MORGAN_H

#include "oefp/count.h"
#include "oefp/fingerprint.h"

#include <cstdint>
#include <vector>
#include <oechem.h>

namespace OEFP {

/// \brief Public options for RDKit-compatible Morgan fingerprints.
struct MorganOptions {
    std::uint32_t radius = 2;
    std::uint32_t num_bits = 2048;
    bool use_chirality = false;
    bool use_bond_types = true;
    bool only_nonzero_invariants = false;
    bool include_ring_membership = true;
    bool include_redundant_environments = false;
    bool count_simulation = false;
    std::vector<std::uint32_t> count_bounds{1u, 2u, 4u, 8u};
};

/// \brief Generate an RDKit-compatible folded binary Morgan fingerprint.
///
/// The production implementation is OEFP-owned; RDKit is used only by
/// conformance tests.
///
/// \param mol Molecule to fingerprint.
/// \param options Morgan generation options.
/// \returns Dense binary Morgan fingerprint.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFP MakeMorganFingerprint(const OEChem::OEMolBase& mol, const MorganOptions& options);
#else
OEFP MakeMorganFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options = MorganOptions{});
#endif

/// \brief Generate an RDKit-compatible folded count Morgan fingerprint.
///
/// Counts are accumulated from the same atom-environment events used by the
/// binary generator, then folded by ``raw_id % num_bits``.
/// Count simulation is only defined for binary Morgan output and is rejected
/// for this API.
///
/// \param mol Molecule to fingerprint.
/// \param options Morgan generation options.
/// \returns Sparse counted Morgan fingerprint.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPCount MakeMorganCountFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
OEFPCount MakeMorganCountFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options = MorganOptions{});
#endif

/// \brief Generate an RDKit-compatible sparse count Morgan fingerprint.
///
/// Counts are accumulated from raw atom-environment identifiers without
/// folding by ``num_bits``. This matches RDKit's sparse count Morgan output.
///
/// \param mol Molecule to fingerprint.
/// \param options Morgan generation options.
/// \returns Sparse counted Morgan fingerprint keyed by raw environment id.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPCount MakeMorganSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
OEFPCount MakeMorganSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options = MorganOptions{});
#endif

} // namespace OEFP

#endif // OEFP_MORGAN_H
