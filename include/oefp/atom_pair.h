#ifndef OEFP_ATOM_PAIR_H
#define OEFP_ATOM_PAIR_H

#include "oefp/count.h"
#include "oefp/fingerprint.h"

#include <cstdint>
#include <vector>
#include <oechem.h>

namespace OEFP {

/// \brief Public options for RDKit-compatible Atom Pair fingerprints.
struct AtomPairOptions {
    std::uint32_t min_distance = 1;
    std::uint32_t max_distance = 30;
    std::uint32_t num_bits = 2048;
    bool use_chirality = false;
    bool use_2d = true;
    bool count_simulation = true;
    std::vector<std::uint32_t> count_bounds{1u, 2u, 4u, 8u};
};

/// \brief Generate an RDKit-compatible folded binary Atom Pair fingerprint.
///
/// The production implementation is OEFP-owned; RDKit is used only by
/// conformance tests.
///
/// \param mol Molecule to fingerprint.
/// \param options Atom Pair generation options.
/// \returns Dense binary Atom Pair fingerprint.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFP MakeAtomPairFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options);
#else
OEFP MakeAtomPairFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options = AtomPairOptions{});
#endif

/// \brief Generate an RDKit-compatible folded count Atom Pair fingerprint.
///
/// Counts are accumulated from the same atom-pair events used by the binary
/// generator, then folded by ``raw_id % num_bits``. Count simulation is a
/// binary-output concern and is normalized away for counted fingerprints.
///
/// \param mol Molecule to fingerprint.
/// \param options Atom Pair generation options.
/// \returns Sparse counted Atom Pair fingerprint.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPCount MakeAtomPairCountFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options);
#else
OEFPCount MakeAtomPairCountFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options = AtomPairOptions{});
#endif

} // namespace OEFP

#endif // OEFP_ATOM_PAIR_H
