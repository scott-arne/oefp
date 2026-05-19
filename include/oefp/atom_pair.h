#ifndef OEFP_ATOM_PAIR_H
#define OEFP_ATOM_PAIR_H

#include "oefp/count.h"
#include "oefp/descriptor.h"
#include "oefp/fingerprint.h"
#include "oefp/sparse.h"

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

/// \brief Benchmark-only native stage timing summary for Atom Pair generation.
struct AtomPairGenerationProfile {
    double molecule_preparation_seconds = 0.0;
    double graph_seconds = 0.0;
    double atom_code_seconds = 0.0;
    double distance_seconds = 0.0;
    double pair_enumeration_seconds = 0.0;
    double bit_folding_seconds = 0.0;
    std::uint32_t atom_count = 0;
    std::uint32_t event_count = 0;
    std::uint32_t on_bit_count = 0;

    /// \brief Return the sum of measured native stage times.
    double TotalSeconds() const;
};

/// \brief Reusable generator for RDKit-compatible dense Atom Pair fingerprints.
class AtomPairGenerator {
public:
    /// \brief Construct a reusable generator from validated Atom Pair options.
    ///
    /// \param options Atom Pair generation options.
    /// \throws std::invalid_argument: When the requested options are unsupported
    ///     or invalid.
#ifdef SWIG
    explicit AtomPairGenerator(AtomPairOptions options);
#else
    explicit AtomPairGenerator(AtomPairOptions options = AtomPairOptions{});
#endif

    /// \brief Generate a folded dense binary Atom Pair fingerprint.
    ///
    /// Count-simulation options are supported because they still produce dense
    /// binary output. Count and sparse output remain on the existing
    /// free-function APIs.
    ///
    /// \param mol Molecule to fingerprint.
    /// \returns Dense binary Atom Pair fingerprint.
    OEFP Fingerprint(const OEChem::OEMolBase& mol) const;

    /// \brief Return the normalized generator options.
    const AtomPairOptions& Options() const;

private:
    AtomPairOptions options_;
    FingerprintSpec binary_spec_;
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

/// \brief Generate an RDKit-compatible sparse binary Atom Pair fingerprint.
///
/// Sparse Atom Pair fingerprints use RDKit's fixed sparse result size and keep
/// count simulation enabled by default, matching
/// ``GetAtomPairGenerator(...).GetSparseFingerprint()``.
///
/// \param mol Molecule to fingerprint.
/// \param options Atom Pair generation options.
/// \returns Sparse binary Atom Pair fingerprint.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPSparse MakeAtomPairSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options);
#else
OEFPSparse MakeAtomPairSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options = AtomPairOptions{});
#endif

/// \brief Generate an RDKit-compatible sparse count Atom Pair fingerprint.
///
/// Sparse count Atom Pair fingerprints use RDKit's fixed raw Atom Pair
/// identifier domain. Count simulation is a binary-output concern and is
/// ignored for this counted output, matching
/// ``GetAtomPairGenerator(...).GetSparseCountFingerprint()``.
///
/// \param mol Molecule to fingerprint.
/// \param options Atom Pair generation options.
/// \returns Sparse counted Atom Pair fingerprint.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPCount MakeAtomPairSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options);
#else
OEFPCount MakeAtomPairSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options = AtomPairOptions{});
#endif

/// \brief Generate raw Atom Pair descriptors as a schema-backed counted-key row.
///
/// The descriptor keys are OEFP-owned, unfurled Atom Pair feature identifiers
/// of the form ``smaller_atom_code_distance_larger_atom_code``. They reuse the
/// same atom-code and graph-distance model as the Atom Pair fingerprint
/// generators but do not fold into a fixed-size fingerprint domain. The
/// descriptor row contains one ``atom_pair`` column with counted string keys.
///
/// \param mol Molecule to describe.
/// \param options Atom Pair generation options. ``num_bits`` and count
///     simulation options do not affect descriptor output.
/// \returns Schema-backed counted string-key Atom Pair descriptors.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
DescriptorSet MakeAtomPairDescriptors(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options);
#else
DescriptorSet MakeAtomPairDescriptors(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options = AtomPairOptions{});
#endif

/// \brief Profile native folded binary Atom Pair generation stages.
///
/// This diagnostic helper is intended for benchmark tooling. It follows the
/// same compatibility logic as ``MakeAtomPairFingerprint`` and reports elapsed
/// native stage times without changing public fingerprint behavior.
#ifdef SWIG
AtomPairGenerationProfile ProfileAtomPairFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options);
#else
AtomPairGenerationProfile ProfileAtomPairFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options = AtomPairOptions{});
#endif

} // namespace OEFP

#endif // OEFP_ATOM_PAIR_H
