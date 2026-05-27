#ifndef OEFP_TOPOLOGICAL_TORSIONS_H
#define OEFP_TOPOLOGICAL_TORSIONS_H

#include "oefp/count.h"
#include "oefp/descriptor.h"
#include "oefp/fingerprint.h"
#include "oefp/sparse.h"

#include <cstdint>
#include <vector>
#include <oechem.h>

namespace OEFP {

/// \brief Public options for RDKit-compatible Topological Torsions fingerprints.
struct TopologicalTorsionsOptions {
    std::uint32_t torsion_atom_count = 4;
    std::uint32_t num_bits = 2048;
    bool use_chirality = false;
    bool count_simulation = true;
    std::vector<std::uint32_t> count_bounds{1u, 2u, 4u, 8u};
};

/// \brief Prerequisites for graph-only Topological Torsions descriptors.
inline constexpr DescriptorPrerequisites kTopologicalTorsionsPrerequisites =
    kDescriptorPrerequisiteGraph;

/// \brief Reusable generator for folded dense binary Topological Torsions fingerprints.
class TopologicalTorsionsGenerator {
public:
    /// \brief Construct a reusable generator from validated options.
    ///
    /// \param options Topological Torsions generation options.
    /// \throws std::invalid_argument: When requested options are unsupported
    ///     or invalid.
#ifdef SWIG
    explicit TopologicalTorsionsGenerator(TopologicalTorsionsOptions options);
#else
    explicit TopologicalTorsionsGenerator(
        TopologicalTorsionsOptions options = TopologicalTorsionsOptions{});
#endif

    /// \brief Generate a folded dense binary Topological Torsions fingerprint.
    ///
    /// \param mol Molecule to fingerprint.
    /// \returns Dense binary Topological Torsions fingerprint.
    OEFP Fingerprint(const OEChem::OEMolBase& mol) const;

    /// \brief Return the normalized generator options.
    const TopologicalTorsionsOptions& Options() const;

private:
    TopologicalTorsionsOptions options_;
    FingerprintSpec binary_spec_;
};

/// \brief Generate an RDKit-compatible folded binary Topological Torsions fingerprint.
///
/// \param mol Molecule to fingerprint.
/// \param options Topological Torsions generation options.
/// \returns Dense binary Topological Torsions fingerprint.
/// \throws std::invalid_argument: When requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFP MakeTopologicalTorsionsFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options);
#else
OEFP MakeTopologicalTorsionsFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options = TopologicalTorsionsOptions{});
#endif

/// \brief Generate an RDKit-compatible folded count Topological Torsions fingerprint.
///
/// Counts are accumulated from the same torsion path events used by the binary
/// generator, then folded by ``raw_hash % num_bits``. Count simulation is a
/// binary-output concern and is normalized away for counted fingerprints.
///
/// \param mol Molecule to fingerprint.
/// \param options Topological Torsions generation options.
/// \returns Sparse counted Topological Torsions fingerprint.
/// \throws std::invalid_argument: When requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPCount MakeTopologicalTorsionsCountFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options);
#else
OEFPCount MakeTopologicalTorsionsCountFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options = TopologicalTorsionsOptions{});
#endif

/// \brief Generate an RDKit-compatible sparse binary Topological Torsions fingerprint.
///
/// Sparse Topological Torsions fingerprints use RDKit's sparse result-size
/// clipping and keep count simulation enabled by default, matching
/// ``GetTopologicalTorsionGenerator(...).GetSparseFingerprint()``.
///
/// \param mol Molecule to fingerprint.
/// \param options Topological Torsions generation options.
/// \returns Sparse binary Topological Torsions fingerprint.
/// \throws std::invalid_argument: When requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPSparse MakeTopologicalTorsionsSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options);
#else
OEFPSparse MakeTopologicalTorsionsSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options = TopologicalTorsionsOptions{});
#endif

/// \brief Generate an RDKit-compatible raw sparse-count Topological Torsions fingerprint.
///
/// Sparse-count Topological Torsions fingerprints use RDKit's unpacked
/// 64-bit path-code identifiers. They are intentionally separate from folded
/// ``OEFPCount`` fingerprints because default four-atom torsion identifiers can
/// exceed the uint32_t range.
///
/// \param mol Molecule to fingerprint.
/// \param options Topological Torsions generation options. ``num_bits`` and
///     count simulation options do not affect sparse-count output.
/// \returns Sparse counted Topological Torsions fingerprint with uint64_t
///     feature identifiers.
/// \throws std::invalid_argument: When requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPCount64 MakeTopologicalTorsionsSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options);
#else
OEFPCount64 MakeTopologicalTorsionsSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options = TopologicalTorsionsOptions{});
#endif

/// \brief Generate raw Topological Torsions descriptors as counted path-code keys.
///
/// Descriptor keys are OEFP-owned canonical path-code strings such as
/// ``32_32_32_32``. They reuse the same graph-only atom-code and path model as
/// the fingerprint generators but do not fold into a fixed-size fingerprint
/// domain.
///
/// \param mol Molecule to describe.
/// \param options Topological Torsions generation options. ``num_bits`` and
///     count simulation options do not affect descriptor output.
/// \returns Schema-backed counted string-key Topological Torsions descriptors.
/// \throws std::invalid_argument: When requested options are unsupported
///     or invalid.
#ifdef SWIG
DescriptorSet MakeTopologicalTorsionsDescriptors(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options);
#else
DescriptorSet MakeTopologicalTorsionsDescriptors(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options = TopologicalTorsionsOptions{});
#endif

} // namespace OEFP

#endif // OEFP_TOPOLOGICAL_TORSIONS_H
