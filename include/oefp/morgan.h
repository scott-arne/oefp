#ifndef OEFP_MORGAN_H
#define OEFP_MORGAN_H

#include "oefp/annotation.h"
#include "oefp/count.h"
#include "oefp/descriptor.h"
#include "oefp/fingerprint.h"
#include "oefp/sparse.h"

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

/// \cond OEFP_BINDING_DETAIL
/// \brief Benchmark-only native stage timing summary for Morgan generation.
struct MorganGenerationProfile {
    double graph_seconds = 0.0;
    double invariant_seconds = 0.0;
    double radius_zero_seconds = 0.0;
    double neighborhood_seconds = 0.0;
    double duplicate_seconds = 0.0;
    double bit_folding_seconds = 0.0;
    std::uint32_t atom_count = 0;
    std::uint32_t bond_count = 0;
    std::uint32_t event_count = 0;
    std::uint32_t on_bit_count = 0;

    /// \brief Return the sum of measured native stage times.
    double TotalSeconds() const;
};
/// \endcond

/// \brief Dense binary Morgan fingerprint plus separate environment mappings.
class MorganFingerprintResult {
public:
    MorganFingerprintResult() = default;
    MorganFingerprintResult(OEFP fingerprint, OEFPMappingSet mapping);

    /// \brief Return the generated fingerprint by value.
    OEFP Fingerprint() const;

    /// \brief Return the generated bit environment mappings by value.
    OEFPMappingSet Mapping() const;

private:
    OEFP fingerprint_;
    OEFPMappingSet mapping_;
};

/// \brief Sparse binary Morgan fingerprint plus separate environment mappings.
class MorganSparseFingerprintResult {
public:
    MorganSparseFingerprintResult() = default;
    MorganSparseFingerprintResult(OEFPSparse fingerprint, OEFPMappingSet mapping);

    /// \brief Return the generated sparse fingerprint by value.
    OEFPSparse Fingerprint() const;

    /// \brief Return the generated bit environment mappings by value.
    OEFPMappingSet Mapping() const;

private:
    OEFPSparse fingerprint_;
    OEFPMappingSet mapping_;
};

/// \brief Folded count Morgan fingerprint plus separate environment mappings.
class MorganCountFingerprintResult {
public:
    MorganCountFingerprintResult() = default;
    MorganCountFingerprintResult(OEFPCount fingerprint, OEFPMappingSet mapping);

    /// \brief Return the generated counted fingerprint by value.
    OEFPCount Fingerprint() const;

    /// \brief Return the generated bit environment mappings by value.
    OEFPMappingSet Mapping() const;

private:
    OEFPCount fingerprint_;
    OEFPMappingSet mapping_;
};

/// \brief Sparse count Morgan fingerprint plus separate environment mappings.
class MorganSparseCountFingerprintResult {
public:
    MorganSparseCountFingerprintResult() = default;
    MorganSparseCountFingerprintResult(OEFPCount fingerprint, OEFPMappingSet mapping);

    /// \brief Return the generated sparse counted fingerprint by value.
    OEFPCount Fingerprint() const;

    /// \brief Return the generated bit environment mappings by value.
    OEFPMappingSet Mapping() const;

private:
    OEFPCount fingerprint_;
    OEFPMappingSet mapping_;
};

/// \brief Reusable generator for RDKit-compatible dense Morgan fingerprints.
class MorganGenerator {
public:
    /// \brief Construct a reusable generator from validated Morgan options.
    ///
    /// \param options Morgan generation options.
    /// \throws std::invalid_argument: When the requested options are unsupported
    ///     or invalid.
#ifdef SWIG
    explicit MorganGenerator(MorganOptions options);
#else
    explicit MorganGenerator(MorganOptions options = MorganOptions{});
#endif

    /// \brief Generate a folded dense binary Morgan fingerprint.
    ///
    /// Count-simulation options are supported because they still produce dense
    /// binary output. Mapping, count, and sparse output remain on the existing
    /// free-function APIs.
    ///
    /// \param mol Molecule to fingerprint.
    /// \returns Dense binary Morgan fingerprint.
    OEFP Fingerprint(const OEChem::OEMolBase& mol) const;

    /// \brief Return the normalized generator options.
    const MorganOptions& Options() const;

private:
    MorganOptions options_;
    FingerprintSpec binary_spec_;
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

/// \brief Generate a folded binary Morgan fingerprint with bit provenance mappings.
///
/// Mapping output records RDKit-style ``bitInfoMap`` entries for normal binary
/// and count-simulated binary Morgan output. Each mapped bit stores the center
/// atom id and Morgan radius for every environment that generated that bit.
///
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
MorganFingerprintResult MakeMorganFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
MorganFingerprintResult MakeMorganFingerprintWithMapping(
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

/// \brief Generate a folded count Morgan fingerprint with bit provenance mappings.
///
/// Mapping keys are folded count ids and mapping values record RDKit-style
/// ``bitInfoMap`` entries for each atom environment contributing to the count.
/// Count simulation is only defined for binary Morgan output and is rejected
/// for this API.
///
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
MorganCountFingerprintResult MakeMorganCountFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
MorganCountFingerprintResult MakeMorganCountFingerprintWithMapping(
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

/// \brief Generate a sparse count Morgan fingerprint with raw bit mappings.
///
/// Mapping keys are raw Morgan environment identifiers, matching RDKit sparse
/// count Morgan bit-info output.
///
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
MorganSparseCountFingerprintResult MakeMorganSparseCountFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
MorganSparseCountFingerprintResult MakeMorganSparseCountFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options = MorganOptions{});
#endif

/// \brief Generate raw Morgan descriptors as typed integer keys with counts.
///
/// Descriptor keys are the same RDKit-compatible raw Morgan environment
/// identifiers used by sparse count output. Count simulation is only defined
/// for binary fingerprints and is rejected for this API.
///
/// \param mol Molecule to fingerprint.
/// \param options Morgan generation options. ``num_bits`` and count simulation
///     options do not affect descriptor output.
/// \returns Counted integer-key Morgan descriptors.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
DescriptorSet MakeMorganDescriptors(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
DescriptorSet MakeMorganDescriptors(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options = MorganOptions{});
#endif

/// \brief Generate an RDKit-compatible sparse binary Morgan fingerprint.
///
/// On-bit identifiers are raw atom-environment identifiers without folding by
/// ``num_bits``. This matches RDKit's sparse binary Morgan output.
///
/// \param mol Molecule to fingerprint.
/// \param options Morgan generation options.
/// \returns Sparse binary Morgan fingerprint keyed by raw environment id.
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
OEFPSparse MakeMorganSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
OEFPSparse MakeMorganSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options = MorganOptions{});
#endif

/// \brief Generate a sparse binary Morgan fingerprint with raw bit mappings.
///
/// Mapping keys are raw Morgan environment identifiers, matching RDKit sparse
/// binary Morgan bit-info output.
///
/// \throws std::invalid_argument: When the requested options are unsupported
///     or invalid.
#ifdef SWIG
MorganSparseFingerprintResult MakeMorganSparseFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
MorganSparseFingerprintResult MakeMorganSparseFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options = MorganOptions{});
#endif

/// \cond OEFP_BINDING_DETAIL
/// \brief Profile native folded binary Morgan generation stages.
///
/// This diagnostic helper is intended for benchmark tooling. It follows the
/// same compatibility logic as ``MakeMorganFingerprint`` and reports elapsed
/// native stage times without changing public fingerprint behavior.
#ifdef SWIG
MorganGenerationProfile ProfileMorganFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options);
#else
MorganGenerationProfile ProfileMorganFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options = MorganOptions{});
#endif
/// \endcond

} // namespace OEFP

#endif // OEFP_MORGAN_H
