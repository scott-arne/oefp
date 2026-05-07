#ifndef OEFP_MORGAN_H
#define OEFP_MORGAN_H

#include "oefp/annotation.h"
#include "oefp/count.h"
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
    explicit MorganGenerator(MorganOptions options = MorganOptions{});

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

} // namespace OEFP

#endif // OEFP_MORGAN_H
