#ifndef OEFP_ANNOTATION_H
#define OEFP_ANNOTATION_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace OEFP {

/// \brief Atom-centered fingerprint environment provenance.
class OEFPBitEnvironment {
public:
    OEFPBitEnvironment() = default;
    OEFPBitEnvironment(std::uint32_t atom_id, std::uint32_t radius);

    /// \brief Return the center atom id that generated the bit.
    std::uint32_t AtomId() const;

    /// \brief Return the Morgan environment radius.
    std::uint32_t Radius() const;

private:
    std::uint32_t atom_id_ = 0;
    std::uint32_t radius_ = 0;
};

/// \brief Labels and descriptions keyed by fingerprint bit or feature id.
class OEFPAnnotationSet {
public:
    /// \brief Store a human-readable label for one bit id.
    void SetBitLabel(std::uint64_t bit_id, std::string label);

    /// \brief Return the label for bit_id, or an empty string when absent.
    std::string BitLabel(std::uint64_t bit_id) const;

private:
    std::map<std::uint64_t, std::string> bit_labels_;
};

/// \brief Sparse provenance mappings from batch row and bit id to atom ids.
class OEFPMappingSet {
public:
    /// \brief Store atom provenance for one row and bit id.
    void AddAtomMapping(
        std::size_t row,
        std::uint64_t bit_id,
        std::vector<std::uint32_t> atom_ids);

    /// \brief Append one atom-centered environment provenance record.
    void AddEnvironmentMapping(
        std::size_t row,
        std::uint64_t bit_id,
        std::uint32_t atom_id,
        std::uint32_t radius);

    /// \brief Return atom provenance for one row and bit id, or an empty vector.
    std::vector<std::uint32_t> AtomsForBit(std::size_t row, std::uint64_t bit_id) const;

    /// \brief Return environment provenance for one row and bit id.
    std::vector<OEFPBitEnvironment> EnvironmentsForBit(
        std::size_t row,
        std::uint64_t bit_id) const;

    /// \brief Return mapped center atom ids for one row and bit id.
    std::vector<std::uint32_t> EnvironmentAtomIdsForBit(
        std::size_t row,
        std::uint64_t bit_id) const;

    /// \brief Return mapped Morgan radii for one row and bit id.
    std::vector<std::uint32_t> EnvironmentRadiiForBit(
        std::size_t row,
        std::uint64_t bit_id) const;

    /// \brief Return mapped bit ids for one row.
    std::vector<std::uint64_t> BitIds(std::size_t row) const;

    /// \brief Return mapped bit ids as uint32 values for Morgan-compatible mappings.
    std::vector<std::uint32_t> BitIds32(std::size_t row) const;

private:
    std::map<std::pair<std::size_t, std::uint64_t>, std::vector<std::uint32_t>> atom_mappings_;
    std::map<std::pair<std::size_t, std::uint64_t>, std::vector<OEFPBitEnvironment>> environment_mappings_;
};

bool operator==(const OEFPBitEnvironment& lhs, const OEFPBitEnvironment& rhs);
bool operator!=(const OEFPBitEnvironment& lhs, const OEFPBitEnvironment& rhs);

} // namespace OEFP

#endif // OEFP_ANNOTATION_H
