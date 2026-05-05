#ifndef OEFP_ANNOTATION_H
#define OEFP_ANNOTATION_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace OEFP {

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

    /// \brief Return atom provenance for one row and bit id, or an empty vector.
    std::vector<std::uint32_t> AtomsForBit(std::size_t row, std::uint64_t bit_id) const;

private:
    std::map<std::pair<std::size_t, std::uint64_t>, std::vector<std::uint32_t>> atom_mappings_;
};

} // namespace OEFP

#endif // OEFP_ANNOTATION_H
