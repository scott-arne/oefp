#ifndef OEFP_ATOM_DESCRIPTOR_H
#define OEFP_ATOM_DESCRIPTOR_H

#include "oefp/descriptor_schema.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace OEFP {

/// \brief Per-atom descriptor values for one molecule.
///
/// Stores one row per atom with a uint32 atom index. Values may be missing
/// (NaN or skipped). Empty sets represent skipped molecules.
class AtomDescriptorSet {
public:
    /// \brief Construct a descriptor set for atoms.
    ///
    /// \param schema Descriptor schema shared across all sets.
    /// \param atom_indices Atom index for each row.
    /// \param columns Per-column values in [column][atom] order.
    /// \throws std::invalid_argument: When column count mismatches schema.
    AtomDescriptorSet(std::shared_ptr<const DescriptorSchema> schema,
                      std::vector<std::uint32_t> atom_indices,
                      std::vector<std::vector<std::optional<double>>> columns);

    /// \brief Construct an empty set for a skipped molecule.
    ///
    /// \param schema Descriptor schema.
    static AtomDescriptorSet Empty(std::shared_ptr<const DescriptorSchema> schema);

    /// \brief Return the descriptor schema.
    const DescriptorSchema& Schema() const;

    /// \brief Return the number of atoms.
    std::size_t AtomCount() const;

    /// \brief Return atom indices.
    const std::vector<std::uint32_t>& AtomIndices() const;

    /// \brief Return one value by atom index and column name.
    ///
    /// \param atom Atom index (0-based row index, not the atom_indices value).
    /// \param name Column name.
    /// \throws std::out_of_range: When atom or column index is out of range.
    std::optional<double> Value(std::size_t atom, const std::string& name) const;

    /// \brief Return one value by atom index and column index.
    ///
    /// \param atom Atom index (0-based row index).
    /// \param column Column index.
    /// \throws std::out_of_range: When atom or column index is out of range.
    std::optional<double> Value(std::size_t atom, std::size_t column) const;

private:
    std::shared_ptr<const DescriptorSchema> schema_;
    std::vector<std::uint32_t> atom_indices_;
    std::vector<std::vector<std::optional<double>>> columns_;
};

/// \brief Per-bond descriptor values for one molecule.
///
/// Stores one row per bond with a (begin, end) uint32 endpoint pair.
class BondDescriptorSet {
public:
    /// \brief Construct a descriptor set for bonds.
    ///
    /// \param schema Descriptor schema shared across all sets.
    /// \param bond_endpoints Bond endpoint pairs for each row.
    /// \param columns Per-column values in [column][bond] order.
    /// \throws std::invalid_argument: When column count mismatches schema.
    BondDescriptorSet(std::shared_ptr<const DescriptorSchema> schema,
                      std::vector<std::pair<std::uint32_t, std::uint32_t>> bond_endpoints,
                      std::vector<std::vector<std::optional<double>>> columns);

    /// \brief Construct an empty set for a skipped molecule.
    ///
    /// \param schema Descriptor schema.
    static BondDescriptorSet Empty(std::shared_ptr<const DescriptorSchema> schema);

    /// \brief Return the descriptor schema.
    const DescriptorSchema& Schema() const;

    /// \brief Return the number of bonds.
    std::size_t BondCount() const;

    /// \brief Return bond endpoint pairs.
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& BondEndpoints() const;

    /// \brief Return one value by bond index and column name.
    ///
    /// \param bond Bond index (0-based row index).
    /// \param name Column name.
    /// \throws std::out_of_range: When bond or column index is out of range.
    std::optional<double> Value(std::size_t bond, const std::string& name) const;

    /// \brief Return one value by bond index and column index.
    ///
    /// \param bond Bond index (0-based row index).
    /// \param column Column index.
    /// \throws std::out_of_range: When bond or column index is out of range.
    std::optional<double> Value(std::size_t bond, std::size_t column) const;

private:
    std::shared_ptr<const DescriptorSchema> schema_;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bond_endpoints_;
    std::vector<std::vector<std::optional<double>>> columns_;
};

/// \brief Flattened CSR batch of per-atom descriptors across molecules.
///
/// Each molecule contributes one segment (may be empty for skipped molecules).
/// Per-column storage uses flattened double value arrays + a parallel per-entry
/// validity array (1=present, 0=missing).
class AtomDescriptorBatch {
public:
    /// \brief Construct an empty batch.
    ///
    /// \param schema Descriptor schema.
    static AtomDescriptorBatch Empty(std::shared_ptr<const DescriptorSchema> schema);

    /// \brief Append one atom descriptor set.
    ///
    /// Empty sets produce zero-length segments but still increment Size().
    ///
    /// \param set Descriptor set to append.
    void Append(const AtomDescriptorSet& set);

    /// \brief Return the number of molecule segments.
    std::size_t Size() const;

    /// \brief Return the total number of atoms across all segments.
    std::size_t AtomCount() const;

    /// \brief Return the number of atoms in one molecule segment.
    ///
    /// \param molecule Molecule index.
    /// \throws std::out_of_range: When molecule index is out of range.
    std::size_t SegmentAtomCount(std::size_t molecule) const;

    /// \brief Return the row offset data pointer as an integer address.
    std::uint64_t RowOffsetDataAddress() const;

    /// \brief Return the flattened atom index data pointer as an integer address.
    std::uint64_t AtomIndexDataAddress() const;

    /// \brief Return one column's value data pointer as an integer address.
    ///
    /// \param column Column index.
    /// \throws std::out_of_range: When column index is out of range.
    std::uint64_t ColumnDataAddress(std::size_t column) const;

    /// \brief Return one column's validity data pointer as an integer address.
    ///
    /// \param column Column index.
    /// \throws std::out_of_range: When column index is out of range.
    std::uint64_t ColumnValidityAddress(std::size_t column) const;

private:
    std::shared_ptr<const DescriptorSchema> schema_;
    std::vector<std::uint64_t> row_offsets_{0u};
    std::vector<std::uint32_t> atom_indices_;
    std::vector<std::vector<double>> column_values_;
    std::vector<std::vector<std::uint8_t>> column_validity_;
};

/// \brief Flattened CSR batch of per-bond descriptors across molecules.
///
/// Each molecule contributes one segment (may be empty). Per-column storage uses
/// flattened double value arrays + a parallel per-entry validity array.
class BondDescriptorBatch {
public:
    /// \brief Construct an empty batch.
    ///
    /// \param schema Descriptor schema.
    static BondDescriptorBatch Empty(std::shared_ptr<const DescriptorSchema> schema);

    /// \brief Append one bond descriptor set.
    ///
    /// Empty sets produce zero-length segments but still increment Size().
    ///
    /// \param set Descriptor set to append.
    void Append(const BondDescriptorSet& set);

    /// \brief Return the number of molecule segments.
    std::size_t Size() const;

    /// \brief Return the total number of bonds across all segments.
    std::size_t BondCount() const;

    /// \brief Return the number of bonds in one molecule segment.
    ///
    /// \param molecule Molecule index.
    /// \throws std::out_of_range: When molecule index is out of range.
    std::size_t SegmentBondCount(std::size_t molecule) const;

    /// \brief Return the row offset data pointer as an integer address.
    std::uint64_t RowOffsetDataAddress() const;

    /// \brief Return the flattened bond begin index data pointer as an integer address.
    std::uint64_t BondBeginDataAddress() const;

    /// \brief Return the flattened bond end index data pointer as an integer address.
    std::uint64_t BondEndDataAddress() const;

    /// \brief Return one column's value data pointer as an integer address.
    ///
    /// \param column Column index.
    /// \throws std::out_of_range: When column index is out of range.
    std::uint64_t ColumnDataAddress(std::size_t column) const;

    /// \brief Return one column's validity data pointer as an integer address.
    ///
    /// \param column Column index.
    /// \throws std::out_of_range: When column index is out of range.
    std::uint64_t ColumnValidityAddress(std::size_t column) const;

private:
    std::shared_ptr<const DescriptorSchema> schema_;
    std::vector<std::uint64_t> row_offsets_{0u};
    std::vector<std::uint32_t> bond_begin_;
    std::vector<std::uint32_t> bond_end_;
    std::vector<std::vector<double>> column_values_;
    std::vector<std::vector<std::uint8_t>> column_validity_;
};

} // namespace OEFP

#endif // OEFP_ATOM_DESCRIPTOR_H
