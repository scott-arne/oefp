#include "oefp/atom_descriptor.h"

#include <stdexcept>
#include <utility>

namespace OEFP {
namespace {

std::uint64_t pointer_address(const void* data) {
    if (data == nullptr) {
        return 0;
    }
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(data));
}

void validate_column_count(
    std::size_t column_count,
    std::size_t schema_size) {
    if (column_count != schema_size) {
        throw std::invalid_argument(
            "Column count does not match descriptor schema size.");
    }
}

void check_atom_index(std::size_t atom, std::size_t atom_count) {
    if (atom >= atom_count) {
        throw std::out_of_range("Atom index is out of range.");
    }
}

void check_bond_index(std::size_t bond, std::size_t bond_count) {
    if (bond >= bond_count) {
        throw std::out_of_range("Bond index is out of range.");
    }
}

void check_column_index(std::size_t column, std::size_t column_count) {
    if (column >= column_count) {
        throw std::out_of_range("Column index is out of range.");
    }
}

void check_molecule_index(std::size_t molecule, std::size_t molecule_count) {
    if (molecule >= molecule_count) {
        throw std::out_of_range("Molecule index is out of range.");
    }
}

} // namespace

// AtomDescriptorSet implementation

AtomDescriptorSet::AtomDescriptorSet(
    std::shared_ptr<const DescriptorSchema> schema,
    std::vector<std::uint32_t> atom_indices,
    std::vector<std::vector<std::optional<double>>> columns)
    : schema_(std::move(schema)),
      atom_indices_(std::move(atom_indices)),
      columns_(std::move(columns)) {
    validate_column_count(columns_.size(), schema_->Size());
}

AtomDescriptorSet AtomDescriptorSet::Empty(
    std::shared_ptr<const DescriptorSchema> schema) {
    std::vector<std::vector<std::optional<double>>> empty_columns(schema->Size());
    return AtomDescriptorSet(std::move(schema), {}, std::move(empty_columns));
}

const DescriptorSchema& AtomDescriptorSet::Schema() const {
    return *schema_;
}

std::size_t AtomDescriptorSet::AtomCount() const {
    return atom_indices_.size();
}

const std::vector<std::uint32_t>& AtomDescriptorSet::AtomIndices() const {
    return atom_indices_;
}

std::optional<double> AtomDescriptorSet::Value(
    std::size_t atom,
    const std::string& name) const {
    const auto column = schema_->IndexOf(name);
    return Value(atom, column);
}

std::optional<double> AtomDescriptorSet::Value(
    std::size_t atom,
    std::size_t column) const {
    check_atom_index(atom, AtomCount());
    check_column_index(column, schema_->Size());
    return columns_[column][atom];
}

// BondDescriptorSet implementation

BondDescriptorSet::BondDescriptorSet(
    std::shared_ptr<const DescriptorSchema> schema,
    std::vector<std::pair<std::uint32_t, std::uint32_t>> bond_endpoints,
    std::vector<std::vector<std::optional<double>>> columns)
    : schema_(std::move(schema)),
      bond_endpoints_(std::move(bond_endpoints)),
      columns_(std::move(columns)) {
    validate_column_count(columns_.size(), schema_->Size());
}

BondDescriptorSet BondDescriptorSet::Empty(
    std::shared_ptr<const DescriptorSchema> schema) {
    std::vector<std::vector<std::optional<double>>> empty_columns(schema->Size());
    return BondDescriptorSet(std::move(schema), {}, std::move(empty_columns));
}

const DescriptorSchema& BondDescriptorSet::Schema() const {
    return *schema_;
}

std::size_t BondDescriptorSet::BondCount() const {
    return bond_endpoints_.size();
}

const std::vector<std::pair<std::uint32_t, std::uint32_t>>&
BondDescriptorSet::BondEndpoints() const {
    return bond_endpoints_;
}

std::optional<double> BondDescriptorSet::Value(
    std::size_t bond,
    const std::string& name) const {
    const auto column = schema_->IndexOf(name);
    return Value(bond, column);
}

std::optional<double> BondDescriptorSet::Value(
    std::size_t bond,
    std::size_t column) const {
    check_bond_index(bond, BondCount());
    check_column_index(column, schema_->Size());
    return columns_[column][bond];
}

// AtomDescriptorBatch implementation

AtomDescriptorBatch AtomDescriptorBatch::Empty(
    std::shared_ptr<const DescriptorSchema> schema) {
    AtomDescriptorBatch batch;
    batch.schema_ = std::move(schema);
    batch.column_values_.resize(batch.schema_->Size());
    batch.column_validity_.resize(batch.schema_->Size());
    return batch;
}

void AtomDescriptorBatch::Append(const AtomDescriptorSet& set) {
    const auto atom_count = set.AtomCount();
    const auto& indices = set.AtomIndices();

    // Append atom indices
    atom_indices_.insert(atom_indices_.end(), indices.begin(), indices.end());

    // Append column values and validity
    for (std::size_t col = 0; col < schema_->Size(); ++col) {
        for (std::size_t atom = 0; atom < atom_count; ++atom) {
            const auto value = set.Value(atom, col);
            if (value.has_value()) {
                column_values_[col].push_back(*value);
                column_validity_[col].push_back(1u);
            } else {
                column_values_[col].push_back(0.0);  // placeholder
                column_validity_[col].push_back(0u);
            }
        }
    }

    // Append row offset
    const auto next_offset = static_cast<std::uint64_t>(atom_indices_.size());
    row_offsets_.push_back(next_offset);
}

std::size_t AtomDescriptorBatch::Size() const {
    return row_offsets_.empty() ? 0u : row_offsets_.size() - 1u;
}

std::size_t AtomDescriptorBatch::AtomCount() const {
    return atom_indices_.size();
}

std::size_t AtomDescriptorBatch::SegmentAtomCount(std::size_t molecule) const {
    check_molecule_index(molecule, Size());
    return static_cast<std::size_t>(row_offsets_[molecule + 1u] - row_offsets_[molecule]);
}

std::uint64_t AtomDescriptorBatch::RowOffsetDataAddress() const {
    return pointer_address(row_offsets_.empty() ? nullptr : row_offsets_.data());
}

std::uint64_t AtomDescriptorBatch::AtomIndexDataAddress() const {
    return pointer_address(atom_indices_.empty() ? nullptr : atom_indices_.data());
}

std::uint64_t AtomDescriptorBatch::ColumnDataAddress(std::size_t column) const {
    check_column_index(column, schema_->Size());
    return pointer_address(
        column_values_[column].empty() ? nullptr : column_values_[column].data());
}

std::uint64_t AtomDescriptorBatch::ColumnValidityAddress(std::size_t column) const {
    check_column_index(column, schema_->Size());
    return pointer_address(
        column_validity_[column].empty() ? nullptr : column_validity_[column].data());
}

// BondDescriptorBatch implementation

BondDescriptorBatch BondDescriptorBatch::Empty(
    std::shared_ptr<const DescriptorSchema> schema) {
    BondDescriptorBatch batch;
    batch.schema_ = std::move(schema);
    batch.column_values_.resize(batch.schema_->Size());
    batch.column_validity_.resize(batch.schema_->Size());
    return batch;
}

void BondDescriptorBatch::Append(const BondDescriptorSet& set) {
    const auto bond_count = set.BondCount();
    const auto& endpoints = set.BondEndpoints();

    // Append bond endpoints
    for (const auto& [begin, end] : endpoints) {
        bond_begin_.push_back(begin);
        bond_end_.push_back(end);
    }

    // Append column values and validity
    for (std::size_t col = 0; col < schema_->Size(); ++col) {
        for (std::size_t bond = 0; bond < bond_count; ++bond) {
            const auto value = set.Value(bond, col);
            if (value.has_value()) {
                column_values_[col].push_back(*value);
                column_validity_[col].push_back(1u);
            } else {
                column_values_[col].push_back(0.0);  // placeholder
                column_validity_[col].push_back(0u);
            }
        }
    }

    // Append row offset
    const auto next_offset = static_cast<std::uint64_t>(bond_begin_.size());
    row_offsets_.push_back(next_offset);
}

std::size_t BondDescriptorBatch::Size() const {
    return row_offsets_.empty() ? 0u : row_offsets_.size() - 1u;
}

std::size_t BondDescriptorBatch::BondCount() const {
    return bond_begin_.size();
}

std::size_t BondDescriptorBatch::SegmentBondCount(std::size_t molecule) const {
    check_molecule_index(molecule, Size());
    return static_cast<std::size_t>(row_offsets_[molecule + 1u] - row_offsets_[molecule]);
}

std::uint64_t BondDescriptorBatch::RowOffsetDataAddress() const {
    return pointer_address(row_offsets_.empty() ? nullptr : row_offsets_.data());
}

std::uint64_t BondDescriptorBatch::BondBeginDataAddress() const {
    return pointer_address(bond_begin_.empty() ? nullptr : bond_begin_.data());
}

std::uint64_t BondDescriptorBatch::BondEndDataAddress() const {
    return pointer_address(bond_end_.empty() ? nullptr : bond_end_.data());
}

std::uint64_t BondDescriptorBatch::ColumnDataAddress(std::size_t column) const {
    check_column_index(column, schema_->Size());
    return pointer_address(
        column_values_[column].empty() ? nullptr : column_values_[column].data());
}

std::uint64_t BondDescriptorBatch::ColumnValidityAddress(std::size_t column) const {
    check_column_index(column, schema_->Size());
    return pointer_address(
        column_validity_[column].empty() ? nullptr : column_validity_[column].data());
}

} // namespace OEFP
