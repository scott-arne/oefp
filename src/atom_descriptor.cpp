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

bool schemas_match(
    const DescriptorSchema& lhs,
    const DescriptorSchema& rhs) {
    if (lhs.Size() != rhs.Size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.Size(); ++i) {
        if (lhs.Definition(i).name != rhs.Definition(i).name) {
            return false;
        }
    }
    return true;
}

void validate_schema_match(
    const std::shared_ptr<const DescriptorSchema>& batch_schema,
    const DescriptorSchema& set_schema) {
    if (!batch_schema) {
        throw std::invalid_argument("Descriptor schema must not be null.");
    }
    if (!schemas_match(*batch_schema, set_schema)) {
        throw std::invalid_argument(
            "Appended set schema does not match batch schema.");
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
    if (!schema_) {
        throw std::invalid_argument("Descriptor schema must not be null.");
    }
    validate_column_count(columns_.size(), schema_->Size());
    const auto expected_rows = atom_indices_.size();
    for (std::size_t col = 0; col < columns_.size(); ++col) {
        if (columns_[col].size() != expected_rows) {
            throw std::invalid_argument(
                "Column length does not match atom count.");
        }
    }
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
    if (!schema_) {
        throw std::invalid_argument("Descriptor schema must not be null.");
    }
    validate_column_count(columns_.size(), schema_->Size());
    const auto expected_rows = bond_endpoints_.size();
    for (std::size_t col = 0; col < columns_.size(); ++col) {
        if (columns_[col].size() != expected_rows) {
            throw std::invalid_argument(
                "Column length does not match bond count.");
        }
    }
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
    validate_schema_match(schema_, set.Schema());

    const auto atom_count = set.AtomCount();
    const auto& indices = set.AtomIndices();

    // Stage all appended data in local temporaries first
    std::vector<std::uint32_t> staged_indices;
    staged_indices.reserve(atom_count);
    staged_indices.insert(staged_indices.end(), indices.begin(), indices.end());

    std::vector<std::vector<double>> staged_values(schema_->Size());
    std::vector<std::vector<std::uint8_t>> staged_validity(schema_->Size());

    for (std::size_t col = 0; col < schema_->Size(); ++col) {
        staged_values[col].reserve(atom_count);
        staged_validity[col].reserve(atom_count);

        for (std::size_t atom = 0; atom < atom_count; ++atom) {
            const auto value = set.Value(atom, col);
            if (value.has_value()) {
                staged_values[col].push_back(*value);
                staged_validity[col].push_back(1u);
            } else {
                staged_values[col].push_back(0.0);  // placeholder
                staged_validity[col].push_back(0u);
            }
        }
    }

    // Commit staged data to member vectors
    atom_indices_.insert(atom_indices_.end(), staged_indices.begin(), staged_indices.end());

    for (std::size_t col = 0; col < schema_->Size(); ++col) {
        column_values_[col].insert(
            column_values_[col].end(),
            staged_values[col].begin(),
            staged_values[col].end());
        column_validity_[col].insert(
            column_validity_[col].end(),
            staged_validity[col].begin(),
            staged_validity[col].end());
    }

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
    validate_schema_match(schema_, set.Schema());

    const auto bond_count = set.BondCount();
    const auto& endpoints = set.BondEndpoints();

    // Stage all appended data in local temporaries first
    std::vector<std::uint32_t> staged_begin;
    std::vector<std::uint32_t> staged_end;
    staged_begin.reserve(bond_count);
    staged_end.reserve(bond_count);

    for (const auto& [begin, end] : endpoints) {
        staged_begin.push_back(begin);
        staged_end.push_back(end);
    }

    std::vector<std::vector<double>> staged_values(schema_->Size());
    std::vector<std::vector<std::uint8_t>> staged_validity(schema_->Size());

    for (std::size_t col = 0; col < schema_->Size(); ++col) {
        staged_values[col].reserve(bond_count);
        staged_validity[col].reserve(bond_count);

        for (std::size_t bond = 0; bond < bond_count; ++bond) {
            const auto value = set.Value(bond, col);
            if (value.has_value()) {
                staged_values[col].push_back(*value);
                staged_validity[col].push_back(1u);
            } else {
                staged_values[col].push_back(0.0);  // placeholder
                staged_validity[col].push_back(0u);
            }
        }
    }

    // Commit staged data to member vectors
    bond_begin_.insert(bond_begin_.end(), staged_begin.begin(), staged_begin.end());
    bond_end_.insert(bond_end_.end(), staged_end.begin(), staged_end.end());

    for (std::size_t col = 0; col < schema_->Size(); ++col) {
        column_values_[col].insert(
            column_values_[col].end(),
            staged_values[col].begin(),
            staged_values[col].end());
        column_validity_[col].insert(
            column_validity_[col].end(),
            staged_validity[col].begin(),
            staged_validity[col].end());
    }

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
