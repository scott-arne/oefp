#ifndef OEFP_DESCRIPTOR_BATCH_H
#define OEFP_DESCRIPTOR_BATCH_H

#include "oefp/descriptor.h"
#include "oefp/descriptor_selection.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace OEFP {

/// \brief Dense row-major numeric view of selected scalar descriptor columns.
///
/// \c values and \c validity are both \c rows * \c columns entries in row-major order.
/// A validity entry of ``1`` marks a present value; the matching \c values entry is
/// meaningless when validity is ``0``.
struct DescriptorNumericMatrix {
    std::vector<double> values;
    std::vector<std::uint8_t> validity;
    std::vector<std::string> names;
    std::size_t rows = 0;
    std::size_t columns = 0;
};

/// \brief Contiguous descriptor batch storage.
///
/// Rows are stored in CSR-like form. ``row_offsets_[row]`` and
/// ``row_offsets_[row + 1]`` bound each descriptor row in the flattened active
/// key vector and count array.
class DescriptorBatch {
public:
    DescriptorBatch() = default;

    /// \brief Construct an empty descriptor batch with an explicit spec.
    explicit DescriptorBatch(DescriptorSpec spec);

    /// \brief Build a descriptor batch from compatible descriptor sets.
    ///
    /// \throws std::invalid_argument: When descriptor sets have mismatched specs.
    static DescriptorBatch FromDescriptorSets(
        const std::vector<::OEFP::DescriptorSet>& descriptors);

    /// \brief Construct an empty schema-backed descriptor batch.
    ///
    /// \param schema Shared scalar descriptor schema.
    /// \returns Empty descriptor batch that preserves the supplied schema.
    /// \throws std::invalid_argument: When the schema is null or contains non-scalar columns.
    static DescriptorBatch Empty(std::shared_ptr<const DescriptorSchema> schema);

    /// \brief Append one descriptor row.
    ///
    /// \throws std::invalid_argument: When the descriptor set is incompatible.
    void Append(const DescriptorSet& descriptors);

    /// \brief Return the shared descriptor specification.
    const DescriptorSpec& Spec() const;

    /// \brief Return the shared descriptor value type.
    DescriptorValueType ValueType() const;

    /// \brief Return the shared descriptor schema for schema-backed batches.
    const DescriptorSchema& Schema() const;

    /// \brief Return the number of descriptor rows.
    std::size_t Size() const;

    /// \brief Return row identifiers for schema-backed batches.
    const std::vector<std::string>& RowIds() const;

    /// \brief Return a copied float descriptor column.
    ///
    /// Missing values are returned as the zero-filled raw storage, not as NaN; consult
    /// \c ColumnValidity to tell a real zero from an absent value. The Python accessor
    /// substitutes NaN instead.
    std::vector<double> FloatColumn(const std::string& name) const;

    /// \brief Return a copied integer descriptor column.
    std::vector<std::int64_t> IntColumn(const std::string& name) const;

    /// \brief Return a copied boolean descriptor column.
    std::vector<std::uint8_t> BoolColumn(const std::string& name) const;

    /// \brief Return a copied string descriptor column.
    std::vector<std::string> StringColumn(const std::string& name) const;

    /// \brief Return copied scalar column validity flags.
    ///
    /// ``1`` marks a present value and ``0`` marks a missing value.
    std::vector<std::uint8_t> ColumnValidity(const std::string& name) const;

    /// \brief Return a column subset in selection order.
    DescriptorBatch Subset(const DescriptorSelection& selection) const;

    /// \brief Materialize selected scalar columns as a dense row-major numeric matrix.
    ///
    /// \c Bool columns map to ``0.0``/``1.0`` and \c Int columns widen to \c double.
    ///
    /// \param selection Columns to materialize, in the order they are resolved.
    /// \return The values, the validity mask, the resolved names, and the extents.
    /// \throws std::invalid_argument When the batch is not schema-backed, a selected
    ///         column is not \c Bool, \c Int, or \c Float, or the matrix extents overflow.
    DescriptorNumericMatrix ToNumericMatrix(const DescriptorSelection& selection) const;

    /// \brief Return the total number of flattened descriptor entries.
    std::size_t EntryCount() const;

    /// \brief Return one row offset.
    ///
    /// ``row`` may equal ``Size()`` to access the sentinel final offset.
    ///
    /// \throws std::out_of_range: When row is greater than Size().
    std::uint64_t RowOffset(std::size_t row) const;

    /// \brief Return the number of entries in one row.
    ///
    /// \throws std::out_of_range: When row is greater than or equal to Size().
    std::size_t RowEntryCount(std::size_t row) const;

    /// \brief Return read-only access to flattened string keys.
    const std::vector<std::string>& StringKeys() const;

    /// \brief Return read-only access to flattened integer keys.
    const std::vector<std::int64_t>& IntegerKeys() const;

    /// \brief Return read-only access to flattened float keys.
    const std::vector<double>& FloatKeys() const;

    /// \brief Return read-only access to flattened descriptor counts.
    const std::vector<std::uint32_t>& Counts() const;

    /// \brief Return read-only access to row offsets.
    const std::vector<std::uint64_t>& RowOffsets() const;

    /// \brief Return read-only access to flattened descriptor counts.
    const std::uint32_t* CountData() const;
    std::uint64_t CountDataAddress() const;

    /// \brief Return read-only access to row offsets.
    const std::uint64_t* RowOffsetData() const;
    std::uint64_t RowOffsetDataAddress() const;

    /// \brief Return read-only access to flattened integer keys.
    const std::int64_t* IntegerKeyData() const;
    std::uint64_t IntegerKeyDataAddress() const;

    /// \brief Return read-only access to flattened float keys.
    const double* FloatKeyData() const;
    std::uint64_t FloatKeyDataAddress() const;

private:
    struct DescriptorColumnBlock {
        DescriptorValueKind value_kind = DescriptorValueKind::Float;
        std::vector<std::uint8_t> validity;
        std::vector<std::int64_t> int_values;
        std::vector<double> float_values;
        std::vector<std::uint8_t> bool_values;
        std::vector<std::string> string_values;
    };

    DescriptorSpec spec_;
    std::vector<std::string> string_keys_;
    std::vector<std::int64_t> integer_keys_;
    std::vector<double> float_keys_;
    std::vector<std::uint32_t> counts_;
    std::vector<std::uint64_t> row_offsets_{0u};
    bool has_spec_ = false;
    std::shared_ptr<const DescriptorSchema> schema_;
    std::vector<std::string> row_ids_;
    std::vector<DescriptorColumnBlock> columns_;

    void ReserveRowsAndEntries(std::size_t row_count, std::size_t entry_count);
    void AppendActiveKeys(const DescriptorSet& descriptors);
    void AppendTypedRow(const DescriptorSet& descriptors);
    void InitializeColumns(std::shared_ptr<const DescriptorSchema> schema);
    void ValidateDescriptorSet(const DescriptorSet& descriptors) const;
    void ValidateTypedDescriptorSet(const DescriptorSet& descriptors) const;
    void CheckRowIndex(std::size_t row) const;
    void CheckOffsetIndex(std::size_t row) const;
    void RequireLegacyStorage() const;
    const DescriptorColumnBlock& Column(const std::string& name, DescriptorValueKind kind) const;

    friend bool operator==(const DescriptorBatch& lhs, const DescriptorBatch& rhs);
};

bool operator==(const DescriptorBatch& lhs, const DescriptorBatch& rhs);
bool operator!=(const DescriptorBatch& lhs, const DescriptorBatch& rhs);

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_BATCH_H
