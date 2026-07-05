#ifndef OEFP_COLUMN_REQUEST_H
#define OEFP_COLUMN_REQUEST_H

#include <cstddef>
#include <unordered_set>
#include <vector>

namespace OEFP {

/// Describes which columns of a source's own schema to compute.
///
/// A ColumnRequest is constructed via static factory methods:
/// - `All()` requests all columns (default state).
/// - `Subset(indices)` requests only the specified columns.
///
/// The `Wants` method determines whether a given column should be computed.
class ColumnRequest {
public:
    /// Returns a request that wants all columns.
    static ColumnRequest All();

    /// Returns a request that wants only the specified source column indices.
    ///
    /// :param source_column_indices: The column indices to request.
    /// :returns: A ColumnRequest for the specified subset.
    static ColumnRequest Subset(std::vector<std::size_t> source_column_indices);

    /// Returns true if all columns are requested.
    ///
    /// :returns: True if this request wants all columns, false otherwise.
    bool WantsAll() const;

    /// Returns true if the specified column is requested.
    ///
    /// A column is wanted if either all columns are requested, or the column
    /// index is in the explicit subset.
    ///
    /// :param source_column_index: The column index to check.
    /// :returns: True if the column should be computed, false otherwise.
    bool Wants(std::size_t source_column_index) const;

private:
    bool all_ = true;
    std::unordered_set<std::size_t> indices_;
};

}  // namespace OEFP

#endif  // OEFP_COLUMN_REQUEST_H
