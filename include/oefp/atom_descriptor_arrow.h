// Arrow serialization for kallisto atom and bond descriptor batches.
//
// Long-table format: one row per atom (or bond) across all molecules.
// Empty segments (skipped molecules) are preserved via molecule_count metadata.

#ifndef OEFP_ATOM_DESCRIPTOR_ARROW_H
#define OEFP_ATOM_DESCRIPTOR_ARROW_H

#include "oefp/atom_descriptor.h"

#include <memory>
#include <string>

namespace arrow {
class RecordBatch;
}

namespace OEFP {

/// \brief Convert an atom descriptor batch to a long-table Arrow record batch.
///
/// Schema: molecule_id (uint32), atom_index (uint32), then one Float64 column
/// per descriptor. molecule_id encodes the 0-based segment index each atom
/// belongs to. Metadata stores oefp.kallisto.molecule_count to preserve empty
/// segments on round-trip.
///
/// \param batch Atom descriptor batch to convert.
/// \returns Arrow record batch containing one row per atom across all segments.
std::shared_ptr<arrow::RecordBatch> AtomDescriptorBatchToArrow(
    const AtomDescriptorBatch& batch);

/// \brief Convert an Arrow record batch to an atom descriptor batch.
///
/// Reconstructs molecule segments from molecule_id groupings.
/// oefp.kallisto.molecule_count metadata (required) determines the number of
/// segments; segments with no rows become AtomDescriptorSet::Empty.
///
/// \param rb Arrow record batch produced by AtomDescriptorBatchToArrow.
/// \returns Atom descriptor batch with Size() == molecule_count.
/// \throws std::invalid_argument: When required metadata or column types are invalid.
AtomDescriptorBatch AtomDescriptorBatchFromArrow(
    const std::shared_ptr<arrow::RecordBatch>& rb);

/// \brief Convert a bond descriptor batch to a long-table Arrow record batch.
///
/// Schema: molecule_id (uint32), begin (uint32), end (uint32), then one Float64
/// column per descriptor. Metadata stores oefp.kallisto.molecule_count to
/// preserve empty segments.
///
/// \param batch Bond descriptor batch to convert.
/// \returns Arrow record batch containing one row per bond across all segments.
std::shared_ptr<arrow::RecordBatch> BondDescriptorBatchToArrow(
    const BondDescriptorBatch& batch);

/// \brief Convert an Arrow record batch to a bond descriptor batch.
///
/// Reconstructs bond segments from molecule_id groupings.
///
/// \param rb Arrow record batch produced by BondDescriptorBatchToArrow.
/// \returns Bond descriptor batch with Size() == molecule_count.
/// \throws std::invalid_argument: When required metadata or column types are invalid.
BondDescriptorBatch BondDescriptorBatchFromArrow(
    const std::shared_ptr<arrow::RecordBatch>& rb);

/// \brief Write an atom descriptor batch to an Arrow IPC file.
///
/// \param batch Atom descriptor batch to write.
/// \param path Destination file path.
/// \throws std::runtime_error: When Arrow cannot write the file.
void WriteKallistoAtomIpc(const AtomDescriptorBatch& batch, const std::string& path);

/// \brief Read an atom descriptor batch from an Arrow IPC file.
///
/// \param path Source file path.
/// \returns Atom descriptor batch reconstructed from Arrow IPC data.
/// \throws std::invalid_argument: When required metadata or column types are invalid.
/// \throws std::runtime_error: When Arrow cannot read the file.
AtomDescriptorBatch ReadKallistoAtomIpc(const std::string& path);

/// \brief Write a bond descriptor batch to an Arrow IPC file.
///
/// \param batch Bond descriptor batch to write.
/// \param path Destination file path.
/// \throws std::runtime_error: When Arrow cannot write the file.
void WriteKallistoBondIpc(const BondDescriptorBatch& batch, const std::string& path);

/// \brief Read a bond descriptor batch from an Arrow IPC file.
///
/// \param path Source file path.
/// \returns Bond descriptor batch reconstructed from Arrow IPC data.
/// \throws std::invalid_argument: When required metadata or column types are invalid.
/// \throws std::runtime_error: When Arrow cannot read the file.
BondDescriptorBatch ReadKallistoBondIpc(const std::string& path);

} // namespace OEFP

#endif // OEFP_ATOM_DESCRIPTOR_ARROW_H
