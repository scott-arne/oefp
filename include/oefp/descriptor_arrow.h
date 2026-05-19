#ifndef OEFP_DESCRIPTOR_ARROW_H
#define OEFP_DESCRIPTOR_ARROW_H

#include "oefp/descriptor_batch.h"

#include <memory>
#include <string>

namespace arrow {
class RecordBatch;
}

namespace OEFP {

/// \brief Convert a schema-backed scalar descriptor batch to an Arrow record batch.
///
/// The Arrow schema stores OEFP descriptor metadata so the descriptor schema can
/// be reconstructed without relying on Arrow field types alone.
///
/// \param batch Schema-backed descriptor batch with scalar columns.
/// \returns Arrow record batch containing one column per descriptor definition.
/// \throws std::invalid_argument: When the batch cannot be represented as scalar Arrow data.
std::shared_ptr<arrow::RecordBatch> ToArrowRecordBatch(const DescriptorBatch& batch);

/// \brief Convert an Arrow record batch with OEFP metadata to a descriptor batch.
///
/// \param batch Arrow record batch produced by ``ToArrowRecordBatch``.
/// \returns Schema-backed descriptor batch reconstructed from Arrow columns.
/// \throws std::invalid_argument: When required metadata or column types are invalid.
DescriptorBatch FromArrowRecordBatch(const std::shared_ptr<arrow::RecordBatch>& batch);

/// \brief Write a descriptor batch to an Arrow IPC file.
///
/// \param batch Schema-backed descriptor batch with scalar columns.
/// \param path Destination file path.
/// \throws std::invalid_argument: When the batch cannot be represented as scalar Arrow data.
/// \throws std::runtime_error: When Arrow cannot write the file.
void WriteDescriptorIpc(const DescriptorBatch& batch, const std::string& path);

/// \brief Read a descriptor batch from an Arrow IPC file.
///
/// \param path Source file path.
/// \returns Schema-backed descriptor batch reconstructed from Arrow IPC data.
/// \throws std::invalid_argument: When required metadata or column types are invalid.
/// \throws std::runtime_error: When Arrow cannot read the file.
DescriptorBatch ReadDescriptorIpc(const std::string& path);

/// \brief Write a descriptor batch to a Parquet file.
///
/// \param batch Schema-backed descriptor batch with scalar columns.
/// \param path Destination file path.
/// \throws std::invalid_argument: When the batch cannot be represented as scalar Arrow data.
/// \throws std::runtime_error: When Arrow or Parquet cannot write the file.
void WriteDescriptorParquet(const DescriptorBatch& batch, const std::string& path);

/// \brief Read a descriptor batch from a Parquet file.
///
/// \param path Source file path.
/// \returns Schema-backed descriptor batch reconstructed from Parquet data.
/// \throws std::invalid_argument: When required metadata or column types are invalid.
/// \throws std::runtime_error: When Arrow or Parquet cannot read the file.
DescriptorBatch ReadDescriptorParquet(const std::string& path);

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_ARROW_H
