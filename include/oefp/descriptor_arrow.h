#ifndef OEFP_DESCRIPTOR_ARROW_H
#define OEFP_DESCRIPTOR_ARROW_H

#include "oefp/descriptor_batch.h"

#include <memory>

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

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_ARROW_H
