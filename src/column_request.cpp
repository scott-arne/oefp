#include "oefp/column_request.h"

namespace OEFP {

ColumnRequest ColumnRequest::All() {
    ColumnRequest req;
    req.all_ = true;
    return req;
}

ColumnRequest ColumnRequest::Subset(std::vector<std::size_t> source_column_indices) {
    ColumnRequest req;
    req.all_ = false;
    req.indices_ = std::unordered_set<std::size_t>(
        source_column_indices.begin(),
        source_column_indices.end()
    );
    return req;
}

bool ColumnRequest::WantsAll() const {
    return all_;
}

bool ColumnRequest::Wants(std::size_t source_column_index) const {
    return all_ || indices_.count(source_column_index) != 0;
}

}  // namespace OEFP
