#include "oefp/descriptor_selection.h"

#include <stdexcept>
#include <utility>

namespace OEFP {

DescriptorSelection::DescriptorSelection(Mode mode)
    : mode_(mode) {}

DescriptorSelection DescriptorSelection::Names(std::vector<std::string> names) {
    DescriptorSelection selection(Mode::Names);
    selection.names_ = std::move(names);
    return selection;
}

DescriptorSelection DescriptorSelection::Group(std::string group) {
    DescriptorSelection selection(Mode::Group);
    selection.group_ = std::move(group);
    return selection;
}

DescriptorSelection DescriptorSelection::Indices(std::vector<std::size_t> indices) {
    DescriptorSelection selection(Mode::Indices);
    selection.indices_ = std::move(indices);
    return selection;
}

std::vector<std::size_t> DescriptorSelection::Resolve(const DescriptorSchema& schema) const {
    switch (mode_) {
    case Mode::Names: {
        std::vector<std::size_t> indices;
        indices.reserve(names_.size());
        for (const auto& name : names_) {
            indices.push_back(schema.IndexOf(name));
        }
        return indices;
    }
    case Mode::Group:
        return schema.IndicesForGroup(group_);
    case Mode::Indices:
        for (const auto index : indices_) {
            if (index >= schema.Size()) {
                throw std::out_of_range("Descriptor selection index is out of range.");
            }
        }
        return indices_;
    }
    throw std::invalid_argument("Descriptor selection mode is invalid.");
}

} // namespace OEFP
