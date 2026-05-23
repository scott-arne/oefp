#ifndef OEFP_DESCRIPTOR_SELECTION_H
#define OEFP_DESCRIPTOR_SELECTION_H

#include "oefp/descriptor_schema.h"

#include <cstddef>
#include <string>
#include <vector>

namespace OEFP {

class DescriptorSelection {
public:
    static DescriptorSelection Names(std::vector<std::string> names);
    static DescriptorSelection Group(std::string group);
    static DescriptorSelection Indices(std::vector<std::size_t> indices);

    std::vector<std::size_t> Resolve(const DescriptorSchema& schema) const;

private:
    enum class Mode {
        Names,
        Group,
        Indices,
    };

    explicit DescriptorSelection(Mode mode);

    Mode mode_;
    std::vector<std::string> names_;
    std::string group_;
    std::vector<std::size_t> indices_;
};

} // namespace OEFP

#endif // OEFP_DESCRIPTOR_SELECTION_H
