#include "oefp/descriptor_source.h"

#include "oefp/mordred.h"

namespace OEFP {

std::shared_ptr<const DescriptorSchema> MordredDescriptorSource::Schema() const {
    return MordredDescriptorSchema();
}

DescriptorSet MordredDescriptorSource::Compute(const OEChem::OEMolBase& mol) const {
    return MakeMordredDescriptors(mol);
}

} // namespace OEFP
