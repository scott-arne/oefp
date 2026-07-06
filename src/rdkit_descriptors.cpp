#include "oefp/rdkit_descriptors.h"
#include "oefp/descriptor_source.h"

namespace OEFP {

DescriptorSet MakeRDKitDescriptors(const OEChem::OEMolBase& mol) {
    ComputeContext ctx(mol);
    return MakeRDKitDescriptors(mol, ctx, ColumnRequest::All());
}

DescriptorSet MakeRDKitDescriptors(const OEChem::OEMolBase& /*mol*/,
                                   ComputeContext& /*ctx*/,
                                   const ColumnRequest& /*request*/) {
    // No families computed yet; every column is left missing. Families are
    // added task-by-task via the group registry (Task 5+).
    DescriptorSetBuilder builder(RDKitDescriptorSchema());
    return builder.Build();
}

std::shared_ptr<const DescriptorSchema> RDKitDescriptorSource::Schema() const {
    return RDKitDescriptorSchema();
}

DescriptorSet RDKitDescriptorSource::Compute(const OEChem::OEMolBase& mol) const {
    ComputeContext ctx(mol);
    return Compute(mol, ctx, ColumnRequest::All());
}

DescriptorSet RDKitDescriptorSource::Compute(const OEChem::OEMolBase& mol,
                                             ComputeContext& ctx,
                                             const ColumnRequest& request) const {
    return MakeRDKitDescriptors(mol, ctx, request);
}

} // namespace OEFP
