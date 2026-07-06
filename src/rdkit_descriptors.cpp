#include "oefp/rdkit_descriptors.h"

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

} // namespace OEFP
