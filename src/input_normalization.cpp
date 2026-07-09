#include "oefp/input_normalization.h"
namespace OEFP {
namespace {
// Rewrite each neutral hypervalent nitro nitrogen (N, formal charge 0, exactly
// two double bonds to terminal degree-1 neutral oxygens) to [N+](=O)[O-]:
// convert one N=O to a single bond, set that oxygen's charge to -1, set the
// nitrogen's charge to +1. RDKit normalizes N(=O)=O to this canonical form;
// OpenEye does not (OENormalize/OEAssignFormalCharges leave it neutral).
void rewrite_neutral_nitro(OEChem::OEMolBase& mol) {
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (atom->GetAtomicNum() != 7 || atom->GetFormalCharge() != 0) {
            continue;
        }
        OEChem::OEBondBase* first_double_o = nullptr;
        unsigned int double_o = 0u;
        for (OESystem::OEIter<OEChem::OEBondBase> bond = atom->GetBonds(); bond; ++bond) {
            OEChem::OEAtomBase* nbr = bond->GetNbr(atom);
            if (bond->GetOrder() == 2u && nbr != nullptr && nbr->GetAtomicNum() == 8
                && nbr->GetDegree() == 1u && nbr->GetFormalCharge() == 0) {
                ++double_o;
                if (first_double_o == nullptr) {
                    first_double_o = &*bond;
                }
            }
        }
        if (double_o == 2u && first_double_o != nullptr) {
            OEChem::OEAtomBase* oxygen = first_double_o->GetNbr(atom);
            first_double_o->SetOrder(1u);
            oxygen->SetFormalCharge(-1);
            oxygen->SetImplicitHCount(0u);
            atom->SetFormalCharge(1);
            atom->SetImplicitHCount(0u);
        }
    }
}
}  // namespace

OEChem::OEGraphMol normalize_molecule(const OEChem::OEMolBase& mol) {
    OEChem::OEGraphMol working(mol);
    OEChem::OESuppressHydrogens(working);
    rewrite_neutral_nitro(working);
    return working;
}
}  // namespace OEFP
