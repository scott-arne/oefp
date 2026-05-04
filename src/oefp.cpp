#include "oefp/oefp.h"

#include <oechem.h>

namespace OEFP {

double calculate_molecular_weight(const OEChem::OEMolBase& mol) {
    return OEChem::OECalculateMolecularWeight(mol);
}

} // namespace OEFP
