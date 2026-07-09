#ifndef OEFP_INPUT_NORMALIZATION_H
#define OEFP_INPUT_NORMALIZATION_H
#include <oechem.h>
namespace OEFP {
/// \brief Return a copy of ``mol`` normalized to RDKit's input convention:
///     explicit/bracket hydrogens suppressed and neutral nitro (N(=O)=O)
///     rewritten to the charged [N+](=O)[O-] form. Idempotent.
OEChem::OEGraphMol normalize_molecule(const OEChem::OEMolBase& mol);
}  // namespace OEFP
#endif
