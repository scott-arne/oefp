#ifndef OEFP_INPUT_NORMALIZATION_H
#define OEFP_INPUT_NORMALIZATION_H
#include <oechem.h>
namespace OEFP {
/// \brief Return a copy of ``mol`` with neutral nitro groups rewritten to
///     RDKit's charged convention: N(=O)=O becomes [N+](=O)[O-]. Idempotent.
///
/// Hydrogen suppression is intentionally NOT performed here. Suppressing
/// explicit/bracket hydrogens at this shared boundary regresses the Gasteiger
/// charge model for molecules carrying stereo bracket hydrogens (e.g. the
/// ``[C@H]`` in L-alanine), so explicit-hydrogen handling remains the
/// responsibility of the individual descriptor sources that require it.
OEChem::OEGraphMol normalize_molecule(const OEChem::OEMolBase& mol);
}  // namespace OEFP
#endif
