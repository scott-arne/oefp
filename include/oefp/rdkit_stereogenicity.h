#ifndef OEFP_RDKIT_STEREOGENICITY_H
#define OEFP_RDKIT_STEREOGENICITY_H

#include <oechem.h>

#include <vector>

namespace OEFP {

/// \brief Per-heavy-atom potential-stereogenicity flags for one molecule.
///
/// Both vectors are indexed by heavy-atom iteration order: the order in which
/// non-hydrogen atoms are visited by ``OEMolBase::GetAtoms()``. This matches
/// :cpp:func:`build_mordred_heavy_atom_graph` so the flags align, index for
/// index, with the SPS caller's heavy-atom graph.
struct RDKitStereogenicity {
    /// True where the heavy atom is a potential tetrahedral stereocenter,
    /// reproducing RDKit ``findPotentialStereo`` filtered to ``Atom_Tetrahedral``.
    std::vector<bool> atom_is_potential_stereocenter;
    /// True where the heavy atom is an endpoint of a potential-stereo double
    /// bond, reproducing the legacy ``MolOps::findPotentialStereoBonds``.
    std::vector<bool> atom_on_potential_stereo_bond;
};

/// \brief Reproduce RDKit's potential-stereogenicity perception natively.
///
/// Reproduces two independent RDKit code paths on an already-perceived molecule
/// (rings/aromaticity/hybridization must be assigned by the caller):
///
/// - Atoms: RDKit ``Chirality::findPotentialStereo(mol, cleanIt=false,
///   findPossible=true)`` keeping only ``StereoInfo`` entries whose type is
///   ``Atom_Tetrahedral`` (the same filter ``FindMolChiralCenters(...,
///   useLegacyImplementation=false)`` applies). This covers classic centers,
///   para/ring/cage (dependent) stereocenters, and the hypervalent-hydride path.
/// - Bonds: the legacy ``MolOps::findPotentialStereoBonds`` (which SPS calls via
///   ``rdmolops.FindPotentialStereoBonds``), marking both endpoints of every
///   double bond that carries non-``STEREONONE`` stereo afterwards.
///
/// The perception is hydrogen-representation-agnostic: heavy neighbours are
/// counted with ``GetHvyDegree`` and hydrogens with ``GetTotalHCount``, so
/// explicit/bracket hydrogens do not perturb the result.
///
/// \param mol Molecule with ring, aromaticity, and hybridization perceived.
/// \returns Per-heavy-atom stereocenter and stereo-bond-endpoint flags.
RDKitStereogenicity rdkit_potential_stereogenicity(const OEChem::OEMolBase& mol);

} // namespace OEFP

#endif // OEFP_RDKIT_STEREOGENICITY_H
