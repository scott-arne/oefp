#ifndef OEFP_MORDRED_INTERMEDIATES_H
#define OEFP_MORDRED_INTERMEDIATES_H

#include <oechem.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace OEFP {

struct MordredCrippenAtomContributions {
    std::vector<double> logp;
    std::vector<double> mr;
    std::vector<unsigned int> atom_ids;
};

struct MordredGasteigerAtomCharges {
    std::vector<double> charges;
    std::vector<double> hydrogen_charges;
    std::vector<unsigned int> atom_ids;
};

struct MordredLabuteAsaValues {
    double total = 0.0;
    std::vector<double> atom_contributions;
    std::vector<unsigned int> atom_ids;
};

struct PathCountNeighbor {
    std::size_t atom_index;
    double bond_order;
};

struct MordredHeavyAtomGraph {
    std::vector<const OEChem::OEAtomBase*> atoms;
    std::vector<std::vector<PathCountNeighbor>> adjacency;
    std::vector<std::pair<std::size_t, std::size_t>> bonds;
    std::vector<std::vector<std::size_t>> bond_neighbors;
};

/// \brief The eight extreme eigenvalues of RDKit's Burden matrix (BCUT2D).
///
/// Each ``*_high`` / ``*_low`` pair is the largest / smallest eigenvalue of the
/// symmetric Burden matrix whose diagonal is the per-heavy-atom property under
/// one of RDKit's four weightings and whose off-diagonal for bonded atoms is
/// ``1/sqrt(bond_order)`` (non-bonded pairs share the constant ``0.001``).
/// ``defined`` is false when the molecule has no heavy atoms or when any heavy
/// atom lacks RDKit Gasteiger parameters (RDKit's ``BCUT2D`` raises there, so all
/// eight values are undefined together, mirroring RDKit's all-or-nothing model).
struct MordredBCUTEigenvalues {
    double mw_high = 0.0;
    double mw_low = 0.0;
    double chg_high = 0.0;
    double chg_low = 0.0;
    double logp_high = 0.0;
    double logp_low = 0.0;
    double mr_high = 0.0;
    double mr_low = 0.0;
    bool defined = false;
};

/// \brief Build the heavy-atom adjacency graph used by Mordred graph descriptors.
///
/// \param mol Molecule to describe.
/// \returns Heavy-atom graph with adjacency, bond, and bond-neighbor tables.
MordredHeavyAtomGraph build_mordred_heavy_atom_graph(const OEChem::OEMolBase& mol);

/// \brief Compute the heavy-atom topological distance matrix from a heavy-atom graph.
///
/// \param graph Heavy-atom graph produced by :cpp:func:`build_mordred_heavy_atom_graph`.
/// \returns Symmetric distance matrix; disconnected pairs use a large sentinel value.
std::vector<std::vector<std::int64_t>> compute_mordred_heavy_atom_distances(
    const MordredHeavyAtomGraph& graph);

/// \brief Compute per-atom Gasteiger partial charges as used by Mordred.
///
/// \param mol Molecule to describe.
/// \param suppress_hydrogens When true, compute on H-suppressed mol (Mordred/RDKit);
///     when false, retain explicit hydrogens.
/// \param cumulene_sp When true, type cumulene centres "sp" (RDKit fidelity);
///     when false, leave them "sp2" (Mordred 1.2.0 fidelity).
/// \returns Per-atom charges, implicit-hydrogen charges, and atom identifiers.
MordredGasteigerAtomCharges compute_gasteiger_atom_charges(
    const OEChem::OEMolBase& mol,
    bool suppress_hydrogens,
    bool cumulene_sp);

MordredGasteigerAtomCharges compute_gasteiger_atom_charges(const OEChem::OEMolBase& mol);

/// \brief Compute per-atom Crippen logP and molar-refractivity contributions.
///
/// \param mol Molecule to describe.
/// \returns Per-atom Crippen contributions, or ``std::nullopt`` when undefined.
std::optional<MordredCrippenAtomContributions> compute_crippen_atom_contributions(
    const OEChem::OEMolBase& mol);

/// \brief Compute the total Wildman-Crippen logP and molar-refractivity sums.
///
/// Sums the per-atom Wildman-Crippen contributions over the hydrogen-added
/// molecule, reproducing RDKit's ``MolLogP`` and ``MolMR`` (which add explicit
/// hydrogens before summing, so the total differs from summing the
/// hydrogen-suppressed :cpp:func:`compute_crippen_atom_contributions`). This is a
/// thin export of the existing Mordred ``SLogP``/``SMR`` computation, shared so
/// the RDKit ``Crippen`` family reuses the identical, oracle-verified model.
///
/// \param mol Molecule to describe.
/// \returns Pair of total logP and total molar refractivity.
std::pair<double, double> compute_crippen_contribution_sums(const OEChem::OEMolBase& mol);

/// \brief Compute Labute's approximate surface area (total plus per-atom).
///
/// Thin export of the Mordred ``LabuteASA`` computation. ``total`` reproduces
/// RDKit's ``LabuteASA`` descriptor (heavy-atom contributions plus the hydrogen
/// shielding term); ``atom_contributions`` holds the per-heavy-atom surface
/// contributions the VSA bins bucket by (their sum excludes the hydrogen term).
///
/// \param mol Molecule to describe.
/// \returns Labute surface-area values, or ``std::nullopt`` when undefined.
std::optional<MordredLabuteAsaValues> compute_labute_asa(const OEChem::OEMolBase& mol);

/// \brief Compute per-atom electrotopological-state (EState) indices.
///
/// Thin export of the Mordred EState computation: suppresses hydrogens, builds
/// the heavy-atom graph, and accumulates the intrinsic-state field over
/// topological distances. Reproduces RDKit's ``rdkit.Chem.EState.EStateIndices``
/// (verified against the oracle through the shared EState/VSA bins).
///
/// \param mol Molecule to describe.
/// \returns Per-heavy-atom EState indices in heavy-atom order.
std::vector<double> compute_estate_indices(const OEChem::OEMolBase& mol);

/// \brief Compute RDKit's eight BCUT2D Burden-matrix extreme eigenvalues.
///
/// Builds RDKit's Burden matrix (diagonal = per-heavy-atom property; bonded
/// off-diagonal = ``1/sqrt(bond_order)``; every other pair = ``0.001``) for each
/// of RDKit's four weightings — atomic mass, Gasteiger charge, Crippen logP, and
/// Crippen molar refractivity — and returns each matrix's largest and smallest
/// eigenvalues via the shared symmetric (Jacobi) eigensolver. The mass diagonal
/// uses RDKit's standard atomic weights (isotope-aware); the charge diagonal uses
/// the per-heavy-atom Gasteiger charges without the redistributed hydrogen term,
/// matching RDKit's ``removeAllHs`` + ``computeGasteigerCharges`` inputs; the
/// logP/MR diagonals use the per-heavy-atom Crippen contributions.
///
/// The result is ``defined == false`` (all eigenvalues left at 0) when the
/// molecule has no heavy atoms or when any heavy atom lacks RDKit Gasteiger
/// parameters, reproducing RDKit's all-or-nothing exception on those inputs.
///
/// \param graph Heavy-atom graph produced by :cpp:func:`build_mordred_heavy_atom_graph`.
/// \param gasteiger_charges Per-atom Gasteiger charges for the same molecule.
/// \param crippen_contributions Per-heavy-atom Crippen logP / MR contributions.
/// \returns The eight extreme eigenvalues, with ``defined`` set accordingly.
MordredBCUTEigenvalues compute_rdkit_bcut_eigenvalues(
    const MordredHeavyAtomGraph& graph,
    const MordredGasteigerAtomCharges& gasteiger_charges,
    const MordredCrippenAtomContributions& crippen_contributions);

/// \brief Enumerate the symmetrized smallest-set-of-smallest-rings (SSSR).
///
/// Returns the ring set RDKit's ``RingInfo`` reports (``symmetrizeSSSR``), which
/// is the ring perception Mordred's ring-count descriptors already use. Each
/// ring is the ordered list of its heavy-atom members, with consecutive atoms
/// bonded and the last atom bonded back to the first; hydrogens are excluded.
/// Enumeration is purely topological, so it does not depend on ring or
/// aromaticity perception having been applied.
///
/// Sharing this with the RDKit ``RingCounts`` family keeps its ``RingCount``
/// equal to RDKit's symmetrized count on fused and caged systems, where the
/// plain cyclomatic ring number diverges (for example cubane and propellanes).
///
/// \param mol Molecule to perceive rings for.
/// \returns Symmetrized SSSR rings as ordered heavy-atom pointer cycles.
std::vector<std::vector<const OEChem::OEAtomBase*>> compute_symmetrized_sssr_rings(
    const OEChem::OEMolBase& mol);

} // namespace OEFP

#endif // OEFP_MORDRED_INTERMEDIATES_H
