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
/// \returns Per-atom charges, implicit-hydrogen charges, and atom identifiers.
MordredGasteigerAtomCharges compute_gasteiger_atom_charges(const OEChem::OEMolBase& mol);

/// \brief Compute per-atom Crippen logP and molar-refractivity contributions.
///
/// \param mol Molecule to describe.
/// \returns Per-atom Crippen contributions, or ``std::nullopt`` when undefined.
std::optional<MordredCrippenAtomContributions> compute_crippen_atom_contributions(
    const OEChem::OEMolBase& mol);

} // namespace OEFP

#endif // OEFP_MORDRED_INTERMEDIATES_H
