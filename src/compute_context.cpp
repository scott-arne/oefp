#include "oefp/compute_context.h"

#include <utility>

namespace OEFP {

ComputeContext::ComputeContext(const OEChem::OEMolBase& mol) : mol_(mol) {}

const OEChem::OEGraphMol& ComputeContext::RingPerceivedMol() const {
    if (!ring_perceived_mol_) {
        OEChem::OEGraphMol working_mol(mol_);
        OEChem::OEFindRingAtomsAndBonds(working_mol);
        OEChem::OEAssignHybridization(working_mol);
        ring_perceived_mol_.emplace(std::move(working_mol));
        ++compute_count_;
    }
    return *ring_perceived_mol_;
}

const MordredHeavyAtomGraph& ComputeContext::HeavyAtomGraph() const {
    if (!heavy_atom_graph_) {
        heavy_atom_graph_.emplace(build_mordred_heavy_atom_graph(mol_));
        ++compute_count_;
    }
    return *heavy_atom_graph_;
}

const std::vector<std::vector<std::int64_t>>& ComputeContext::HeavyAtomDistances() const {
    if (!heavy_atom_distances_) {
        // Reuse the cached graph so it is shared, never rebuilt.
        heavy_atom_distances_.emplace(compute_mordred_heavy_atom_distances(HeavyAtomGraph()));
        ++compute_count_;
    }
    return *heavy_atom_distances_;
}

const MordredGasteigerAtomCharges& ComputeContext::GasteigerAtomCharges() const {
    if (!gasteiger_charges_) {
        gasteiger_charges_.emplace(compute_gasteiger_atom_charges(mol_));
        ++compute_count_;
    }
    return *gasteiger_charges_;
}


const MordredCrippenAtomContributions& ComputeContext::CrippenContributions() const {
    if (!crippen_contributions_) {
        // A molecule with no defined Crippen contributions caches empty vectors,
        // matching today's missing-value behavior downstream.
        auto contributions = compute_crippen_atom_contributions(mol_);
        crippen_contributions_.emplace(
            contributions ? std::move(*contributions) : MordredCrippenAtomContributions{});
        ++compute_count_;
    }
    return *crippen_contributions_;
}

const std::vector<double>& ComputeContext::EStateIndices() const {
    if (!estate_indices_) {
        estate_indices_.emplace(compute_estate_indices(mol_));
        ++compute_count_;
    }
    return *estate_indices_;
}

const std::vector<double>& ComputeContext::LabuteAtomContributions() const {
    if (!labute_atom_contributions_) {
        // A molecule with no defined Labute surface caches an empty vector,
        // matching today's missing-value behavior downstream.
        auto values = compute_labute_asa(mol_);
        labute_atom_contributions_.emplace(
            values ? std::move(values->atom_contributions) : std::vector<double>{});
        ++compute_count_;
    }
    return *labute_atom_contributions_;
}

const MordredBCUTEigenvalues& ComputeContext::BCUTEigenvalues() const {
    if (!bcut_eigenvalues_) {
        // Reuse the shared heavy-atom graph, Gasteiger charges, and Crippen
        // contributions so BCUT2D shares those computations rather than rebuilding
        // them; each is served from cache when already warmed by another accessor.
        bcut_eigenvalues_.emplace(compute_rdkit_bcut_eigenvalues(
            HeavyAtomGraph(), GasteigerAtomCharges(), CrippenContributions()));
        ++compute_count_;
    }
    return *bcut_eigenvalues_;
}

std::size_t ComputeContext::ComputeCount() const {
    return compute_count_;
}

} // namespace OEFP
