#ifndef OEFP_COMPUTE_CONTEXT_H
#define OEFP_COMPUTE_CONTEXT_H

#include "oefp/mordred_intermediates.h"

#include <oechem.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace OEFP {

/// \brief Per-molecule cache of shared descriptor-computation intermediates.
///
/// A ``ComputeContext`` borrows one molecule by const reference and memoizes the
/// expensive intermediates several descriptor groups share (ring-perceived
/// working molecule, heavy-atom graph, heavy-atom distance matrix, Gasteiger
/// charges, Crippen contributions). Each accessor computes its intermediate on
/// first call, caches it, and returns a const reference to the cached value;
/// subsequent calls return the same object without recomputing.
///
/// Ownership invariant: a context is per-molecule-per-thread. It must never be
/// shared across molecules or threads, and performs no internal locking. The
/// caches are ``mutable`` so accessors can be ``const`` while still populating
/// them on demand. Each ``CalculateBatch`` worker constructs its own context per
/// molecule.
class ComputeContext {
public:
    /// \brief Construct a context borrowing ``mol`` for its lifetime.
    ///
    /// \param mol Molecule to describe. Must outlive the context; it is
    ///     borrowed by const reference and never mutated.
    explicit ComputeContext(const OEChem::OEMolBase& mol);

    /// \brief Return the ring-perceived, hybridization-assigned working molecule.
    ///
    /// A copy of the borrowed molecule with ``OEFindRingAtomsAndBonds`` and
    /// ``OEAssignHybridization`` applied, mirroring the working-molecule prep
    /// the OpenEye and Mordred descriptor groups perform.
    ///
    /// \returns Cached ring-perceived working molecule.
    const OEChem::OEGraphMol& RingPerceivedMol() const;

    /// \brief Return the shared Mordred heavy-atom adjacency graph.
    ///
    /// \returns Cached heavy-atom graph built from the borrowed molecule.
    const MordredHeavyAtomGraph& HeavyAtomGraph() const;

    /// \brief Return the heavy-atom topological distance matrix.
    ///
    /// Computed from :cpp:func:`HeavyAtomGraph` so the graph is shared rather
    /// than rebuilt.
    ///
    /// \returns Cached symmetric heavy-atom distance matrix.
    const std::vector<std::vector<std::int64_t>>& HeavyAtomDistances() const;

    /// \brief Return per-atom Gasteiger partial charges.
    ///
    /// \returns Cached Gasteiger charges for the borrowed molecule.
    const MordredGasteigerAtomCharges& GasteigerAtomCharges() const;

    /// \brief Return per-atom Crippen logP and molar-refractivity contributions.
    ///
    /// When the molecule has no defined Crippen contributions the cached value
    /// is a default-constructed :cpp:class:`MordredCrippenAtomContributions`
    /// (empty vectors), matching today's missing-value behavior downstream.
    ///
    /// \returns Cached Crippen contributions for the borrowed molecule.
    const MordredCrippenAtomContributions& CrippenContributions() const;

    /// \brief Number of times an intermediate was actually computed (not
    ///     served from cache). Used by tests to prove memoization/sharing.
    ///
    /// \returns Count of distinct intermediates computed, one per accessor,
    ///     independent of how often each accessor is called.
    std::size_t ComputeCount() const;

private:
    const OEChem::OEMolBase& mol_;
    mutable std::optional<OEChem::OEGraphMol> ring_perceived_mol_;
    mutable std::optional<MordredHeavyAtomGraph> heavy_atom_graph_;
    mutable std::optional<std::vector<std::vector<std::int64_t>>> heavy_atom_distances_;
    mutable std::optional<MordredGasteigerAtomCharges> gasteiger_charges_;
    mutable std::optional<MordredCrippenAtomContributions> crippen_contributions_;
    mutable std::size_t compute_count_ = 0;  // incremented once per actual computation
};

} // namespace OEFP

#endif // OEFP_COMPUTE_CONTEXT_H
