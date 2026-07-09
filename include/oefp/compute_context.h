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
/// A ``ComputeContext`` owns a normalized copy of one molecule (see
/// :cpp:func:`normalize_molecule`) and memoizes the
/// expensive intermediates several descriptor groups share (ring-perceived
/// working molecule, heavy-atom graph, heavy-atom distance matrix, Gasteiger
/// charges, Crippen contributions, per-atom EState indices, per-atom Labute
/// surface contributions). Each accessor computes its intermediate on first
/// call, caches it, and returns a const reference to the cached value;
/// subsequent calls return the same object without recomputing.
///
/// Ownership invariant: a context is per-molecule-per-thread. It must never be
/// shared across molecules or threads, and performs no internal locking. The
/// caches are ``mutable`` so accessors can be ``const`` while still populating
/// them on demand. Each ``CalculateBatch`` worker constructs its own context per
/// molecule.
class ComputeContext {
public:
    /// \brief Construct a context owning ``normalize_molecule(mol)``.
    ///
    /// The context stores its own normalized copy of ``mol`` (neutral nitro
    /// groups rewritten to RDKit's charged form) and computes every cached
    /// intermediate from it. ``mol`` is read only during construction and need
    /// not outlive the context.
    ///
    /// \param mol Molecule to describe. Read once during construction; never
    ///     mutated.
    explicit ComputeContext(const OEChem::OEMolBase& mol);

    /// \brief Return the normalized molecule the context owns and describes.
    ///
    /// The input molecule with neutral nitro groups rewritten to the charged
    /// [N+](=O)[O-] form (see :cpp:func:`normalize_molecule`). Descriptor
    /// sources pass this to their compute entry points so every source
    /// describes the same normalized structure the cached intermediates use.
    ///
    /// \returns The owned normalized molecule.
    const OEChem::OEGraphMol& NormalizedMol() const;

    /// \brief Return the ring-perceived, hybridization-assigned working molecule.
    ///
    /// A copy of the owned normalized molecule with ``OEFindRingAtomsAndBonds`` and
    /// ``OEAssignHybridization`` applied, mirroring the working-molecule prep
    /// the OpenEye and Mordred descriptor groups perform.
    ///
    /// \returns Cached ring-perceived working molecule.
    const OEChem::OEGraphMol& RingPerceivedMol() const;

    /// \brief Return the shared Mordred heavy-atom adjacency graph.
    ///
    /// \returns Cached heavy-atom graph built from the owned normalized molecule.
    const MordredHeavyAtomGraph& HeavyAtomGraph() const;

    /// \brief Return the heavy-atom topological distance matrix.
    ///
    /// Computed from :cpp:func:`HeavyAtomGraph` so the graph is shared rather
    /// than rebuilt.
    ///
    /// \returns Cached symmetric heavy-atom distance matrix.
    const std::vector<std::vector<std::int64_t>>& HeavyAtomDistances() const;

    /// \brief Compute RDKit-faithful Gasteiger partial charges.
    ///
    /// Reproduces RDKit's ComputeGasteigerCharges: cumulene centres typed sp,
    /// molecule-wide NaN when any heavy atom lacks an RDKit Gasteiger parameter.
    /// Consumed by BOTH the Mordred descriptor source (which delegates to RDKit's
    /// ComputeGasteigerCharges in Mordred 1.2.0) and the RDKit descriptor source
    /// (PartialCharge, PEOE_VSA, BCUT2D_CHG families).
    ///
    /// \returns Cached RDKit-faithful Gasteiger charges for the owned normalized molecule.
    const MordredGasteigerAtomCharges& GasteigerAtomCharges() const;

    /// \brief Return per-atom Crippen logP and molar-refractivity contributions.
    ///
    /// When the molecule has no defined Crippen contributions the cached value
    /// is a default-constructed :cpp:class:`MordredCrippenAtomContributions`
    /// (empty vectors), matching today's missing-value behavior downstream.
    ///
    /// \returns Cached Crippen contributions for the owned normalized molecule.
    const MordredCrippenAtomContributions& CrippenContributions() const;

    /// \brief Per-atom electrotopological-state (EState) indices (RDKit model).
    ///
    /// Memoized per-atom EState vector for the owned normalized molecule, shared so the
    /// EState-derived descriptors and the EState VSA bins reuse one computation.
    ///
    /// \returns Cached per-atom EState vector for the owned normalized molecule.
    const std::vector<double>& EStateIndices() const;

    /// \brief Per-atom Labute approximate surface-area contributions.
    ///
    /// Memoized per-heavy-atom Labute surface contributions (the vector the VSA
    /// bins bucket by; its sum excludes the hydrogen shielding term that the
    /// ``LabuteASA`` total adds). Empty when Labute's model is undefined for the
    /// molecule, matching the missing-value behavior downstream.
    ///
    /// \returns Cached per-atom LabuteASA contribution vector.
    const std::vector<double>& LabuteAtomContributions() const;

    /// \brief RDKit's eight BCUT2D Burden-matrix extreme eigenvalues.
    ///
    /// Memoized high/low eigenvalue pairs of the Burden matrix under RDKit's four
    /// weightings (atomic mass, Gasteiger charge, Crippen logP, Crippen molar
    /// refractivity), built from the shared heavy-atom graph, Gasteiger charges,
    /// and Crippen contributions so the BCUT2D descriptors reuse one computation.
    /// ``defined`` is false (all eigenvalues 0) when the molecule has no heavy
    /// atoms or any heavy atom lacks RDKit Gasteiger parameters, matching RDKit's
    /// all-or-nothing behavior on those inputs.
    ///
    /// \returns Cached BCUT2D extreme eigenvalues for the owned normalized molecule.
    const MordredBCUTEigenvalues& BCUTEigenvalues() const;

    /// \brief Number of times an intermediate was actually computed (not
    ///     served from cache). Used by tests to prove memoization/sharing.
    ///
    /// \returns Count of distinct intermediates computed, one per accessor,
    ///     independent of how often each accessor is called.
    std::size_t ComputeCount() const;

private:
    const OEChem::OEGraphMol mol_;
    mutable std::optional<OEChem::OEGraphMol> ring_perceived_mol_;
    mutable std::optional<MordredHeavyAtomGraph> heavy_atom_graph_;
    mutable std::optional<std::vector<std::vector<std::int64_t>>> heavy_atom_distances_;
    mutable std::optional<MordredGasteigerAtomCharges> gasteiger_charges_;
    mutable std::optional<MordredCrippenAtomContributions> crippen_contributions_;
    mutable std::optional<std::vector<double>> estate_indices_;
    mutable std::optional<std::vector<double>> labute_atom_contributions_;
    mutable std::optional<MordredBCUTEigenvalues> bcut_eigenvalues_;
    mutable std::size_t compute_count_ = 0;  // incremented once per actual computation
};

} // namespace OEFP

#endif // OEFP_COMPUTE_CONTEXT_H
