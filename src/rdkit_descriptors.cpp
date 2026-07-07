#include "oefp/rdkit_descriptors.h"

#include "oefp/descriptor_source.h"
#include "oefp/molecular_properties.h"
#include "oefp/morgan.h"

#include <oesystem.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace OEFP {
namespace {

/// \brief Gate each descriptor column write by a :cpp:class:`ColumnRequest`.
///
/// Wraps a :cpp:class:`DescriptorSetBuilder` and forwards ``Set`` only when the
/// request wants the named column's schema index. This mirrors the identical
/// choke point in ``src/mordred.cpp``: every ``set_*`` helper writes through it,
/// so column pruning never changes how a value is computed, only whether it is
/// stored. A small amount of duplication with Mordred is intentional here so the
/// RDKit source stays self-contained.
class RequestGatedBuilder {
public:
    RequestGatedBuilder(
        DescriptorSetBuilder& builder,
        const DescriptorSchema& schema,
        const ColumnRequest& request)
        : builder_(builder), schema_(schema), request_(request) {}

    void Set(const std::string& name, DescriptorValue value) {
        const auto index = schema_.IndexOf(name);
        if (request_.Wants(index)) {
            builder_.Set(index, std::move(value));
        }
    }

private:
    DescriptorSetBuilder& builder_;
    const DescriptorSchema& schema_;
    const ColumnRequest& request_;
};

void set_int(RequestGatedBuilder& builder, const std::string& name, std::int64_t value) {
    builder.Set(name, DescriptorValue::Int(value));
}

void set_float(RequestGatedBuilder& builder, const std::string& name, double value) {
    builder.Set(name, DescriptorValue::Float(value));
}

/// \brief Number of outer-shell (valence) electrons per element.
///
/// Mirrors RDKit's ``PeriodicTable::GetNOuterElecs`` so ``NumValenceElectrons``
/// reproduces the oracle without an RDKit runtime dependency. Duplicated from the
/// identical table in ``src/mordred.cpp`` to keep the RDKit source standalone.
std::int64_t rdkit_outer_electrons(std::uint32_t atomic_number) {
    static constexpr std::array<int, 119> outer_electrons{{
        0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5,
        6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3,
        4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11,
        2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 3, 4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2,
    }};
    if (atomic_number >= outer_electrons.size()) {
        return 0;
    }
    return outer_electrons[atomic_number];
}

/// \brief Default (neutral) valence per element, or ``-1`` when undefined.
///
/// Mirrors the ``gasteiger_default_valence`` table in ``src/mordred.cpp``. Used
/// to reproduce RDKit's ``NumRadicalElectrons`` as the shortfall between an
/// atom's default valence and its realized valence.
int rdkit_default_valence(std::uint32_t atomic_number) {
    static constexpr std::array<int, 119> default_valences{{
        -1, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3,
        2, 1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 3, 4, 3, 2, 1, 0, 1, 2, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, 3, 2, 3, 2, 1, 0, 1, 2, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2,
        3, 2, 1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    }};
    if (atomic_number >= default_valences.size()) {
        return -1;
    }
    return default_valences[atomic_number];
}

/// \brief Count root-atom (pattern index 0) matches of the given SMARTS.
///
/// Mirrors ``count_unique_smarts_root_atoms`` in ``src/mordred.cpp``: it collects
/// the distinct target atoms matched by the first SMARTS atom, reproducing
/// RDKit's ``GetSubstructMatches(..., uniquify=1)`` count for descriptors whose
/// definition counts unique matched root atoms (Lipinski HBD/HBA).
std::int64_t count_smarts_root_atoms(const OEChem::OEMolBase& mol, const char* smarts) {
    OEChem::OESubSearch search(smarts);
    if (!search) {
        return 0;
    }
    std::unordered_set<unsigned int> matched;
    for (OESystem::OEIter<OEChem::OEMatchBase> match = search.Match(mol, true); match; ++match) {
        for (OESystem::OEIter<OEChem::OEMatchPair<OEChem::OEAtomBase>> atom_match =
                 match->GetAtoms();
             atom_match; ++atom_match) {
            if (atom_match->pattern != nullptr && atom_match->target != nullptr
                && atom_match->pattern->GetIdx() == 0u) {
                matched.insert(atom_match->target->GetIdx());
            }
        }
    }
    return static_cast<std::int64_t>(matched.size());
}

/// \brief Count distinct target-atom sets matched by the given SMARTS.
///
/// For bond-oriented patterns (rotatable bonds, amide bonds) RDKit counts unique
/// matches. Deduplicating on the sorted set of matched target atoms reproduces
/// that count under OpenEye's substructure search.
std::int64_t count_unique_smarts_matches(const OEChem::OEMolBase& mol, const char* smarts) {
    OEChem::OESubSearch search(smarts);
    if (!search) {
        return 0;
    }
    std::set<std::vector<unsigned int>> matches;
    for (OESystem::OEIter<OEChem::OEMatchBase> match = search.Match(mol, true); match; ++match) {
        std::vector<unsigned int> atoms;
        for (OESystem::OEIter<OEChem::OEMatchPair<OEChem::OEAtomBase>> atom_match =
                 match->GetAtoms();
             atom_match; ++atom_match) {
            if (atom_match->target != nullptr) {
                atoms.push_back(atom_match->target->GetIdx());
            }
        }
        std::sort(atoms.begin(), atoms.end());
        matches.insert(std::move(atoms));
    }
    return static_cast<std::int64_t>(matches.size());
}

// RDKit's Lipinski HBD/HBA and strict rotatable-bond SMARTS, reproduced verbatim
// from rdkit.Chem.Lipinski / the RDKit C++ defaults so the OpenEye substructure
// counts match the oracle on the conformance panel.
constexpr const char* kHDonorSmarts =
    "[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]";
constexpr const char* kHAcceptorSmarts =
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),"
    "$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]";
constexpr const char* kAmideBondSmarts = "C(=O)N";
constexpr const char* kRotatableBondSmarts =
    "[!$(*#*)&!D1&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&"
    "!$(C([CH3])([CH3])[CH3])&!$([CD3](=[N,O,S])-!@[#7,O,S!D1])&"
    "!$([#7,O,S!D1]-!@[CD3]=[N,O,S])&!$([CD3](=[N+])-!@[#7!D1])&"
    "!$([#7!D1]-!@[CD3]=[N+])]-!@[!$(*#*)&!D1&!$(C(F)(F)F)&"
    "!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]";

std::int64_t count_valence_electrons(const OEChem::OEMolBase& mol) {
    // RDKit: sum over all atoms of NOuterElecs(Z) - formalCharge + totalNumHs.
    // Implicit and explicit hydrogens are folded in via GetTotalHCount on heavy
    // atoms; explicit hydrogen atoms contribute NOuterElecs(1) - formalCharge.
    std::int64_t total = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        total += rdkit_outer_electrons(atomic_number) - atom->GetFormalCharge();
        if (!is_hydrogen(*atom)) {
            total += static_cast<std::int64_t>(atom->GetTotalHCount());
        }
    }
    return total;
}

std::int64_t count_radical_electrons(const OEChem::OEMolBase& mol) {
    // RDKit: sum of GetNumRadicalElectrons over all atoms. OpenEye represents an
    // open shell as valence below the element's neutral default, so the radical
    // count is the positive shortfall (default_valence - realized_valence) for
    // neutral atoms with a defined default valence.
    std::int64_t total = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (atom->GetFormalCharge() != 0) {
            continue;
        }
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        const auto default_valence = rdkit_default_valence(atomic_number);
        if (default_valence < 0) {
            continue;
        }
        const auto shortfall = default_valence - static_cast<int>(atom->GetValence());
        if (shortfall > 0) {
            total += shortfall;
        }
    }
    return total;
}

std::int64_t count_heteroatoms(const OEChem::OEMolBase& mol) {
    std::int64_t heteroatoms = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        if (atomic_number != 1u && atomic_number != 6u) {
            ++heteroatoms;
        }
    }
    return heteroatoms;
}

std::int64_t count_nitrogen_oxygen(const OEChem::OEMolBase& mol) {
    // RDKit NOCount = CalcNumLipinskiHBA = number of nitrogen and oxygen atoms.
    std::int64_t count = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        if (atomic_number == 7u || atomic_number == 8u) {
            ++count;
        }
    }
    return count;
}

std::int64_t count_nh_oh_hydrogens(const OEChem::OEMolBase& mol) {
    // RDKit NHOHCount = CalcNumLipinskiHBD = total hydrogens attached to any
    // nitrogen or oxygen atom.
    std::int64_t hydrogens = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        if (atomic_number == 7u || atomic_number == 8u) {
            hydrogens += static_cast<std::int64_t>(atom->GetTotalHCount());
        }
    }
    return hydrogens;
}

std::int64_t count_bridgehead_atoms(const OEChem::OEMolBase& mol) {
    // Mirror Mordred: OpenEye's bridgehead perception flags fused-ring aromatic
    // atoms that RDKit's CalcNumBridgeheadAtoms does not, so aromatic atoms are
    // excluded.
    OEChem::OEIsBridgeHead is_bridgehead(mol);
    std::int64_t bridgeheads = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (!atom->IsAromatic() && is_bridgehead(*atom)) {
            ++bridgeheads;
        }
    }
    return bridgeheads;
}

std::int64_t count_spiro_atoms(const OEChem::OEMolBase& mol) {
    // A spiro atom joins two rings at a single shared atom, so it carries four
    // ring bonds. Matches Mordred's is_spiro_atom.
    std::int64_t spiro = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (!atom->IsInRing()) {
            continue;
        }
        std::uint32_t ring_bonds = 0u;
        for (OESystem::OEIter<OEChem::OEBondBase> bond = atom->GetBonds(); bond; ++bond) {
            if (bond->IsInRing()) {
                ++ring_bonds;
            }
        }
        if (ring_bonds >= 4u) {
            ++spiro;
        }
    }
    return spiro;
}

std::int64_t count_atom_stereo_centers(const OEChem::OEMolBase& mol) {
    std::int64_t centers = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (atom->IsChiral()) {
            ++centers;
        }
    }
    return centers;
}

std::int64_t count_unspecified_atom_stereo_centers(const OEChem::OEMolBase& mol) {
    std::int64_t centers = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (atom->IsChiral() && !atom->HasStereoSpecified()) {
            ++centers;
        }
    }
    return centers;
}

double fraction_csp3(const OEChem::OEMolBase& mol) {
    // RDKit FractionCSP3 = (number of sp3 carbons) / (number of carbons). A carbon
    // is sp3 when it has four connections (heavy neighbors plus hydrogens) and no
    // multiple or aromatic bonds. This graph-derived rule matches RDKit even where
    // OpenEye's hybridization perception disagrees (e.g. a diazo carbanion).
    std::int64_t carbons = 0;
    std::int64_t sp3_carbons = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (atom->GetAtomicNum() != 6) {
            continue;
        }
        ++carbons;
        bool all_single = true;
        for (OESystem::OEIter<OEChem::OEBondBase> bond = atom->GetBonds(); bond; ++bond) {
            if (bond->GetOrder() != 1u || bond->IsAromatic()) {
                all_single = false;
                break;
            }
        }
        if (all_single && atom->GetDegree() == 4u) {
            ++sp3_carbons;
        }
    }
    if (carbons == 0) {
        return 0.0;
    }
    return static_cast<double>(sp3_carbons) / static_cast<double>(carbons);
}

double fp_density_morgan(const OEChem::OEMolBase& mol, std::uint32_t radius) {
    // RDKit FpDensityMorgan{r} = (distinct nonzero environments of the unfolded
    // sparse count Morgan fingerprint at radius r) / heavy-atom count.
    const auto heavy_atoms = HeavyAtomCount(mol);
    if (heavy_atoms == 0u) {
        return 0.0;
    }
    MorganOptions options;
    options.radius = radius;
    const auto fingerprint = MakeMorganSparseCountFingerprint(mol, options);
    return static_cast<double>(fingerprint.NonzeroCount()) / static_cast<double>(heavy_atoms);
}

// SPS (RDKit SpacialScore) is intentionally NOT implemented here. It is deferred
// and left MISSING: reproducing it requires RDKit's potential-stereocenter
// enumeration and exact conjugated-heteroatom hybridization model, neither
// exposed by OpenEye (a best-effort native port deviated by up to 50% on cage
// systems). A dedicated deep-dive follow-up task will own it.

// RDKit-internal group-to-group intermediates that are NOT molecule-level
// shareable (those live on ComputeContext instead). Per spec §4.3, the per-atom
// EState / LabuteASA vectors and BCUT2D eigenvalues are ComputeContext
// accessors, NOT fields here. Kept for parity with Mordred's
// MordredGroupArtifacts and to carry the Connectivity group's Kappa values to
// the `Phi` column (Task 6).
struct RDKitGroupArtifacts {
    // Populated by the Connectivity group (Task 6) for the CountsWeights `Phi`
    // column; extended only for intermediates that are truly not context-level.
};

enum class RDKitGroupId {
    CountsWeights,
    Count_,
};

/// \brief One compute group: the columns it fills, the groups it depends on, and
///     the computation that fills the artifact and emits columns.
struct RDKitGroup {
    RDKitGroupId id;
    std::vector<std::size_t> emitted_columns;
    std::vector<RDKitGroupId> dependency_groups;
    std::function<void(const OEChem::OEMolBase&, ComputeContext&,
                       RDKitGroupArtifacts&, const ColumnRequest&,
                       RequestGatedBuilder&)>
        run;
};

/// \brief Resolve a list of schema column names to indices once.
std::vector<std::size_t> rdkit_column_indices(
    const DescriptorSchema& schema,
    const std::vector<std::string>& names) {
    std::vector<std::size_t> indices;
    indices.reserve(names.size());
    for (const auto& name : names) {
        indices.push_back(schema.IndexOf(name));
    }
    return indices;
}

const std::vector<RDKitGroup>& rdkit_group_registry() {
    static const std::vector<RDKitGroup> registry = [] {
        const auto schema = RDKitDescriptorSchema();
        const DescriptorSchema& s = *schema;
        std::vector<RDKitGroup> groups;

        // Group: rdkit:CountsWeights — the 21 DEPENDENCY-FREE descriptors this
        // task computes. `Phi` is a member but needs the Connectivity group's
        // Kappa values, so it is added in Task 6. `SPS` is a member but is
        // deferred and left MISSING (see the deferral note at the SPS emission
        // site below). Neither is listed in emitted_columns so the subtractive
        // pruning check treats them as unrequested/missing columns.
        groups.push_back(RDKitGroup{
            RDKitGroupId::CountsWeights,
            rdkit_column_indices(
                s,
                {"MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
                 "NumRadicalElectrons", "FpDensityMorgan1", "FpDensityMorgan2",
                 "FpDensityMorgan3", "FractionCSP3", "HeavyAtomCount", "NHOHCount",
                 "NOCount", "NumAmideBonds", "NumAtomStereoCenters", "NumBridgeheadAtoms",
                 "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
                 "NumSpiroAtoms", "NumUnspecifiedAtomStereoCenters"}),
            {},  // dependency-free
            [](const OEChem::OEMolBase&, ComputeContext& ctx,
               RDKitGroupArtifacts&, const ColumnRequest&,
               RequestGatedBuilder& builder) {
                // The shared ring-perceived, hybridization-assigned working
                // molecule carries the ring, hybridization, and chirality state
                // every count below relies on, and reuses the cached preparation.
                const OEChem::OEGraphMol& mol = ctx.RingPerceivedMol();

                set_int(builder, "HeavyAtomCount",
                        static_cast<std::int64_t>(HeavyAtomCount(mol)));
                set_int(builder, "NumValenceElectrons", count_valence_electrons(mol));
                set_int(builder, "NumRadicalElectrons", count_radical_electrons(mol));
                set_int(builder, "NumHeteroatoms", count_heteroatoms(mol));
                set_int(builder, "NOCount", count_nitrogen_oxygen(mol));
                set_int(builder, "NHOHCount", count_nh_oh_hydrogens(mol));
                set_int(builder, "NumAmideBonds",
                        count_unique_smarts_matches(mol, kAmideBondSmarts));
                set_int(builder, "NumRotatableBonds",
                        count_unique_smarts_matches(mol, kRotatableBondSmarts));
                set_int(builder, "NumHDonors", count_smarts_root_atoms(mol, kHDonorSmarts));
                set_int(builder, "NumHAcceptors",
                        count_smarts_root_atoms(mol, kHAcceptorSmarts));
                set_int(builder, "NumBridgeheadAtoms", count_bridgehead_atoms(mol));
                set_int(builder, "NumSpiroAtoms", count_spiro_atoms(mol));
                set_int(builder, "NumAtomStereoCenters", count_atom_stereo_centers(mol));
                set_int(builder, "NumUnspecifiedAtomStereoCenters",
                        count_unspecified_atom_stereo_centers(mol));

                set_float(builder, "ExactMolWt", ExactMolecularWeight(mol));
                // RDKit MolWt = sum of standard AVERAGE atomic weights per element
                // (including implicit + explicit H); AverageMolecularWeight is
                // Mordred's AMW and is the WRONG helper here.
                set_float(builder, "MolWt", StandardMolecularWeight(mol));
                // RDKit HeavyAtomMolWt = MolWt minus the average mass of every H.
                set_float(builder, "HeavyAtomMolWt", heavy_atom_standard_weight(mol));
                set_float(builder, "FractionCSP3", fraction_csp3(mol));
                set_float(builder, "FpDensityMorgan1", fp_density_morgan(mol, 1u));
                set_float(builder, "FpDensityMorgan2", fp_density_morgan(mol, 2u));
                set_float(builder, "FpDensityMorgan3", fp_density_morgan(mol, 3u));
                // SPS deferred — RDKit SpacialScore depends on RDKit's
                // potential-stereocenter enumeration (FindMolChiralCenters with
                // includeUnassigned=True, useLegacyImplementation=False) and its
                // exact conjugated-heteroatom hybridization model, neither exposed
                // by OpenEye (up to 50% deviation on cage systems such as cubane).
                // Left MISSING rather than shipping a knowingly-wrong value; a
                // dedicated deep-dive follow-up task is needed. See the SPS
                // deferral note in the task-3 report.
            }});

        return groups;
    }();
    return registry;
}

/// \brief Map from group id to its index in the registry vector.
const std::array<std::size_t, static_cast<std::size_t>(RDKitGroupId::Count_)>&
rdkit_group_index_by_id() {
    static const auto index_by_id = [] {
        std::array<std::size_t, static_cast<std::size_t>(RDKitGroupId::Count_)> map{};
        const auto& registry = rdkit_group_registry();
        for (std::size_t index = 0u; index < registry.size(); ++index) {
            map[static_cast<std::size_t>(registry[index].id)] = index;
        }
        return map;
    }();
    return index_by_id;
}

}  // namespace

DescriptorSet MakeRDKitDescriptors(const OEChem::OEMolBase& mol) {
    ComputeContext ctx(mol);
    return MakeRDKitDescriptors(mol, ctx, ColumnRequest::All());
}

DescriptorSet MakeRDKitDescriptors(const OEChem::OEMolBase& mol,
                                   ComputeContext& ctx,
                                   const ColumnRequest& request) {
    const auto schema = RDKitDescriptorSchema();
    const auto& registry = rdkit_group_registry();
    const auto& index_by_id = rdkit_group_index_by_id();

    // 1. Seed the run-set with groups that have a wanted emitted column.
    std::vector<bool> in_run_set(registry.size(), false);
    std::vector<std::size_t> pending;
    for (std::size_t index = 0u; index < registry.size(); ++index) {
        const auto& group = registry[index];
        const bool wanted = std::any_of(
            group.emitted_columns.begin(), group.emitted_columns.end(),
            [&request](std::size_t column) { return request.Wants(column); });
        if (wanted) {
            in_run_set[index] = true;
            pending.push_back(index);
        }
    }

    // 2. Transitively close the run-set over dependency edges: a dependency runs
    //    (to feed its dependent's artifact) even when no column of it is wanted.
    while (!pending.empty()) {
        const auto index = pending.back();
        pending.pop_back();
        for (const auto dependency : registry[index].dependency_groups) {
            const auto dependency_index = index_by_id[static_cast<std::size_t>(dependency)];
            if (!in_run_set[dependency_index]) {
                in_run_set[dependency_index] = true;
                pending.push_back(dependency_index);
            }
        }
    }

    // 3. Order the run-set so each group runs after every group it depends on.
    std::vector<std::size_t> order;
    order.reserve(registry.size());
    std::vector<bool> emitted(registry.size(), false);
    std::function<void(std::size_t)> visit = [&](std::size_t index) {
        if (emitted[index] || !in_run_set[index]) {
            return;
        }
        emitted[index] = true;
        for (const auto dependency : registry[index].dependency_groups) {
            visit(index_by_id[static_cast<std::size_t>(dependency)]);
        }
        order.push_back(index);
    };
    for (std::size_t index = 0u; index < registry.size(); ++index) {
        visit(index);
    }

    // 4. Run each group: it reads ctx intermediates and upstream artifacts,
    //    populates its own artifact, and emits only request-wanted columns.
    DescriptorSetBuilder raw_builder(schema);
    RequestGatedBuilder builder(raw_builder, *schema, request);
    RDKitGroupArtifacts artifacts;
    for (const auto index : order) {
        registry[index].run(mol, ctx, artifacts, request, builder);
    }
    return raw_builder.Build();
}

std::shared_ptr<const DescriptorSchema> RDKitDescriptorSource::Schema() const {
    return RDKitDescriptorSchema();
}

DescriptorSet RDKitDescriptorSource::Compute(const OEChem::OEMolBase& mol) const {
    ComputeContext ctx(mol);
    return Compute(mol, ctx, ColumnRequest::All());
}

DescriptorSet RDKitDescriptorSource::Compute(const OEChem::OEMolBase& mol,
                                             ComputeContext& ctx,
                                             const ColumnRequest& request) const {
    return MakeRDKitDescriptors(mol, ctx, request);
}

} // namespace OEFP
