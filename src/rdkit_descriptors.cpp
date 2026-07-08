#include "oefp/rdkit_descriptors.h"

#include "oefp/descriptor_source.h"
#include "oefp/molecular_properties.h"
#include "oefp/mordred_intermediates.h"
#include "oefp/morgan.h"

#include <oesystem.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
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

/// \brief Whether an element has a defined RDKit default valence.
///
/// True exactly for the elements where RDKit's
/// ``PeriodicTable::getDefaultValence`` returns a non-negative value (main-group
/// elements with a fixed common valence). Transition metals and other
/// variable-valence elements return ``-1`` there. This predicate is the
/// load-bearing gate that selects the radical model: main-group elements use the
/// octet-shortfall formula, while undefined-valence elements use RDKit's
/// odd-electron parity rule. The table is derived directly from RDKit's
/// ``getDefaultValence`` sign across ``Z = 0..118``.
bool has_defined_default_valence(std::uint32_t atomic_number) {
    static constexpr std::array<bool, 119> defined{{
        false, true, true, true, true, true, true, true, true, true, true,
        true, true, true, true, true, true, true, true, true, true,
        false, false, false, false, false, false, false, false, false, false,
        true, true, true, true, true, true, true, true, false, false,
        false, false, false, false, false, false, false, false, true, true,
        true, true, true, true, true, true, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, true, true, true, true, true, true, true, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false, false, false, false,
        false, false, false, false, false, false, false,
    }};
    if (atomic_number >= defined.size()) {
        return false;
    }
    return defined[atomic_number];
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
    // This assumes implicit/suppressed-H input (the pipeline-wide convention,
    // shared with Mordred): each heavy atom's bonded hydrogens are added once via
    // GetTotalHCount. On explicit-H input it double-counts those hydrogens (a
    // known cross-cutting limitation deferred to a follow-up task); it is NOT
    // correct for explicit-H molecules.
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
    // RDKit: sum of GetNumRadicalElectrons over all atoms. RDKit assigns radicals
    // by two different rules depending on whether the element has a defined
    // default valence, so this mirrors both:
    //
    //   * Main-group elements (defined default valence): the open shell is the
    //     positive shortfall between a charge-shifted target valence and the
    //     realized valence. Formal charge SHIFTS the target rather than zeroing
    //     the radical. With the element's outer-electron count v, the target is
    //     v - charge for shells up to half full (v < 4) and 8 - v + charge
    //     otherwise (e.g. [O-]=1, [O+]=3, [NH3+]=1, [CH3]=1).
    //   * Variable-valence elements with NO defined default valence (transition
    //     metals, etc.): RDKit reports the odd-electron parity of the remaining
    //     free electrons, (v - charge - valence) mod 2 when positive, else 0
    //     (e.g. [Cu]=1, [Co]=1, [V]=1 but [Ti]=[Zn]=[Fe+2]=0). Without this gate
    //     the octet formula would invent spurious radicals for d-block metals.
    std::int64_t total = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        const auto outer_electrons = rdkit_outer_electrons(atomic_number);
        if (outer_electrons == 0) {
            continue;
        }
        const auto charge = static_cast<std::int64_t>(atom->GetFormalCharge());
        const auto valence = static_cast<std::int64_t>(atom->GetValence());
        if (has_defined_default_valence(atomic_number)) {
            const auto target_valence = outer_electrons < 4
                                            ? outer_electrons - charge
                                            : 8 - outer_electrons + charge;
            const auto shortfall = target_valence - valence;
            if (shortfall > 0) {
                total += shortfall;
            }
        } else {
            const auto free_electrons = outer_electrons - charge - valence;
            if (free_electrons > 0) {
                total += free_electrons % 2;
            }
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

bool is_spiro_atom(const OEChem::OEAtomBase& atom) {
    // A spiro atom joins two otherwise-separate rings at exactly one shared atom
    // (the rings share that atom and no bond). Such a center carries at least four
    // ring bonds, but a raw "four ring bonds" heuristic also matches propellane
    // bridge atoms, whose rings DO share a bond and are genuine bridgeheads, not
    // spiro. To distinguish them, delete this atom and walk the remaining ring
    // bonds: if its ring neighbors then fall into two or more disconnected ring
    // components, the rings met only here and it is a true spiro center; if they
    // stay connected (a shared bond bypasses this atom), it is a fused/bridged
    // junction, not spiro.
    if (!atom.IsInRing()) {
        return false;
    }
    std::vector<const OEChem::OEAtomBase*> ring_neighbors;
    for (OESystem::OEIter<OEChem::OEBondBase> bond = atom.GetBonds(); bond; ++bond) {
        if (bond->IsInRing()) {
            ring_neighbors.push_back(bond->GetNbr(&atom));
        }
    }
    if (ring_neighbors.size() < 4u) {
        return false;
    }

    const auto center_index = atom.GetIdx();
    // Flood-fill over ring bonds, never crossing this atom, to find the connected
    // ring component reachable from a starting neighbor.
    const auto ring_component = [center_index](const OEChem::OEAtomBase* start) {
        std::unordered_set<unsigned int> seen{start->GetIdx()};
        std::vector<const OEChem::OEAtomBase*> stack{start};
        while (!stack.empty()) {
            const auto* current = stack.back();
            stack.pop_back();
            for (OESystem::OEIter<OEChem::OEBondBase> bond = current->GetBonds(); bond;
                 ++bond) {
                if (!bond->IsInRing()) {
                    continue;
                }
                const auto* neighbor = bond->GetNbr(current);
                if (neighbor->GetIdx() == center_index) {
                    continue;
                }
                if (seen.insert(neighbor->GetIdx()).second) {
                    stack.push_back(neighbor);
                }
            }
        }
        return seen;
    };

    std::unordered_set<unsigned int> assigned;
    std::size_t components = 0u;
    for (const auto* neighbor : ring_neighbors) {
        if (assigned.count(neighbor->GetIdx()) != 0u) {
            continue;
        }
        const auto component = ring_component(neighbor);
        assigned.insert(component.begin(), component.end());
        ++components;
    }
    return components >= 2u;
}

std::int64_t count_bridgehead_atoms(const OEChem::OEMolBase& mol) {
    // RDKit's CalcNumBridgeheadAtoms counts only ring-fusion/bridge atoms: an atom
    // shared by two rings that also share a bond. OpenEye's OEIsBridgeHead is
    // broader on two axes, so it is narrowed to match RDKit:
    //   * it flags fused-ring aromatic atoms RDKit excludes (drop aromatic atoms,
    //     as Mordred's nBridgehead also does), and
    //   * it flags pure spiro centers, where two rings share exactly one atom and
    //     no bond; RDKit does not treat those as bridgeheads, so they are excluded
    //     via the precise is_spiro_atom test (which, unlike a raw ring-bond count,
    //     leaves genuine propellane/bridged junctions counted).
    OEChem::OEIsBridgeHead is_bridgehead(mol);
    std::int64_t bridgeheads = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (!atom->IsAromatic() && is_bridgehead(*atom) && !is_spiro_atom(*atom)) {
            ++bridgeheads;
        }
    }
    return bridgeheads;
}

std::int64_t count_spiro_atoms(const OEChem::OEMolBase& mol) {
    std::int64_t spiro = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (is_spiro_atom(*atom)) {
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

// ---------------------------------------------------------------------------
// rdkit:Connectivity family (Task 6): the 20 float connectivity/shape indices
// plus the Kappa values that feed the CountsWeights `Phi` column.
//
// Every descriptor here is computed from the shared heavy-atom graph
// (ctx.HeavyAtomGraph()), whose adjacency carries bond orders (aromatic bonds
// are 1.5) and whose atoms are the raw molecule's heavy atoms. RDKit's
// definitions were reproduced exactly against the oracle across the conformance
// panel (worst deviation ~1e-15). The subtle bits, each verified:
//   * ChiN path enumeration matches RDKit's findAllPathsOfLengthN (atom paths
//     with ring-closure handling and the "don't multiply the closing atom
//     twice" rule from RDKit github #463).
//   * HallKierAlpha needs RDKit's hybridization, which differs from OpenEye's on
//     a few conjugated O/N atoms, so RDKit's setConjugation + setHybridization
//     is reproduced from graph primitives rather than read from OpenEye.
//   * BertzCT is byte-identical to Mordred's compute_bertz_ct on the panel, so
//     it is reused additively; BalabanJ uses RDKit's bond-order-weighted
//     distances (different from Mordred's), so a family-local helper computes it.

/// \brief RDKit ``PeriodicTable::getDefaultValence`` for ``Z = 0..118``.
///
/// A non-negative common valence for main-group elements; ``-1`` for
/// variable-valence elements (transition metals, etc.). Used by the
/// conjugation and lone-pair models below, which branch on whether an element
/// has a defined default valence exactly as RDKit does.
int rdkit_default_valence(std::uint32_t atomic_number) {
    static constexpr std::array<int, 119> defaults{{
        -1, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 4, 3, 2, 1, 0, 1, 2,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3, 2, 3, 2, 1, 0, 1, 2,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, 2, 3, 2, 1, 0, 1, 2,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    }};
    if (atomic_number >= defaults.size()) {
        return -1;
    }
    return defaults[atomic_number];
}

/// \brief RDKit ``PeriodicTable::getRb0`` covalent radii for ``Z = 0..118``.
///
/// Only used by HallKierAlpha's fallback for elements outside its explicit
/// C/N/O/F/P/S/Cl/Br/I table (``alpha = Rb0(Z)/Rb0(C) - 1``), matching RDKit's
/// ``calcHallKierAlpha``.
double rdkit_covalent_radius(std::uint32_t atomic_number) {
    static constexpr std::array<double, 119> radii{{
        0.0, 0.33, 0.7, 1.23, 0.9, 0.82, 0.77, 0.7, 0.66, 0.611, 0.7, 1.54,
        1.36, 1.18, 0.937, 0.89, 1.04, 0.997, 1.74, 2.03, 1.74, 1.44, 1.32,
        1.22, 1.18, 1.17, 1.17, 1.16, 1.15, 1.17, 1.25, 1.26, 1.188, 1.2, 1.17,
        1.167, 1.91, 2.16, 1.91, 1.62, 1.45, 1.34, 1.3, 1.27, 1.25, 1.25, 1.28,
        1.34, 1.48, 1.44, 1.385, 1.4, 1.378, 1.387, 1.98, 2.35, 1.98, 1.69,
        1.83, 1.82, 1.81, 1.8, 1.8, 1.99, 1.79, 1.76, 1.75, 1.74, 1.73, 1.72,
        1.94, 1.72, 1.44, 1.34, 1.3, 1.28, 1.26, 1.27, 1.3, 1.34, 1.49, 1.48,
        1.48, 1.45, 1.46, 1.45, 2.4, 2.0, 1.9, 1.88, 1.79, 1.61, 1.58, 1.55,
        1.53, 1.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    }};
    if (atomic_number >= radii.size()) {
        return 0.0;
    }
    return radii[atomic_number];
}

/// \brief Per-heavy-atom primitives the connectivity descriptors read.
///
/// Precomputed once per heavy atom so the hybridization and delta calculations
/// share a single pass over the molecule. Field names mirror the RDKit
/// primitives they stand in for: OpenEye's ``GetDegree`` already counts implicit
/// hydrogens, so it maps to RDKit's ``getTotalDegree``; ``GetHvyDegree`` maps to
/// RDKit's ``getDegree``; ``GetValence`` (integer, hydrogens included) maps to
/// RDKit's total (explicit + implicit) valence.
struct ConnectivityAtomInfo {
    std::uint32_t atomic_number = 0;
    int total_degree = 0;       // heavy neighbors + hydrogens (RDKit totalDegree)
    int heavy_degree = 0;       // heavy neighbors only (RDKit degree)
    int total_hydrogens = 0;    // RDKit getTotalNumHs
    int valence = 0;            // integer valence incl. H (RDKit expl+impl valence)
    int formal_charge = 0;
    int radical_electrons = 0;  // RDKit getNumRadicalElectrons
};

/// \brief Number of unpaired (radical) electrons, mirroring RDKit exactly.
///
/// Identical model to :cpp:func:`count_radical_electrons`'s per-atom branch:
/// main-group elements use the charge-shifted octet shortfall, variable-valence
/// elements use the odd-electron parity. Duplicated here at atom granularity
/// because the hybridization and lone-pair models below need the per-atom value.
int rdkit_atom_radicals(std::uint32_t atomic_number, int formal_charge, int valence) {
    const auto outer = static_cast<int>(rdkit_outer_electrons(atomic_number));
    if (outer == 0) {
        return 0;
    }
    if (has_defined_default_valence(atomic_number)) {
        const auto target = outer < 4 ? outer - formal_charge : 8 - outer + formal_charge;
        const auto shortfall = target - valence;
        return shortfall > 0 ? shortfall : 0;
    }
    const auto free_electrons = outer - formal_charge - valence;
    return free_electrons > 0 ? free_electrons % 2 : 0;
}

/// \brief Reproduce RDKit's ``MolOps::countAtomElec`` for a graph atom.
///
/// The pi-donation electron count RDKit's conjugation candidate test consults.
/// ``heavy_valence`` is the summed heavy-bond order (RDKit ``getExplicitValence``
/// on the hydrogen-suppressed graph) used only to cap the result at 1 for
/// atoms bearing a triple-or-higher bond.
int rdkit_count_atom_elec(const ConnectivityAtomInfo& info, int heavy_valence) {
    const auto default_valence = rdkit_default_valence(info.atomic_number);
    if (default_valence <= 1) {
        return -1;
    }
    int degree = info.heavy_degree + info.total_hydrogens;
    if (degree > 3) {
        return -1;
    }
    const auto outer = static_cast<int>(rdkit_outer_electrons(info.atomic_number));
    int lone_pairs = outer - default_valence - info.formal_charge;
    if (lone_pairs < 0) {
        lone_pairs = 0;
    }
    int result = (default_valence - degree) + lone_pairs - info.radical_electrons;
    if (result > 1) {
        const auto unsaturations = heavy_valence - info.heavy_degree;
        if (unsaturations > 1) {
            result = 1;
        }
    }
    return result;
}

/// \brief RDKit's ``isAtomConjugCand``: whether an atom may carry conjugation.
bool rdkit_is_conjugation_candidate(const ConnectivityAtomInfo& info, int heavy_valence) {
    const auto outer = static_cast<int>(rdkit_outer_electrons(info.atomic_number));
    const bool row_ok = info.atomic_number <= 10u || (outer != 5 && outer != 6)
                        || (outer == 6 && info.total_degree < 2);
    return row_ok && rdkit_count_atom_elec(info, heavy_valence) > 0;
}

/// \brief RDKit's ``numBondsPlusLonePairs``: the orbital count that sets hybridization.
int rdkit_num_bonds_plus_lone_pairs(const ConnectivityAtomInfo& info) {
    const int degree = info.total_degree;
    if (info.atomic_number <= 1u) {
        return degree;
    }
    const auto outer = static_cast<int>(rdkit_outer_electrons(info.atomic_number));
    const int total_valence = info.valence;
    const int free_electrons = outer - (total_valence + info.formal_charge);
    if (total_valence + outer - info.formal_charge < 8) {
        const int lone_pairs = (free_electrons - info.radical_electrons) / 2;
        return degree + lone_pairs + info.radical_electrons;
    }
    const int lone_pairs = free_electrons / 2;
    return degree + lone_pairs;
}

/// \brief RDKit hybridization classes the Hall-Kier alpha table distinguishes.
enum class RDKitHybridization { S, SP, SP2, SP3, SP3D, SP3D2, Unspecified };

/// \brief Per-heavy-atom hybridization reproducing RDKit's
///     ``setConjugation`` + ``setHybridization``.
///
/// OpenEye's own hybridization perception disagrees with RDKit on a handful of
/// conjugated O/N atoms (e.g. the hydroxyl O of a carboxylic acid), which would
/// shift HallKierAlpha and every Kappa/Phi that depends on it. Reproducing
/// RDKit's algorithm from the heavy-atom graph removes that dependency and
/// matches the oracle exactly. Aromatic bonds seed conjugation; the orbital
/// count (bonds + lone pairs) picks S/SP/SP2/SP3/…; a 4-orbital atom drops to
/// SP2 only when it has a conjugated bond and total degree <= 3 (RDKit Issue276).
std::vector<RDKitHybridization> rdkit_hybridizations(
    const MordredHeavyAtomGraph& graph, const std::vector<ConnectivityAtomInfo>& atoms) {
    const auto atom_count = graph.atoms.size();

    // heavy_valence[i] = rounded sum of incident heavy-bond orders (aromatic
    // counts as 1.5, so it rounds toward the Kekule contribution RDKit uses).
    std::vector<int> heavy_valence(atom_count, 0);
    for (std::size_t i = 0u; i < atom_count; ++i) {
        double order_sum = 0.0;
        for (const auto& neighbor : graph.adjacency[i]) {
            order_sum += neighbor.bond_order;
        }
        heavy_valence[i] = static_cast<int>(std::lround(order_sum));
    }

    // Seed conjugation from aromatic bonds, then mark conjugated bond pairs
    // exactly as RDKit's markConjAtomBonds does.
    std::vector<std::vector<char>> conjugated(atom_count);
    for (std::size_t i = 0u; i < atom_count; ++i) {
        conjugated[i].assign(graph.adjacency[i].size(), 0);
        for (std::size_t k = 0u; k < graph.adjacency[i].size(); ++k) {
            if (graph.adjacency[i][k].bond_order == 1.5) {
                conjugated[i][k] = 1;
            }
        }
    }
    // Conjugation is a property of the shared bond, so mark both directed
    // adjacency slots. Marking only the originating side would leave a
    // neighbour (e.g. a carboxylate O) reading its own slot as unconjugated and
    // mis-hybridizing to SP3.
    const auto set_conjugated = [&](std::size_t from, std::size_t to) {
        for (std::size_t k = 0u; k < graph.adjacency[from].size(); ++k) {
            if (graph.adjacency[from][k].atom_index == to) {
                conjugated[from][k] = 1;
            }
        }
        for (std::size_t k = 0u; k < graph.adjacency[to].size(); ++k) {
            if (graph.adjacency[to][k].atom_index == from) {
                conjugated[to][k] = 1;
            }
        }
    };
    for (std::size_t i = 0u; i < atom_count; ++i) {
        if (!rdkit_is_conjugation_candidate(atoms[i], heavy_valence[i])) {
            continue;
        }
        const int substitutions = atoms[i].heavy_degree + atoms[i].total_hydrogens;
        if (substitutions < 2 || substitutions > 3) {
            continue;
        }
        for (std::size_t a = 0u; a < graph.adjacency[i].size(); ++a) {
            if (graph.adjacency[i][a].bond_order < 1.5) {
                continue;
            }
            for (std::size_t b = 0u; b < graph.adjacency[i].size(); ++b) {
                if (a == b) {
                    continue;
                }
                const auto other = graph.adjacency[i][b].atom_index;
                const int other_subs = atoms[other].heavy_degree + atoms[other].total_hydrogens;
                if (other_subs > 3) {
                    continue;
                }
                if (rdkit_is_conjugation_candidate(atoms[other], heavy_valence[other])) {
                    set_conjugated(i, graph.adjacency[i][a].atom_index);
                    set_conjugated(i, other);
                }
            }
        }
    }

    std::vector<RDKitHybridization> hybridizations(atom_count, RDKitHybridization::Unspecified);
    for (std::size_t i = 0u; i < atom_count; ++i) {
        const int orbitals = atoms[i].atomic_number < 89u
                                 ? rdkit_num_bonds_plus_lone_pairs(atoms[i])
                                 : atoms[i].total_degree;
        const bool has_conjugated_bond =
            std::any_of(conjugated[i].begin(), conjugated[i].end(), [](char c) { return c != 0; });
        switch (orbitals) {
            case 0:
            case 1:
                hybridizations[i] = RDKitHybridization::S;
                break;
            case 2:
                hybridizations[i] = RDKitHybridization::SP;
                break;
            case 3:
                hybridizations[i] = RDKitHybridization::SP2;
                break;
            case 4:
                hybridizations[i] = (atoms[i].total_degree > 3 || !has_conjugated_bond)
                                        ? RDKitHybridization::SP3
                                        : RDKitHybridization::SP2;
                break;
            case 5:
                hybridizations[i] = RDKitHybridization::SP3D;
                break;
            case 6:
                hybridizations[i] = RDKitHybridization::SP3D2;
                break;
            default:
                hybridizations[i] = RDKitHybridization::Unspecified;
                break;
        }
    }
    return hybridizations;
}

/// \brief Hall-Kier alpha contribution for one atom, matching RDKit's ``getAlpha``.
double rdkit_hall_kier_alpha_contribution(std::uint32_t atomic_number,
                                          RDKitHybridization hybridization) {
    const bool sp = hybridization == RDKitHybridization::SP;
    const bool sp2 = hybridization == RDKitHybridization::SP2;
    switch (atomic_number) {
        case 1u:
            return 0.0;
        case 6u:
            return sp ? -0.22 : (sp2 ? -0.13 : 0.0);
        case 7u:
            return sp ? -0.29 : (sp2 ? -0.20 : -0.04);
        case 8u:
            return sp2 ? -0.20 : -0.04;
        case 9u:
            return -0.07;
        case 15u:
            return sp2 ? 0.30 : 0.43;
        case 16u:
            return sp2 ? 0.22 : 0.35;
        case 17u:
            return 0.29;
        case 35u:
            return 0.48;
        case 53u:
            return 0.73;
        default: {
            const auto carbon_radius = rdkit_covalent_radius(6u);
            if (carbon_radius == 0.0) {
                return 0.0;
            }
            return rdkit_covalent_radius(atomic_number) / carbon_radius - 1.0;
        }
    }
}

// RDKit forms every Kier-Hall delta NUMERATOR ``Zv - h`` in UNSIGNED 32-bit
// integer arithmetic (both ``getNouterElecs`` and ``getTotalNumHs`` return
// ``unsigned int``). For a hypervalent hydride anion the numerator is negative
// (e.g. boron in [BH4-]: Zv=3, h=4 -> 3 - 4) and therefore WRAPS to
// ``2^32 - 1 = 4294967295``, so RDKit's oracle emits a finite tiny delta
// (``1/sqrt(4294967295) = 1.5258789e-05``) rather than a NaN. Reproducing that
// wrap exactly — not clamping or guard-returning-zero — is what matches the
// oracle bit-for-bit; for all non-negative numerators the unsigned cast is a
// no-op so ordinary atoms are unchanged.
std::uint32_t rdkit_wrapped_delta_numerator(std::int64_t outer_electrons, int hydrogens) {
    return static_cast<std::uint32_t>(outer_electrons) - static_cast<std::uint32_t>(hydrogens);
}

/// \brief Kier-Hall valence delta (``v`` variant): ``1/sqrt(delta_v)``.
///
/// Mirrors RDKit's ``hkDeltas``: ``delta_v = (Zv - h)`` for first-row atoms and
/// ``(Zv - h)/(Z - Zv - 1)`` from the second row up, where ``Zv`` is the
/// outer-shell electron count and ``h`` the attached-hydrogen count, with the
/// numerator formed in unsigned 32-bit (see ``rdkit_wrapped_delta_numerator``).
/// The denominator is likewise unsigned; ``Z - Zv - 1 == 0`` yields an infinite
/// delta whose reciprocal square root is 0, matching RDKit. Returns 0 for a zero
/// delta (the atom is then skipped from the reciprocal-square-root sum).
double rdkit_valence_delta(const ConnectivityAtomInfo& info) {
    if (info.atomic_number <= 1u) {
        return 0.0;
    }
    const auto outer = rdkit_outer_electrons(info.atomic_number);
    const auto numerator =
        static_cast<double>(rdkit_wrapped_delta_numerator(outer, info.total_hydrogens));
    double delta = 0.0;
    if (info.atomic_number <= 10u) {
        delta = numerator;
    } else {
        const auto denominator = static_cast<double>(
            static_cast<std::uint32_t>(info.atomic_number)
            - static_cast<std::uint32_t>(outer) - 1u);
        delta = numerator / denominator;
    }
    return delta != 0.0 ? 1.0 / std::sqrt(delta) : 0.0;
}

/// \brief Sigma-electron delta (``n`` variant): ``1/sqrt(Zv - h)``.
///
/// Mirrors RDKit's ``nVals``: the numerator ``Zv - h`` is formed in unsigned
/// 32-bit, so hypervalent hydride anions wrap to a finite tiny delta rather than
/// a NaN (see ``rdkit_wrapped_delta_numerator``).
double rdkit_sigma_delta(const ConnectivityAtomInfo& info) {
    const auto outer = rdkit_outer_electrons(info.atomic_number);
    const auto delta =
        static_cast<double>(rdkit_wrapped_delta_numerator(outer, info.total_hydrogens));
    return delta != 0.0 ? 1.0 / std::sqrt(delta) : 0.0;
}

/// \brief Enumerate RDKit's ``findAllPathsOfLengthN`` atom paths (bond length ``n``).
///
/// Reproduces RDKit's path finder: paths start at every atom and grow one
/// neighbor at a time, never revisiting an atom except that the final step may
/// close a ring (when the target length > 2 and the closing atom is not the
/// path's second-to-last). Paths are then de-duplicated by their bond set, so a
/// ring's many rotations/reflections collapse to distinct bond compositions —
/// matching RDKit's invariant. Returns atom-index paths of ``length + 1`` atoms.
std::vector<std::vector<std::size_t>> rdkit_atom_paths(
    const MordredHeavyAtomGraph& graph, std::size_t length) {
    const auto atom_count = graph.adjacency.size();
    std::vector<std::vector<std::size_t>> paths;
    paths.reserve(atom_count);
    for (std::size_t i = 0u; i < atom_count; ++i) {
        paths.push_back({i});
    }
    for (std::size_t step = 1u; step < length + 1u; ++step) {
        std::vector<std::vector<std::size_t>> next;
        for (const auto& path : paths) {
            const auto end = path.back();
            for (const auto& neighbor : graph.adjacency[end]) {
                const auto candidate = neighbor.atom_index;
                const bool already_in =
                    std::find(path.begin(), path.end(), candidate) != path.end();
                if (!already_in) {
                    auto extended = path;
                    extended.push_back(candidate);
                    next.push_back(std::move(extended));
                } else if (length + 1u > 2u && path.size() == length
                           && path.size() >= 2u && path[path.size() - 2u] != candidate) {
                    // Ring-closure step (RDKit github #463): permitted only on
                    // the last extension and never by doubling back.
                    auto extended = path;
                    extended.push_back(candidate);
                    next.push_back(std::move(extended));
                }
            }
        }
        paths = std::move(next);
    }

    // De-duplicate by bond set so ring rotations collapse, matching RDKit.
    std::vector<std::vector<std::size_t>> unique_paths;
    std::set<std::set<std::pair<std::size_t, std::size_t>>> seen;
    for (const auto& path : paths) {
        std::set<std::pair<std::size_t, std::size_t>> bonds;
        for (std::size_t k = 0u; k + 1u < path.size(); ++k) {
            const auto left = std::min(path[k], path[k + 1u]);
            const auto right = std::max(path[k], path[k + 1u]);
            bonds.emplace(left, right);
        }
        if (seen.insert(bonds).second) {
            unique_paths.push_back(path);
        }
    }
    return unique_paths;
}

/// \brief ChiN connectivity index for a per-atom delta table.
///
/// Sums, over every RDKit path of ``order`` bonds, the product of the path
/// atoms' delta values, omitting the closing atom of a ring path (RDKit's
/// ``p[n] != p[0]`` guard). ``order == 0`` is the atom sum; ``order == 1`` the
/// bond sum. Covers both the ``n`` (sigma) and ``v`` (valence) variants via the
/// supplied ``deltas`` vector (already ``1/sqrt`` transformed per atom).
double rdkit_chi_order(const MordredHeavyAtomGraph& graph, const std::vector<double>& deltas,
                       std::size_t order) {
    if (order == 0u) {
        double total = 0.0;
        for (const auto delta : deltas) {
            total += delta;
        }
        return total;
    }
    if (order == 1u) {
        double total = 0.0;
        for (const auto& [begin, end] : graph.bonds) {
            total += deltas[begin] * deltas[end];
        }
        return total;
    }
    double total = 0.0;
    for (const auto& path : rdkit_atom_paths(graph, order)) {
        double product = 1.0;
        for (std::size_t i = 0u; i < order; ++i) {
            product *= deltas[path[i]];
        }
        if (path[order] != path[0]) {
            product *= deltas[path[order]];
        }
        total += product;
    }
    return total;
}

/// \brief Chi0/Chi1 using the simple heavy-atom degree (not a valence delta).
double rdkit_chi_simple(const MordredHeavyAtomGraph& graph, bool order_one) {
    if (!order_one) {
        double total = 0.0;
        for (const auto& neighbors : graph.adjacency) {
            const auto degree = neighbors.size();
            if (degree > 0u) {
                total += 1.0 / std::sqrt(static_cast<double>(degree));
            }
        }
        return total;
    }
    double total = 0.0;
    for (const auto& [begin, end] : graph.bonds) {
        const auto product = graph.adjacency[begin].size() * graph.adjacency[end].size();
        if (product > 0u) {
            total += 1.0 / std::sqrt(static_cast<double>(product));
        }
    }
    return total;
}

/// \brief Kier Kappa1/2/3 shape indices with the Hall-Kier alpha correction.
///
/// Uses path counts ``P1`` (bonds), ``P2``, ``P3`` (RDKit ``findAllPathsOfLengthN``
/// with 2 and 3 bonds), the heavy-atom count ``A`` and ``alpha`` exactly as
/// RDKit's ``kappa*Helper``. Kappa3 branches on the parity of ``A``. Each guards
/// a zero denominator (returning 0), matching RDKit.
struct RDKitKappaValues {
    double kappa1 = 0.0;
    double kappa2 = 0.0;
    double kappa3 = 0.0;
};

RDKitKappaValues rdkit_kappa_values(const MordredHeavyAtomGraph& graph, double alpha) {
    RDKitKappaValues values;
    const auto atom_count = static_cast<double>(graph.atoms.size());
    const auto p1 = static_cast<double>(graph.bonds.size());
    const auto p2 = static_cast<double>(rdkit_atom_paths(graph, 2u).size());
    const auto p3 = static_cast<double>(rdkit_atom_paths(graph, 3u).size());

    const auto denom1 = p1 + alpha;
    if (denom1 != 0.0) {
        values.kappa1 = (atom_count + alpha) * (atom_count + alpha - 1.0)
                        * (atom_count + alpha - 1.0) / (denom1 * denom1);
    }
    const auto denom2 = (p2 + alpha) * (p2 + alpha);
    if (denom2 != 0.0) {
        values.kappa2 = (atom_count + alpha - 1.0) * (atom_count + alpha - 2.0)
                        * (atom_count + alpha - 2.0) / denom2;
    }
    const auto denom3 = (p3 + alpha) * (p3 + alpha);
    if (denom3 != 0.0) {
        const auto odd = static_cast<long>(graph.atoms.size()) % 2 == 1;
        const auto leading = odd ? (atom_count + alpha - 1.0) : (atom_count + alpha - 2.0);
        values.kappa3 = leading * (atom_count + alpha - 3.0) * (atom_count + alpha - 3.0) / denom3;
    }
    return values;
}

/// \brief Shannon entropy (bits) of a value vector, matching RDKit's ``InfoEntropy``.
double rdkit_info_entropy(const std::vector<double>& values) {
    double total = 0.0;
    for (const auto value : values) {
        total += value;
    }
    if (total == 0.0) {
        return 0.0;
    }
    double entropy = 0.0;
    for (const auto value : values) {
        const auto probability = value / total;
        if (probability > 0.0) {
            entropy += -probability * std::log2(probability);
        }
    }
    return entropy;
}

/// \brief Bond-order-weighted all-pairs shortest-path distance matrix.
///
/// Edge weight ``1/bond_order`` (aromatic bonds weigh 1/1.5), matching RDKit's
/// ``GetDistanceMatrix(useBO=1)`` that both BalabanJ and BertzCT's symmetry
/// classes rely on. Disconnected pairs stay at the infinity sentinel.
std::vector<std::vector<double>> rdkit_bond_order_distances(
    const MordredHeavyAtomGraph& graph) {
    const auto atom_count = graph.adjacency.size();
    constexpr auto kInfinity = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> distances(atom_count,
                                               std::vector<double>(atom_count, kInfinity));
    for (std::size_t i = 0u; i < atom_count; ++i) {
        distances[i][i] = 0.0;
        for (const auto& neighbor : graph.adjacency[i]) {
            const auto weight = 1.0 / neighbor.bond_order;
            distances[i][neighbor.atom_index] =
                std::min(distances[i][neighbor.atom_index], weight);
        }
    }
    for (std::size_t via = 0u; via < atom_count; ++via) {
        for (std::size_t begin = 0u; begin < atom_count; ++begin) {
            for (std::size_t end = 0u; end < atom_count; ++end) {
                const auto through = distances[begin][via] + distances[via][end];
                if (through < distances[begin][end]) {
                    distances[begin][end] = through;
                }
            }
        }
    }
    return distances;
}

/// \brief RDKit's Balaban J index.
///
/// Follows RDKit's ``BalabanJ``: distance row-sums come from the
/// bond-order-weighted distance matrix; ``J = q/(mu+1) * sum(1/sqrt(s_i*s_j))``
/// over bonds, with ``mu = q - n + 1``. Returns 0 for the degenerate ``mu+1==0``.
double rdkit_balaban_j(const MordredHeavyAtomGraph& graph) {
    const auto atom_count = graph.atoms.size();
    if (atom_count == 0u) {
        return 0.0;
    }
    const auto distances = rdkit_bond_order_distances(graph);
    std::vector<double> row_sums(atom_count, 0.0);
    for (std::size_t i = 0u; i < atom_count; ++i) {
        double sum = 0.0;
        for (const auto value : distances[i]) {
            sum += value;
        }
        row_sums[i] = sum;
    }
    double edge_sum = 0.0;
    for (const auto& [begin, end] : graph.bonds) {
        edge_sum += 1.0 / std::sqrt(row_sums[begin] * row_sums[end]);
    }
    const auto bond_count = static_cast<double>(graph.bonds.size());
    const auto mu = static_cast<long>(graph.bonds.size())
                    - static_cast<long>(atom_count) + 1;
    if (mu + 1 == 0) {
        return 0.0;
    }
    return bond_count / static_cast<double>(mu + 1) * edge_sum;
}

/// \brief RDKit's BertzCT graph-complexity index.
///
/// Reproduces RDKit's ``BertzCT``: symmetry classes come from the sorted rows
/// of the bond-order distance matrix (formatted to 4 decimals, capped at the
/// 100-neighbour cutoff), then two information terms are summed — a
/// connection-complexity term over hinge/neighbour class triples (and
/// multiple-bond pairs) and an atom-type term. This matches Mordred's
/// ``compute_bertz_ct`` byte-for-byte on the conformance panel (both track
/// RDKit's ``forceDMat=1`` aromatic-bond-order convention); it is duplicated
/// here rather than shared so the RDKit source stays self-contained and no
/// Mordred code path is disturbed.
double rdkit_bertz_ct(const MordredHeavyAtomGraph& graph) {
    const auto atom_count = graph.atoms.size();
    if (atom_count < 2u) {
        return 0.0;
    }

    // Neighbours sorted by atom index give a deterministic connection order.
    auto sorted_adjacency = graph.adjacency;
    for (auto& neighbors : sorted_adjacency) {
        std::sort(neighbors.begin(), neighbors.end(),
                  [](const PathCountNeighbor& left, const PathCountNeighbor& right) {
                      return left.atom_index < right.atom_index;
                  });
    }

    // Symmetry classes: distinct sorted distance vectors (4-decimal keys, capped
    // at the 100-nearest-neighbour cutoff), matching RDKit's _AssignSymmetryClasses.
    const auto distances = rdkit_bond_order_distances(graph);
    std::vector<std::vector<std::string>> keys_seen;
    std::vector<std::uint32_t> symmetry_classes;
    symmetry_classes.reserve(atom_count);
    for (const auto& row : distances) {
        auto sorted_row = row;
        std::sort(sorted_row.begin(), sorted_row.end());
        if (sorted_row.size() > 100u) {
            sorted_row.resize(100u);
        }
        std::vector<std::string> key;
        key.reserve(sorted_row.size());
        for (const auto distance : sorted_row) {
            std::ostringstream stream;
            stream << std::fixed << std::setprecision(4) << distance;
            key.push_back(stream.str());
        }
        const auto found = std::find(keys_seen.begin(), keys_seen.end(), key);
        if (found == keys_seen.end()) {
            keys_seen.push_back(std::move(key));
            symmetry_classes.push_back(static_cast<std::uint32_t>(keys_seen.size()));
        } else {
            symmetry_classes.push_back(
                static_cast<std::uint32_t>(std::distance(keys_seen.begin(), found) + 1));
        }
    }

    std::map<unsigned int, double> atom_type_counts;
    std::map<std::vector<std::uint32_t>, double> connection_counts;
    for (std::size_t i = 0u; i < atom_count; ++i) {
        atom_type_counts[graph.atoms[i]->GetAtomicNum()] += 1.0;
        const auto hinge_class = symmetry_classes[i];
        const auto& neighbors = sorted_adjacency[i];
        for (std::size_t left = 0u; left < neighbors.size(); ++left) {
            const auto left_index = neighbors[left].atom_index;
            const auto left_class = symmetry_classes[left_index];
            const auto left_order = neighbors[left].bond_order;
            if (left_order > 1.0 && left_index > i) {
                const auto lower = std::min(hinge_class, left_class);
                const auto upper = std::max(hinge_class, left_class);
                connection_counts[{lower, upper}] += left_order * (left_order - 1.0) / 2.0;
            }
            for (std::size_t right = left + 1u; right < neighbors.size(); ++right) {
                const auto right_index = neighbors[right].atom_index;
                const auto right_class = symmetry_classes[right_index];
                const auto lower = std::min(left_class, right_class);
                const auto upper = std::max(left_class, right_class);
                connection_counts[{lower, hinge_class, upper}] +=
                    left_order * neighbors[right].bond_order;
            }
        }
    }
    if (connection_counts.empty()) {
        connection_counts[{0u}] = 1.0;
    }

    std::vector<double> connection_values;
    connection_values.reserve(connection_counts.size());
    double total_connections = 0.0;
    for (const auto& entry : connection_counts) {
        connection_values.push_back(entry.second);
        total_connections += entry.second;
    }
    std::vector<double> atom_type_values;
    atom_type_values.reserve(atom_type_counts.size());
    for (const auto& entry : atom_type_counts) {
        atom_type_values.push_back(entry.second);
    }

    const auto connection_ie =
        total_connections * (rdkit_info_entropy(connection_values) + std::log2(total_connections));
    const auto atom_type_ie =
        static_cast<double>(atom_count) * rdkit_info_entropy(atom_type_values);
    return connection_ie + atom_type_ie;
}

/// \brief Le Verrier-Faddeev-Frame characteristic polynomial of the 0/1
///     adjacency matrix, used by Ipc.
///
/// Reproduces RDKit's ``Graphs.CharacteristicPolynomial`` on the
/// hydrogen-suppressed adjacency matrix. Returns the ``n + 1`` coefficients with
/// the sign convention RDKit applies (``res[1:] *= -1``).
std::vector<double> rdkit_characteristic_polynomial(const MordredHeavyAtomGraph& graph) {
    const auto n = graph.adjacency.size();
    std::vector<std::vector<double>> adjacency(n, std::vector<double>(n, 0.0));
    for (std::size_t i = 0u; i < n; ++i) {
        for (const auto& neighbor : graph.adjacency[i]) {
            adjacency[i][neighbor.atom_index] = 1.0;
        }
    }
    std::vector<double> coefficients(n + 1u, 0.0);
    coefficients[0] = 1.0;
    auto current = adjacency;  // A^1
    for (std::size_t step = 1u; step <= n; ++step) {
        double trace = 0.0;
        for (std::size_t i = 0u; i < n; ++i) {
            trace += current[i][i];
        }
        const auto coefficient = trace / static_cast<double>(step);
        coefficients[step] = coefficient;
        // Bn = current - coefficient * I
        auto bn = current;
        for (std::size_t i = 0u; i < n; ++i) {
            bn[i][i] -= coefficient;
        }
        // current = adjacency * Bn
        std::vector<std::vector<double>> product(n, std::vector<double>(n, 0.0));
        for (std::size_t i = 0u; i < n; ++i) {
            for (std::size_t k = 0u; k < n; ++k) {
                const auto a = adjacency[i][k];
                if (a == 0.0) {
                    continue;
                }
                for (std::size_t j = 0u; j < n; ++j) {
                    product[i][j] += a * bn[k][j];
                }
            }
        }
        current = std::move(product);
    }
    for (std::size_t step = 1u; step <= n; ++step) {
        coefficients[step] = -coefficients[step];
    }
    return coefficients;
}

/// \brief Ipc / AvgIpc information content of the characteristic polynomial.
///
/// RDKit's ``Ipc``: ``sum(|coeff|) * H(|coeff|)`` (``avg == false``) or the
/// entropy ``H(|coeff|)`` alone (``avg == true``). The product form grows large
/// (hundreds of thousands on the panel) but stays well within ``double`` range;
/// the entropy is computed directly in bits so no intermediate overflows.
double rdkit_ipc(const MordredHeavyAtomGraph& graph, bool averaged) {
    if (graph.adjacency.empty()) {
        return 0.0;
    }
    const auto coefficients = rdkit_characteristic_polynomial(graph);
    std::vector<double> magnitudes;
    magnitudes.reserve(coefficients.size());
    for (const auto coefficient : coefficients) {
        magnitudes.push_back(std::abs(coefficient));
    }
    const auto entropy = rdkit_info_entropy(magnitudes);
    if (averaged) {
        return entropy;
    }
    double total = 0.0;
    for (const auto magnitude : magnitudes) {
        total += magnitude;
    }
    return total * entropy;
}

/// \brief Gather the per-heavy-atom primitives the Connectivity family needs.
std::vector<ConnectivityAtomInfo> rdkit_connectivity_atom_info(
    const MordredHeavyAtomGraph& graph) {
    std::vector<ConnectivityAtomInfo> atoms;
    atoms.reserve(graph.atoms.size());
    for (std::size_t i = 0u; i < graph.atoms.size(); ++i) {
        const auto* atom = graph.atoms[i];
        ConnectivityAtomInfo info;
        info.atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        info.total_degree = static_cast<int>(atom->GetDegree());
        info.heavy_degree = static_cast<int>(graph.adjacency[i].size());
        info.total_hydrogens = static_cast<int>(atom->GetTotalHCount());
        info.valence = static_cast<int>(atom->GetValence());
        info.formal_charge = static_cast<int>(atom->GetFormalCharge());
        info.radical_electrons =
            rdkit_atom_radicals(info.atomic_number, info.formal_charge, info.valence);
        atoms.push_back(info);
    }
    return atoms;
}

/// \brief RDKit's Ertl topological polar surface area (N/O only), rounded.
///
/// Reproduces ``rdkit.Chem.Descriptors.TPSA``'s default model — the Ertl
/// fragment contributions for nitrogen and oxygen, EXCLUDING sulfur and
/// phosphorus (RDKit's default ``includeSandP=False``). It is ported natively
/// from RDKit's fragment table rather than read from OpenEye's ``OEGet2dPSA``
/// because the two disagree on a few edge structures (an alkyl azide such as
/// CCN=[N+]=[N-], and atomic oxygen [O]): OpenEye's polar-surface model assigns
/// those nitrogen/oxygen environments differently, so no float tolerance can
/// bridge the gap. The native port matches the oracle exactly across the
/// conformance panel. Each atom's contribution is selected from its neighbor
/// bond-order profile (single/double/triple/aromatic counts), attached-hydrogen
/// count, formal charge, and three-membered-ring membership; unmatched
/// nitrogen/oxygen fall back to RDKit's degree-and-hydrogen formula. The total is
/// rounded with the shared two-decimal PSA rounding so it matches the oracle's
/// stored precision.
double rdkit_tpsa(const OEChem::OEMolBase& mol) {
    double total = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = atom->GetAtomicNum();
        if (atomic_number != 7 && atomic_number != 8) {
            continue;
        }
        // GetTotalHCount() already returns implicit PLUS bonded explicit
        // hydrogens, which is exactly the total-hydrogen count RDKit's Ertl TPSA
        // keys on. Explicit-H neighbors must NOT be re-added below (doing so
        // double-counts them and inflates TPSA for explicit-H N/O input); the
        // neighbor loop only skips them so they never inflate heavy_neighbors.
        const int hydrogens = static_cast<int>(atom->GetTotalHCount());
        const int charge = atom->GetFormalCharge();
        const bool in_three_ring = OEChem::OEAtomIsInRingSize(*atom, 3u);
        int heavy_neighbors = 0;
        int singles = 0;
        int doubles = 0;
        int triples = 0;
        int aromatics = 0;
        for (OESystem::OEIter<OEChem::OEBondBase> bond = atom->GetBonds(); bond; ++bond) {
            const auto* other = bond->GetNbr(&*atom);
            if (other != nullptr && other->GetAtomicNum() == 1) {
                continue;  // already in GetTotalHCount(); exclude from heavy_neighbors
            }
            ++heavy_neighbors;
            if (bond->IsAromatic()) {
                ++aromatics;
            } else {
                switch (bond->GetOrder()) {
                    case 1u: ++singles; break;
                    case 2u: ++doubles; break;
                    case 3u: ++triples; break;
                    default: break;
                }
            }
        }

        double contribution = -1.0;
        if (atomic_number == 7) {
            if (heavy_neighbors == 1) {
                if (hydrogens == 0 && triples == 1 && charge == 0) contribution = 23.79;
                else if (hydrogens == 1 && doubles == 1 && charge == 0) contribution = 23.85;
                else if (hydrogens == 2 && singles == 1 && charge == 0) contribution = 26.02;
                else if (hydrogens == 2 && doubles == 1 && charge == 1) contribution = 25.59;
                else if (hydrogens == 3 && singles == 1 && charge == 1) contribution = 27.64;
            } else if (heavy_neighbors == 2) {
                if (hydrogens == 0 && singles == 1 && doubles == 1 && charge == 0) contribution = 12.36;
                else if (hydrogens == 0 && triples == 1 && doubles == 1 && charge == 0) contribution = 13.60;
                else if (hydrogens == 1 && singles == 2 && charge == 0) contribution = in_three_ring ? 21.94 : 12.03;
                else if (hydrogens == 0 && triples == 1 && singles == 1 && charge == 1) contribution = 4.36;
                else if (hydrogens == 1 && doubles == 1 && singles == 1 && charge == 1) contribution = 13.97;
                else if (hydrogens == 2 && singles == 2 && charge == 1) contribution = 16.61;
                else if (hydrogens == 0 && aromatics == 2 && charge == 0) contribution = 12.89;
                else if (hydrogens == 1 && aromatics == 2 && charge == 0) contribution = 15.79;
                else if (hydrogens == 1 && aromatics == 2 && charge == 1) contribution = 14.14;
            } else if (heavy_neighbors == 3) {
                if (hydrogens == 0 && singles == 3 && charge == 0) contribution = in_three_ring ? 3.01 : 3.24;
                else if (hydrogens == 0 && singles == 1 && doubles == 2 && charge == 0) contribution = 11.68;
                else if (hydrogens == 0 && singles == 2 && doubles == 1 && charge == 1) contribution = 3.01;
                else if (hydrogens == 1 && singles == 3 && charge == 1) contribution = 4.44;
                else if (hydrogens == 0 && aromatics == 3 && charge == 0) contribution = 4.41;
                else if (hydrogens == 0 && singles == 1 && aromatics == 2 && charge == 0) contribution = 4.93;
                else if (hydrogens == 0 && doubles == 1 && aromatics == 2 && charge == 0) contribution = 8.39;
                else if (hydrogens == 0 && aromatics == 3 && charge == 1) contribution = 4.10;
                else if (hydrogens == 0 && singles == 1 && aromatics == 2 && charge == 1) contribution = 3.88;
            } else if (heavy_neighbors == 4) {
                if (hydrogens == 0 && singles == 4 && charge == 1) contribution = 0.00;
            }
            if (contribution < 0.0) {
                contribution = 30.5 - heavy_neighbors * 8.2 + hydrogens * 1.5;
                if (contribution < 0.0) contribution = 0.0;
            }
        } else {  // oxygen
            if (heavy_neighbors == 1) {
                if (hydrogens == 0 && doubles == 1 && charge == 0) contribution = 17.07;
                else if (hydrogens == 1 && singles == 1 && charge == 0) contribution = 20.23;
                else if (hydrogens == 0 && singles == 1 && charge == -1) contribution = 23.06;
            } else if (heavy_neighbors == 2) {
                if (hydrogens == 0 && singles == 2 && charge == 0) contribution = in_three_ring ? 12.53 : 9.23;
                else if (hydrogens == 0 && aromatics == 2 && charge == 0) contribution = 13.14;
            }
            if (contribution < 0.0) {
                contribution = 28.5 - heavy_neighbors * 8.6 + hydrogens * 1.5;
                if (contribution < 0.0) contribution = 0.0;
            }
        }
        total += contribution;
    }
    return RoundTopologicalPsa(total);
}

// RDKit-internal group-to-group intermediates that are NOT molecule-level
// shareable (those live on ComputeContext instead). Per spec §4.3, the per-atom
// EState / LabuteASA vectors and BCUT2D eigenvalues are ComputeContext
// accessors, NOT fields here. Kept for parity with Mordred's
// MordredGroupArtifacts and to carry the Connectivity group's Kappa values to
// the `Phi` column (Task 6).
struct RDKitGroupArtifacts {
    // Populated by the Connectivity group (Task 6) for the CountsWeights `Phi`
    // column; extended only for intermediates that are truly not context-level.
    //
    // Kappa1/Kappa2 are Kier shape indices the Connectivity group already
    // computes. `Phi` (a CountsWeights column) is Kappa1*Kappa2/heavy-count, so
    // the group stashes them here whenever it runs. The group populates them
    // request-independently (even if the Kappa* columns themselves are not
    // wanted), so `Phi` still resolves correctly under column pruning when only
    // `Phi` is requested and Connectivity runs solely as a dependency.
    double kappa1 = 0.0;
    double kappa2 = 0.0;
    bool kappas_ready = false;
};

enum class RDKitGroupId {
    CountsWeights,
    RingCounts,
    Connectivity,
    Crippen,
    SurfacePolarity,
    EState,
    VSA,
    Count_,
};

/// \brief The 11 RingCounts descriptor values for one molecule.
///
/// Each field is an SSSR-ring-level classification total. ``ring_count`` is the
/// symmetrized SSSR ring count; the aromatic/aliphatic split is exhaustive
/// (``aromatic + aliphatic == ring_count``), and ``saturated`` is a subset of
/// ``aliphatic``. Each of the three families is further split into carbocycle
/// (all ring atoms carbon) and heterocycle (any ring atom a heteroatom).
struct RDKitRingCounts {
    std::int64_t ring_count = 0;
    std::int64_t aromatic = 0;
    std::int64_t aliphatic = 0;
    std::int64_t saturated = 0;
    std::int64_t aromatic_carbocycle = 0;
    std::int64_t aromatic_heterocycle = 0;
    std::int64_t aliphatic_carbocycle = 0;
    std::int64_t aliphatic_heterocycle = 0;
    std::int64_t saturated_carbocycle = 0;
    std::int64_t saturated_heterocycle = 0;
    std::int64_t heterocycle = 0;
};

/// \brief Classify the symmetrized SSSR rings into RDKit's 11 ring counts.
///
/// Mirrors RDKit's ring-count definitions exactly (verified against the oracle
/// on the conformance panel). A ring is classified from its own atoms and its
/// in-ring bonds only:
///   * aromatic — every in-ring bond is aromatic; otherwise aliphatic. RDKit's
///     ``NumAliphaticRings`` counts every non-fully-aromatic ring, so the
///     aromatic/aliphatic split is exhaustive and sums to ``RingCount``.
///   * saturated — no in-ring bond is aromatic and none is a double or triple
///     bond. An exocyclic multiple bond (e.g. a cyclohexanone C=O) does NOT
///     unsaturate the ring, so only in-ring bonds are inspected.
///   * heterocycle — at least one ring atom is not carbon; carbocycle otherwise.
/// The ring set comes from :cpp:func:`compute_symmetrized_sssr_rings`, whose
/// ``RingCount`` equals RDKit's symmetrized count on fused and caged systems.
RDKitRingCounts classify_ring_counts(const OEChem::OEMolBase& mol) {
    RDKitRingCounts counts;
    const auto rings = compute_symmetrized_sssr_rings(mol);
    counts.ring_count = static_cast<std::int64_t>(rings.size());
    for (const auto& ring : rings) {
        const auto ring_size = ring.size();
        bool all_bonds_aromatic = true;
        bool any_multiple_or_aromatic_bond = false;
        bool has_heteroatom = false;
        for (std::size_t i = 0u; i < ring_size; ++i) {
            const auto* atom = ring[i];
            if (atom->GetAtomicNum() != 6) {
                has_heteroatom = true;
            }
            const auto* next = ring[(i + 1u) % ring_size];
            const OEChem::OEBondBase* bond = atom->GetBond(next);
            if (bond == nullptr) {
                continue;
            }
            if (bond->IsAromatic()) {
                any_multiple_or_aromatic_bond = true;
            } else {
                all_bonds_aromatic = false;
                if (bond->GetOrder() > 1u) {
                    any_multiple_or_aromatic_bond = true;
                }
            }
        }
        const bool carbocycle = !has_heteroatom;
        const bool saturated = !any_multiple_or_aromatic_bond;

        if (has_heteroatom) {
            ++counts.heterocycle;
        }
        if (all_bonds_aromatic) {
            ++counts.aromatic;
            if (carbocycle) {
                ++counts.aromatic_carbocycle;
            } else {
                ++counts.aromatic_heterocycle;
            }
        } else {
            ++counts.aliphatic;
            if (carbocycle) {
                ++counts.aliphatic_carbocycle;
            } else {
                ++counts.aliphatic_heterocycle;
            }
        }
        if (saturated) {
            ++counts.saturated;
            if (carbocycle) {
                ++counts.saturated_carbocycle;
            } else {
                ++counts.saturated_heterocycle;
            }
        }
    }
    return counts;
}

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

/// \brief Bin one per-atom vector by another and accumulate a weight per bin.
///
/// Reproduces the RDKit MOE-style VSA binning kernel shared by all five VSA
/// sub-families: for each atom, find its bin from ``bin_keys[a]`` via
/// ``bisect_right`` (C++ ``std::upper_bound``) over ``bounds`` and add
/// ``weights[a]`` into that bin. ``N`` bin bounds define ``N + 1`` bins (the open
/// tail bin included), so the result is always fully sized and the caller decides
/// which bins the schema exposes. The two input vectors are the SHARED per-atom
/// context vectors, already aligned per heavy atom in the same order.
///
/// The kernel is deliberately unguarded on value finiteness, matching RDKit
/// exactly: RDKit bins every atom regardless of a non-finite key or weight. A
/// non-finite key (for example the NaN Gasteiger charge RDKit assigns to an atom
/// it has no PEOE parameters for) makes ``key < bound`` always false, so
/// ``upper_bound`` returns the tail bin — the identical bin ``bisect_right`` picks
/// for NaN in RDKit. Returning ``std::nullopt`` is reserved for a genuine
/// alignment fault (the two per-atom vectors disagreeing in length), which would
/// indicate an upstream context bug rather than a degenerate-but-valid molecule.
///
/// \param bin_keys Per-atom values selecting each atom's bin.
/// \param weights Per-atom values accumulated into the selected bin.
/// \param bounds Ascending bin upper bounds (``bisect_right`` cut points).
/// \returns Per-bin accumulated weights, or ``std::nullopt`` on a length mismatch.
template <std::size_t N>
std::optional<std::array<double, N + 1>> rdkit_vsa_bin_accumulate(
    const std::vector<double>& bin_keys,
    const std::vector<double>& weights,
    const std::array<double, N>& bounds) {
    if (bin_keys.size() != weights.size()) {
        return std::nullopt;
    }
    std::array<double, N + 1> result{};
    for (std::size_t a = 0u; a < bin_keys.size(); ++a) {
        const auto bin = static_cast<std::size_t>(
            std::upper_bound(bounds.begin(), bounds.end(), bin_keys[a]) - bounds.begin());
        result[bin] += weights[a];
    }
    return result;
}

/// \brief Emit the schema bins of one VSA sub-family, skipping the excluded bin.
///
/// The bin vector is fully sized (``1..bin_count``) but three bins are structural
/// zeros excluded from the 214-column schema (``SlogP_VSA9``, ``SMR_VSA8``,
/// ``EState_VSA11``); calling ``builder.Set`` for such a name would look up a
/// non-existent schema index. ``excluded_bin`` (1-based, ``0`` for none) is
/// therefore skipped so only the columns that exist are emitted.
template <std::size_t BinCount>
void rdkit_emit_vsa_family(
    RequestGatedBuilder& builder,
    const char* prefix,
    const std::array<double, BinCount>& values,
    int excluded_bin) {
    for (int bin = 1; bin <= static_cast<int>(BinCount); ++bin) {
        if (bin == excluded_bin) {
            continue;
        }
        set_float(builder, prefix + std::to_string(bin),
                  values[static_cast<std::size_t>(bin - 1)]);
    }
}

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

        // Group: rdkit:CountsWeights — the 21 DEPENDENCY-FREE descriptors plus
        // `Phi`. `Phi` needs the Connectivity group's Kappa values, so this
        // group declares a dependency on Connectivity (the first real
        // cross-group dependency in the RDKit registry) and reads the Kappa
        // artifact. `SPS` is a member but is deferred and left MISSING (see the
        // deferral note at the SPS emission site below), so it is not listed in
        // emitted_columns and the subtractive pruning check treats it as an
        // unrequested/missing column.
        groups.push_back(RDKitGroup{
            RDKitGroupId::CountsWeights,
            rdkit_column_indices(
                s,
                {"MolWt", "HeavyAtomMolWt", "ExactMolWt", "NumValenceElectrons",
                 "NumRadicalElectrons", "FpDensityMorgan1", "FpDensityMorgan2",
                 "FpDensityMorgan3", "FractionCSP3", "HeavyAtomCount", "NHOHCount",
                 "NOCount", "NumAmideBonds", "NumAtomStereoCenters", "NumBridgeheadAtoms",
                 "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
                 "NumSpiroAtoms", "NumUnspecifiedAtomStereoCenters", "Phi"}),
            {RDKitGroupId::Connectivity},  // Phi reads the Connectivity Kappa artifact
            [](const OEChem::OEMolBase&, ComputeContext& ctx,
               RDKitGroupArtifacts& artifacts, const ColumnRequest&,
               RequestGatedBuilder& builder) {
                // RDKit's oracle treats a stereo bracket hydrogen (e.g. the H in
                // C[C@H](N)C(=O)O) as implicit, but OpenEye keeps it as an
                // explicit atom. Counting that stray hydrogen would inflate the
                // weights, valence-electron count, and Morgan-environment density
                // relative to RDKit. Suppress redundant explicit hydrogens on a
                // LOCAL copy so the whole group computes on the implicit-H graph
                // RDKit uses. The copy is essential: ctx.RingPerceivedMol() returns
                // a const reference shared with the Mordred source, and must not be
                // mutated. Re-perceive rings/aromaticity/hybridization after
                // suppression (FractionCSP3 relies on hybridization) and preserve
                // stereo parity, so the stereocenter counts still match RDKit.
                OEChem::OEGraphMol mol(ctx.RingPerceivedMol());
                OEChem::OESuppressHydrogens(mol);
                OEChem::OEFindRingAtomsAndBonds(mol);
                OEChem::OEAssignAromaticFlags(mol);
                OEChem::OEAssignHybridization(mol);

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
                // Phi = Kappa1 * Kappa2 / heavy-atom count (RDKit calcPhi). The
                // Connectivity group runs first (declared dependency) and stashes
                // Kappa1/Kappa2 into the artifact regardless of whether the
                // Kappa* columns are wanted, so Phi resolves even when only Phi
                // is requested. Guard a zero heavy-atom count (RDKit returns 0).
                const auto heavy_atom_count = HeavyAtomCount(mol);
                if (artifacts.kappas_ready && heavy_atom_count != 0u) {
                    set_float(builder, "Phi",
                              artifacts.kappa1 * artifacts.kappa2
                                  / static_cast<double>(heavy_atom_count));
                } else {
                    set_float(builder, "Phi", 0.0);
                }
                // SPS deferred — RDKit SpacialScore depends on RDKit's
                // potential-stereocenter enumeration (FindMolChiralCenters with
                // includeUnassigned=True, useLegacyImplementation=False) and its
                // exact conjugated-heteroatom hybridization model, neither exposed
                // by OpenEye (up to 50% deviation on cage systems such as cubane).
                // Left MISSING rather than shipping a knowingly-wrong value; a
                // dedicated deep-dive follow-up task is needed. See the SPS
                // deferral note in the task-3 report.
            }});

        // Group: rdkit:RingCounts — the 11 dependency-free SSSR ring-count
        // classifications. All are integer counts and every one must match
        // RDKit exactly; the tier label is moot for integers (the conformance
        // test compares them for exact equality regardless of tier). RingCount
        // is RDKit's symmetrized SSSR count, not the plain cyclomatic ring
        // number, so it is computed from compute_symmetrized_sssr_rings rather
        // than an OpenEye ring-system count (which would diverge on caged and
        // fused systems such as cubane and propellanes).
        groups.push_back(RDKitGroup{
            RDKitGroupId::RingCounts,
            rdkit_column_indices(
                s,
                {"RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
                 "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAliphaticCarbocycles",
                 "NumAliphaticHeterocycles", "NumSaturatedCarbocycles", "NumSaturatedHeterocycles",
                 "NumHeterocycles"}),
            {},  // dependency-free
            [](const OEChem::OEMolBase&, ComputeContext& ctx,
               RDKitGroupArtifacts&, const ColumnRequest&,
               RequestGatedBuilder& builder) {
                // Ring perception + aromaticity define every one of these
                // counts. RingPerceivedMol() applies ring/hybridization but not
                // aromaticity (matching the existing prep), so assign aromatic
                // flags on a LOCAL copy — the context reference is shared with
                // the Mordred source and must not be mutated. The SSSR ring set
                // itself is a purely topological property of this graph.
                OEChem::OEGraphMol mol(ctx.RingPerceivedMol());
                OEChem::OEAssignAromaticFlags(mol);

                const auto counts = classify_ring_counts(mol);
                set_int(builder, "RingCount", counts.ring_count);
                set_int(builder, "NumAromaticRings", counts.aromatic);
                set_int(builder, "NumAliphaticRings", counts.aliphatic);
                set_int(builder, "NumSaturatedRings", counts.saturated);
                set_int(builder, "NumAromaticCarbocycles", counts.aromatic_carbocycle);
                set_int(builder, "NumAromaticHeterocycles", counts.aromatic_heterocycle);
                set_int(builder, "NumAliphaticCarbocycles", counts.aliphatic_carbocycle);
                set_int(builder, "NumAliphaticHeterocycles", counts.aliphatic_heterocycle);
                set_int(builder, "NumSaturatedCarbocycles", counts.saturated_carbocycle);
                set_int(builder, "NumSaturatedHeterocycles", counts.saturated_heterocycle);
                set_int(builder, "NumHeterocycles", counts.heterocycle);
            }});

        // Group: rdkit:Connectivity — the 20 float connectivity/shape indices.
        // Everything is computed from the shared heavy-atom graph
        // (ctx.HeavyAtomGraph()), whose adjacency degree is the heavy-atom
        // degree and whose bond orders carry aromaticity (1.5). The group also
        // stashes Kappa1/Kappa2 into the artifact for the CountsWeights `Phi`
        // column, unconditionally (not gated on the Kappa* columns being wanted)
        // so `Phi` still resolves when Connectivity runs only as a dependency
        // under column pruning.
        groups.push_back(RDKitGroup{
            RDKitGroupId::Connectivity,
            rdkit_column_indices(
                s,
                {"Chi0", "Chi1", "Chi0n", "Chi1n", "Chi2n", "Chi3n", "Chi4n",
                 "Chi0v", "Chi1v", "Chi2v", "Chi3v", "Chi4v", "HallKierAlpha",
                 "Kappa1", "Kappa2", "Kappa3", "BertzCT", "BalabanJ", "Ipc", "AvgIpc"}),
            {},  // dependency-free (feeds CountsWeights' Phi via the artifact)
            [](const OEChem::OEMolBase&, ComputeContext& ctx,
               RDKitGroupArtifacts& artifacts, const ColumnRequest&,
               RequestGatedBuilder& builder) {
                const auto& graph = ctx.HeavyAtomGraph();
                const auto atom_info = rdkit_connectivity_atom_info(graph);

                // Per-atom delta tables (already 1/sqrt-transformed); the sigma
                // (n) and valence (v) variants differ only for second-row+ atoms.
                std::vector<double> sigma_deltas;
                std::vector<double> valence_deltas;
                sigma_deltas.reserve(atom_info.size());
                valence_deltas.reserve(atom_info.size());
                for (const auto& info : atom_info) {
                    sigma_deltas.push_back(rdkit_sigma_delta(info));
                    valence_deltas.push_back(rdkit_valence_delta(info));
                }

                set_float(builder, "Chi0", rdkit_chi_simple(graph, false));
                set_float(builder, "Chi1", rdkit_chi_simple(graph, true));
                set_float(builder, "Chi0n", rdkit_chi_order(graph, sigma_deltas, 0u));
                set_float(builder, "Chi1n", rdkit_chi_order(graph, sigma_deltas, 1u));
                set_float(builder, "Chi2n", rdkit_chi_order(graph, sigma_deltas, 2u));
                set_float(builder, "Chi3n", rdkit_chi_order(graph, sigma_deltas, 3u));
                set_float(builder, "Chi4n", rdkit_chi_order(graph, sigma_deltas, 4u));
                set_float(builder, "Chi0v", rdkit_chi_order(graph, valence_deltas, 0u));
                set_float(builder, "Chi1v", rdkit_chi_order(graph, valence_deltas, 1u));
                set_float(builder, "Chi2v", rdkit_chi_order(graph, valence_deltas, 2u));
                set_float(builder, "Chi3v", rdkit_chi_order(graph, valence_deltas, 3u));
                set_float(builder, "Chi4v", rdkit_chi_order(graph, valence_deltas, 4u));

                const auto hybridizations = rdkit_hybridizations(graph, atom_info);
                double alpha = 0.0;
                for (std::size_t i = 0u; i < atom_info.size(); ++i) {
                    alpha += rdkit_hall_kier_alpha_contribution(
                        atom_info[i].atomic_number, hybridizations[i]);
                }
                set_float(builder, "HallKierAlpha", alpha);

                const auto kappas = rdkit_kappa_values(graph, alpha);
                set_float(builder, "Kappa1", kappas.kappa1);
                set_float(builder, "Kappa2", kappas.kappa2);
                set_float(builder, "Kappa3", kappas.kappa3);
                // Publish the Kappa values for the CountsWeights `Phi` column,
                // request-independently so pruning to only {"Phi"} still works.
                artifacts.kappa1 = kappas.kappa1;
                artifacts.kappa2 = kappas.kappa2;
                artifacts.kappas_ready = true;

                set_float(builder, "BertzCT", rdkit_bertz_ct(graph));
                set_float(builder, "BalabanJ", rdkit_balaban_j(graph));
                set_float(builder, "Ipc", rdkit_ipc(graph, false));
                set_float(builder, "AvgIpc", rdkit_ipc(graph, true));
            }});

        // Group: rdkit:Crippen — MolLogP, MolMR. RDKit sums the Wildman-Crippen
        // atom contributions over the HYDROGEN-ADDED molecule (so the total
        // differs from summing ctx.CrippenContributions(), which is the
        // hydrogen-suppressed per-atom vector). The shared, oracle-verified
        // Mordred SLogP/SMR computation reproduces exactly this H-added sum, so
        // it is reused directly rather than re-summing the context accessor.
        groups.push_back(RDKitGroup{
            RDKitGroupId::Crippen,
            rdkit_column_indices(s, {"MolLogP", "MolMR"}),
            {},  // dependency-free
            [](const OEChem::OEMolBase& mol, ComputeContext&,
               RDKitGroupArtifacts&, const ColumnRequest&,
               RequestGatedBuilder& builder) {
                const auto [logp, mr] = compute_crippen_contribution_sums(mol);
                set_float(builder, "MolLogP", logp);
                set_float(builder, "MolMR", mr);
            }});

        // Group: rdkit:SurfacePolarity — LabuteASA (total), TPSA. LabuteASA reuses
        // the shared, oracle-verified Labute model's total (heavy-atom surface
        // contributions plus the hydrogen shielding term); the NEW
        // ctx.LabuteAtomContributions() exposes the per-atom vector — whose sum
        // deliberately omits that hydrogen term — for Task 8's VSA bins, so it
        // cannot supply the total on its own. TPSA is RDKit's N/O-only Ertl polar
        // surface area, ported natively (see rdkit_tpsa) because OpenEye's
        // OEGet2dPSA diverges from RDKit on a few nitrogen/oxygen edge cases.
        groups.push_back(RDKitGroup{
            RDKitGroupId::SurfacePolarity,
            rdkit_column_indices(s, {"LabuteASA", "TPSA"}),
            {},  // dependency-free
            [](const OEChem::OEMolBase& mol, ComputeContext& ctx,
               RDKitGroupArtifacts&, const ColumnRequest&,
               RequestGatedBuilder& builder) {
                const auto values = compute_labute_asa(mol);
                if (values.has_value()) {
                    set_float(builder, "LabuteASA", values->total);
                }
                // Ring/aromaticity perception drives the Ertl fragment matching.
                // RingPerceivedMol() carries ring + hybridization but not aromatic
                // flags, so assign them on a LOCAL copy — the context reference is
                // shared with the Mordred source and must not be mutated.
                OEChem::OEGraphMol tpsa_mol(ctx.RingPerceivedMol());
                OEChem::OEAssignAromaticFlags(tpsa_mol);
                set_float(builder, "TPSA", rdkit_tpsa(tpsa_mol));
            }});

        // PartialCharge deferred — RDKit's Max/Min PartialCharge use RDKit's
        // Gasteiger PEOE solver, which diverges from OpenEye's (both the shared
        // context solver and the QuacpacTk OEGasteigerCharges implementation) on
        // cumulated-double-bond systems (allenes, isocyanates, isothiocyanates,
        // carbon dioxide, azides, carbodiimides, and related cumulenes). The
        // MaxPartialCharge, MinPartialCharge, MaxAbsPartialCharge, and
        // MinAbsPartialCharge columns are therefore left uncomputed (missing)
        // until a native RDKit-Gasteiger port is available (see follow-up).

        // Group: rdkit:EState — Max/Min/MaxAbs/MinAbsEStateIndex over the per-atom
        // EState vector read from the SHARED context (ctx.EStateIndices()), so
        // Task 8's EState VSA bins reuse the same memoized vector. RDKit reduces
        // over the signed indices for Max/Min and over |index| for the Abs
        // variants. An empty vector (no heavy atoms) leaves every column missing,
        // matching RDKit's degenerate handling downstream.
        groups.push_back(RDKitGroup{
            RDKitGroupId::EState,
            rdkit_column_indices(s, {"MaxEStateIndex", "MinEStateIndex",
                                     "MaxAbsEStateIndex", "MinAbsEStateIndex"}),
            {},  // dependency-free
            [](const OEChem::OEMolBase&, ComputeContext& ctx,
               RDKitGroupArtifacts&, const ColumnRequest&,
               RequestGatedBuilder& builder) {
                const std::vector<double>& estate = ctx.EStateIndices();
                if (estate.empty()) {
                    return;
                }
                double max_index = estate.front();
                double min_index = estate.front();
                double max_abs = std::abs(estate.front());
                double min_abs = std::abs(estate.front());
                for (const auto value : estate) {
                    max_index = std::max(max_index, value);
                    min_index = std::min(min_index, value);
                    max_abs = std::max(max_abs, std::abs(value));
                    min_abs = std::min(min_abs, std::abs(value));
                }
                set_float(builder, "MaxEStateIndex", max_index);
                set_float(builder, "MinEStateIndex", min_index);
                set_float(builder, "MaxAbsEStateIndex", max_abs);
                set_float(builder, "MinAbsEStateIndex", min_abs);
            }});

        // Group: rdkit:VSA — the surface-area-binned descriptors across five
        // sub-families (40 emitted; the 14 PEOE_VSA bins are deferred, see the note
        // below). Every family bins two SHARED per-atom context vectors
        // (already memoized per heavy atom in one aligned order): the Labute
        // surface contributions (ctx.LabuteAtomContributions(), RDKit's
        // VSAContribs), the Crippen SlogP/SMR contributions
        // (ctx.CrippenContributions()), the Gasteiger charges
        // (ctx.GasteigerAtomCharges()), and the EState indices
        // (ctx.EStateIndices()). SlogP/SMR/PEOE bin by their property and
        // accumulate the surface contribution; EState_VSA bins by EState and
        // accumulates surface; VSA_EState transposes that — it bins by surface and
        // accumulates EState (the property binned and the value accumulated are
        // SWAPPED between the two, per RDKit's EState_VSA_/VSA_EState_). The group
        // declares a dependency on EState so the emission ordering is exercised;
        // the per-atom vectors themselves come from ctx, not a group artifact.
        // SlogP_VSA9, SMR_VSA8, and EState_VSA11 are structural zeros excluded from
        // the schema, so those bins are computed but not emitted.
        //
        // PEOE_VSA deferred — its 14 bins bucket the Gasteiger partial charges,
        // the SAME model whose OpenEye-vs-RDKit divergence on cumulated-double-bond
        // systems (allenes, isocyanates, isothiocyanates, azides, ...) and on
        // elements RDKit has no Gasteiger parameters for (RDKit assigns those a NaN
        // charge that bisect_right routes to the open tail bin, PEOE_VSA14, whereas
        // OpenEye returns a finite charge that lands in a middle bin) caused the
        // MaxPartialCharge family to be deferred above. Binning inherits that
        // divergence one-to-one (verified: every panel PEOE_VSA divergence is a
        // cumulene or an unparametrized-element molecule; the other four VSA
        // sub-families, which never touch Gasteiger, match RDKit within loose across
        // the whole panel). The 14 PEOE_VSA columns therefore stay in the schema but
        // are left uncomputed (missing) until a native RDKit-Gasteiger port lands,
        // mirroring the PartialCharge deferral rather than shipping wrong bins.
        std::vector<std::string> vsa_columns;
        for (int k = 1; k <= 12; ++k) {
            if (k != 9) vsa_columns.push_back("SlogP_VSA" + std::to_string(k));
        }
        for (int k = 1; k <= 10; ++k) {
            if (k != 8) vsa_columns.push_back("SMR_VSA" + std::to_string(k));
        }
        for (int k = 1; k <= 10; ++k) {
            vsa_columns.push_back("EState_VSA" + std::to_string(k));
        }
        for (int k = 1; k <= 10; ++k) {
            vsa_columns.push_back("VSA_EState" + std::to_string(k));
        }
        groups.push_back(RDKitGroup{
            RDKitGroupId::VSA,
            rdkit_column_indices(s, vsa_columns),
            {RDKitGroupId::EState},  // exercises the emission-ordering dependency
            [](const OEChem::OEMolBase&, ComputeContext& ctx,
               RDKitGroupArtifacts&, const ColumnRequest&,
               RequestGatedBuilder& builder) {
                // RDKit bin bounds (verified against rdkit.Chem.MolSurf +
                // EState_VSA), used exactly. N bounds define N+1 bins; the excluded
                // schema bins are skipped on emission, not here.
                static constexpr std::array<double, 11> kSlogpBins{
                    {-0.4, -0.2, 0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6}};
                static constexpr std::array<double, 9> kSmrBins{
                    {1.29, 1.82, 2.24, 2.45, 2.75, 3.05, 3.63, 3.8, 4.0}};
                static constexpr std::array<double, 10> kEstateBins{
                    {-0.39, 0.29, 0.717, 1.165, 1.54, 1.807, 2.05, 4.69, 9.17, 15.0}};
                static constexpr std::array<double, 9> kVsaBins{
                    {4.78, 5.0, 5.41, 5.74, 6.0, 6.07, 6.45, 7.0, 11.0}};

                const std::vector<double>& surface = ctx.LabuteAtomContributions();
                const auto& crippen = ctx.CrippenContributions();
                const std::vector<double>& estate = ctx.EStateIndices();

                // SlogP_VSA / SMR_VSA: bin by the Crippen property, accumulate the
                // surface contribution. PEOE_VSA is deferred (see the note above),
                // so the Gasteiger charges are intentionally not binned here.
                if (const auto slogp =
                        rdkit_vsa_bin_accumulate(crippen.logp, surface, kSlogpBins)) {
                    rdkit_emit_vsa_family(builder, "SlogP_VSA", *slogp, 9);
                }
                if (const auto smr =
                        rdkit_vsa_bin_accumulate(crippen.mr, surface, kSmrBins)) {
                    rdkit_emit_vsa_family(builder, "SMR_VSA", *smr, 8);
                }
                // EState_VSA: bin by EState, accumulate surface.
                if (const auto estate_vsa =
                        rdkit_vsa_bin_accumulate(estate, surface, kEstateBins)) {
                    rdkit_emit_vsa_family(builder, "EState_VSA", *estate_vsa, 11);
                }
                // VSA_EState: the transpose — bin by surface, accumulate EState.
                if (const auto vsa_estate =
                        rdkit_vsa_bin_accumulate(surface, estate, kVsaBins)) {
                    rdkit_emit_vsa_family(builder, "VSA_EState", *vsa_estate, 0);
                }
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
