#include "oefp/rdkit_stereogenicity.h"

#include "oefp/molecular_properties.h"
#include "oefp/mordred_intermediates.h"

#include <oesystem.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace OEFP {
namespace {

/// RDKit's minimum ring size for retaining specified double-bond stereo.
/// Ring double bonds in rings smaller than this have their stereo cleared at
/// parse time (trans double bond is geometrically impossible in small rings).
/// Mirrors RDKit's Chirality::minRingSizeForDoubleBondStereo.
constexpr std::size_t MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO = 8u;

// Heavy-atom-only view of the molecule. Iteration order mirrors
// build_mordred_heavy_atom_graph (GetAtoms() filtered to non-hydrogen), so the
// output flags align with the SPS caller's heavy-atom graph. The module is
// hydrogen-representation-agnostic: heavy neighbours come from GetHvyDegree()
// and hydrogens from GetTotalHCount(), so explicit/bracket hydrogens never
// perturb the perceived result.
//
// RDKit's getAtomNonzeroDegree also excludes UNSPECIFIED/ZERO/dative-donor
// bonds; GetHvyDegree() keeps them, which for typical organics is identical (no
// divergence hit on the validated panel).
struct HeavyGraph {
    std::vector<const OEChem::OEAtomBase*> atoms;  // heavy atoms, iteration order
    std::unordered_map<unsigned int, std::size_t> index_by_oeidx;
    std::vector<std::uint32_t> atomic_num;
    std::vector<std::uint32_t> isotope;
    std::vector<int> formal_charge;
    std::vector<int> heavy_degree;      // RDKit getAtomNonzeroDegree (heavy only)
    std::vector<int> h_count;           // RDKit getTotalNumHs
    std::vector<unsigned int> hyb;      // OEHybridization
    std::vector<int> explicit_valence;  // sum of heavy bond orders (0 H in the paths that read it)

    // Heavy adjacency with a per-edge bond key parallel to each neighbour list,
    // and the edge id joining each neighbour pair (for ring/tolerance lookups).
    std::vector<std::vector<std::size_t>> adjacency;
    std::vector<std::vector<int>> neighbor_bond_key;
    std::vector<std::vector<std::size_t>> neighbor_edge;

    // Undirected heavy-edge table.
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> edge_id;
    std::vector<std::vector<std::size_t>> atom_edges;  // incident edge ids per atom
};

// Encode a bond for the symmetry refinement. Aromatic bonds get their own code
// (RDKit orders them by the ":" symbol regardless of the kekulé order), every
// other bond is keyed by its integer order. Codes >= 2 (aromatic, double,
// triple) also serve as the "multiple/conjugation-carrying" test.
int bond_key(const OEChem::OEBondBase& bond) {
    if (bond.IsAromatic()) {
        return 100;
    }
    return static_cast<int>(bond.GetOrder());
}

std::size_t intern_edge(HeavyGraph& g, std::size_t a, std::size_t b) {
    const auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
    const auto it = g.edge_id.find(key);
    if (it != g.edge_id.end()) {
        return it->second;
    }
    const std::size_t id = g.edges.size();
    g.edge_id.emplace(key, id);
    g.edges.push_back(key);
    return id;
}

HeavyGraph build_heavy_graph(const OEChem::OEMolBase& mol) {
    HeavyGraph g;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (is_hydrogen(*atom)) {
            continue;
        }
        g.index_by_oeidx.emplace(atom->GetIdx(), g.atoms.size());
        g.atoms.push_back(&*atom);
        g.atomic_num.push_back(atom->GetAtomicNum());
        g.isotope.push_back(atom->GetIsotope());
        g.formal_charge.push_back(atom->GetFormalCharge());
        g.heavy_degree.push_back(static_cast<int>(atom->GetHvyDegree()));
        g.h_count.push_back(static_cast<int>(atom->GetTotalHCount()));
        g.hyb.push_back(atom->GetHyb());

        int explicit_valence = 0;
        for (OESystem::OEIter<OEChem::OEBondBase> bond = atom->GetBonds(); bond; ++bond) {
            const auto* other = bond->GetNbr(&*atom);
            if (other != nullptr && !is_hydrogen(*other)) {
                explicit_valence += static_cast<int>(bond->GetOrder());
            }
        }
        g.explicit_valence.push_back(explicit_valence);
    }

    const std::size_t n = g.atoms.size();
    g.adjacency.resize(n);
    g.neighbor_bond_key.resize(n);
    g.neighbor_edge.resize(n);
    g.atom_edges.resize(n);
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
            continue;
        }
        const auto bi = g.index_by_oeidx.find(begin->GetIdx());
        const auto ei = g.index_by_oeidx.find(end->GetIdx());
        if (bi == g.index_by_oeidx.end() || ei == g.index_by_oeidx.end()) {
            continue;
        }
        const std::size_t a = bi->second;
        const std::size_t b = ei->second;
        const int key = bond_key(*bond);
        const std::size_t eid = intern_edge(g, a, b);
        g.adjacency[a].push_back(b);
        g.neighbor_bond_key[a].push_back(key);
        g.neighbor_edge[a].push_back(eid);
        g.atom_edges[a].push_back(eid);
        g.adjacency[b].push_back(a);
        g.neighbor_bond_key[b].push_back(key);
        g.neighbor_edge[b].push_back(eid);
        g.atom_edges[b].push_back(eid);
    }
    return g;
}

// Symmetrized-SSSR ring set (RDKit RingInfo) expressed on heavy indices, plus
// the per-atom / per-bond ring-membership counts flagRingStereo consults.
struct RingInfo {
    std::vector<std::vector<std::size_t>> atom_rings;  // heavy indices, cyclic order
    std::vector<std::vector<std::size_t>> bond_rings;  // parallel edge ids
    std::vector<int> num_atom_rings;                   // per heavy atom
    std::vector<int> num_bond_rings;                   // per edge id
};

RingInfo build_ring_info(const OEChem::OEMolBase& mol, HeavyGraph& g) {
    RingInfo ri;
    ri.num_atom_rings.assign(g.atoms.size(), 0);
    for (const auto& ring : compute_symmetrized_sssr_rings(mol)) {
        std::vector<std::size_t> aring;
        aring.reserve(ring.size());
        for (const auto* atom : ring) {
            const auto it = g.index_by_oeidx.find(atom->GetIdx());
            if (it != g.index_by_oeidx.end()) {
                aring.push_back(it->second);
            }
        }
        if (aring.size() < 3u) {
            continue;
        }
        std::vector<std::size_t> bring;
        bring.reserve(aring.size());
        for (std::size_t k = 0u; k < aring.size(); ++k) {
            bring.push_back(intern_edge(g, aring[k], aring[(k + 1u) % aring.size()]));
        }
        for (const auto a : aring) {
            ++ri.num_atom_rings[a];
        }
        ri.atom_rings.push_back(std::move(aring));
        ri.bond_rings.push_back(std::move(bring));
    }
    ri.num_bond_rings.assign(g.edges.size(), 0);
    for (const auto& bring : ri.bond_rings) {
        for (const auto e : bring) {
            ++ri.num_bond_rings[e];
        }
    }
    return ri;
}

bool in_ring_of_size(const RingInfo& ri, std::size_t i, std::size_t size) {
    for (const auto& ring : ri.atom_rings) {
        if (ring.size() == size
            && std::find(ring.begin(), ring.end(), i) != ring.end()) {
            return true;
        }
    }
    return false;
}

/// Return the size of the smallest ring containing the given edge, or 0 if
/// the edge is not in any ring.
std::size_t min_ring_size_for_bond(const RingInfo& ri, std::size_t edge_id) {
    std::size_t min_size = 0u;
    for (std::size_t ridx = 0u; ridx < ri.bond_rings.size(); ++ridx) {
        const auto& bring = ri.bond_rings[ridx];
        if (std::find(bring.begin(), bring.end(), edge_id) != bring.end()) {
            const std::size_t ring_size = ri.atom_rings[ridx].size();
            if (min_size == 0u || ring_size < min_size) {
                min_size = ring_size;
            }
        }
    }
    return min_size;
}

// RDKit QueryOps queryIsAtomBridgehead: at least three ring bonds on the atom,
// each ring through the atom sharing >= 2 bonds with another ring through the
// atom (SSSR-based, so bridged systems with only two SSSRs still qualify).
bool is_bridgehead(const HeavyGraph& g, const RingInfo& ri, std::size_t i) {
    if (g.heavy_degree[i] < 3) {
        return false;
    }
    std::vector<std::size_t> atom_ring_bonds;
    for (const auto e : g.atom_edges[i]) {
        if (ri.num_bond_rings[e] > 0) {
            atom_ring_bonds.push_back(e);
        }
    }
    if (atom_ring_bonds.size() < 3u) {
        return false;
    }
    const auto has = [&](const std::vector<std::size_t>& v, std::size_t x) {
        return std::find(v.begin(), v.end(), x) != v.end();
    };
    const std::size_t nrings = ri.bond_rings.size();
    std::vector<bool> overlap(nrings, false);
    for (std::size_t a = 0u; a < nrings; ++a) {
        const auto& ring_a = ri.bond_rings[a];
        bool atom_in_a = false;
        for (const auto e : ring_a) {
            if (has(atom_ring_bonds, e)) {
                atom_in_a = true;
                break;
            }
        }
        if (!atom_in_a) {
            continue;
        }
        for (std::size_t b = a + 1u; b < nrings; ++b) {
            unsigned int shared = 0u;
            bool atom_in_b = false;
            for (const auto e : ri.bond_rings[b]) {
                if (has(atom_ring_bonds, e)) {
                    atom_in_b = true;
                }
                if (has(ring_a, e)) {
                    ++shared;
                }
                if (shared >= 2u && atom_in_b) {
                    overlap[a] = true;
                    overlap[b] = true;
                    break;
                }
            }
        }
        if (!overlap[a]) {
            return false;
        }
    }
    return true;
}

// Whether the atom carries a conjugated bond, a proxy for RDKit's
// MolOps::atomHasConjugatedBond used only by the three-coordinate N special
// case: any incident multiple/aromatic bond, or a neighbour bearing one (the
// nitrogen lone pair then conjugates into the adjacent pi system).
bool has_conjugated_bond(const HeavyGraph& g, std::size_t i) {
    const auto carries_pi = [&](std::size_t a) {
        for (const auto key : g.neighbor_bond_key[a]) {
            if (key >= 2) {
                return true;
            }
        }
        return false;
    };
    if (carries_pi(i)) {
        return true;
    }
    for (const auto nbr : g.adjacency[i]) {
        if (carries_pi(nbr)) {
            return true;
        }
    }
    return false;
}

// RDKit detail::isAtomPotentialTetrahedralCenter (FindStereo.cpp:50-114),
// remapped to hydrogen-agnostic heavy primitives.
bool is_potential_tetrahedral_center(const HeavyGraph& g, const RingInfo& ri, std::size_t i) {
    const int nz = g.heavy_degree[i];
    const int tnz = nz + g.h_count[i];
    const std::uint32_t z = g.atomic_num[i];
    if (tnz > 4) {
        return false;
    }
    if (nz == 4) {
        return true;
    }
    if (nz <= 1) {
        return false;
    }
    if (nz == 2 && z != 15u && z != 33u) {
        return false;
    }
    if (z == 15u || z == 33u) {  // phosphine / arsine, always accepted
        return true;
    }
    if (nz == 3) {
        if (g.h_count[i] == 1) {
            // One heavy-invisible hydrogen: RDKit's has_protium_neighbor test
            // guards against a second (explicit) hydrogen, but GetTotalHCount()
            // already folds every hydrogen into this count, so exactly one here
            // means no second protium and the centre is accepted.
            return true;
        }
        // Three heavy neighbours, no hydrogen: only a few special cases qualify.
        // Sulfur / selenium with a lone pair (a double bond or a positive charge).
        if ((z == 16u || z == 34u)
            && (g.explicit_valence[i] == 4
                || (g.explicit_valence[i] == 3 && g.formal_charge[i] == 1))) {
            return true;
        }
        // Three-coordinate nitrogen: SP3, not conjugated, and either in a
        // three-membered ring (InChI) or a bridgehead (RDKit extension).
        if (z == 7u && g.hyb[i] == OEChem::OEHybridization::sp3
            && !has_conjugated_bond(g, i)
            && (in_ring_of_size(ri, i, 3u) || is_bridgehead(g, ri, i))) {
            return true;
        }
        return false;
    }
    return false;
}

// RDKit detail::isAtomPotentialNontetrahedralCenter (FindStereo.cpp:27-49):
// element >= Mg (or Be), 2 <= total degree <= 6, and (unspecified chirality)
// total degree >= 4. getStereoInfo then keeps only total-degree-4 as
// Atom_Tetrahedral, which the FindMolChiralCenters filter retains.
bool is_potential_nontetrahedral_center(const HeavyGraph& g, std::size_t i) {
    const int tnz = g.heavy_degree[i] + g.h_count[i];
    const std::uint32_t z = g.atomic_num[i];
    if (tnz > 6 || tnz < 2 || (z < 12u && z != 4u)) {
        return false;
    }
    return tnz >= 4;
}

// RDKit's getAtomCompareSymbol seed (isotope + element + charge), extended with
// heavy degree. This is the per-atom invariant fed to the equitable-partition
// refinement (the module's stand-in for Canon::rankFragmentAtoms). Stereo
// suffixes ("_CW"/"_CCW" for specified centres, a unique "_idx" for possible
// centres) are appended to this seed by the caller.
std::string base_atom_symbol(const HeavyGraph& g, std::size_t i) {
    return std::to_string(g.heavy_degree[i]) + '|' + std::to_string(g.isotope[i]) + '|'
           + std::to_string(g.atomic_num[i]) + '|' + std::to_string(g.formal_charge[i]);
}

// Parity (0 or 1) of the swaps needed to bring the index-ordered neighbour ranks
// into ascending (rank-sorted) order. A faithful port of RDKit's
// countSwapsToInterconvert (RDGeneral/utils.h) with ref = the index order and
// probe = the sorted order, so its tie-handling matches RDKit exactly; this is
// what drives the "_CW"/"_CCW" descriptor flip in updateAtoms.
int neighbor_rank_swap_parity(const std::vector<int>& index_order) {
    std::vector<int> probe = index_order;
    std::sort(probe.begin(), probe.end());
    unsigned int swaps = 0u;
    for (std::size_t i = 0u; i < index_order.size(); ++i) {
        if (probe[i] == index_order[i]) {
            continue;
        }
        std::size_t j = i;
        while (j < probe.size() && probe[j] != index_order[i]) {
            ++j;
        }
        if (j < probe.size()) {
            std::swap(probe[i], probe[j]);
            ++swaps;
        }
    }
    return static_cast<int>(swaps & 1u);
}

// Equitable-partition (colour) refinement reproducing the symmetry classes that
// RDKit's Canon::rankFragmentAtoms(breakTies=false) yields for FindStereo: only
// equivalence of ranks matters (the duplicate rule tests rank equality), so the
// class labels themselves are arbitrary. The per-atom seed symbols are supplied
// by the caller (RDKit's atomSymbols array); refinement then folds in each
// atom's sorted multiset of (bond key, neighbour class).
std::vector<int> rank_by_symbols(const HeavyGraph& g, const std::vector<std::string>& symbols) {
    const std::size_t n = g.atoms.size();

    auto compress = [](const std::vector<std::string>& sigs) {
        std::map<std::string, int> ids;
        for (const auto& s : sigs) {
            ids.emplace(s, 0);
        }
        int next = 0;
        for (auto& kv : ids) {
            kv.second = next++;
        }
        std::vector<int> out(sigs.size());
        for (std::size_t i = 0u; i < sigs.size(); ++i) {
            out[i] = ids[sigs[i]];
        }
        return std::make_pair(out, next);
    };

    auto [color, classes] = compress(symbols);

    while (true) {
        std::vector<std::string> next(n);
        for (std::size_t i = 0u; i < n; ++i) {
            std::vector<std::pair<int, int>> nbrs;
            nbrs.reserve(g.adjacency[i].size());
            for (std::size_t k = 0u; k < g.adjacency[i].size(); ++k) {
                nbrs.emplace_back(g.neighbor_bond_key[i][k], color[g.adjacency[i][k]]);
            }
            std::sort(nbrs.begin(), nbrs.end());
            std::string s = std::to_string(color[i]);
            for (const auto& [bk, nc] : nbrs) {
                s += ';' + std::to_string(bk) + ',' + std::to_string(nc);
            }
            next[i] = std::move(s);
        }
        auto [ncolor, nclasses] = compress(next);
        if (nclasses == classes) {
            return ncolor;
        }
        color = std::move(ncolor);
        classes = nclasses;
    }
}

// RDKit's internal potential-stereo-bond sets, used ONLY to support ring-atom
// perception in flagRingStereo. These reproduce RDKit's initBondInfo /
// detail::isBondPotentialStereoBond (FindStereo.cpp:378-409): a DOUBLE bond
// whose two ends each have 1 < total degree < 4 and fewer than two hydrogens,
// and which is not in a ring smaller than MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO.
// This LOCAL gate deliberately does NOT test substituent distinguishability, so
// e.g. an exocyclic =C(C)C (two identical methyls) still qualifies. This set is
// separate from compute_stereo_bond_atoms (the legacy OUTPUT path) and must not
// be conflated with it: total degree here is heavy + H (RDKit getTotalDegree),
// whereas the output path keys on heavy degree only.
struct StereoBondSets {
    std::vector<bool> possible;  // per edge id: gate passed, no specified E/Z
    std::vector<bool> known;     // per edge id: gate passed, specified E/Z
};

StereoBondSets compute_internal_stereo_bonds(const OEChem::OEMolBase& mol,
                                             const HeavyGraph& g, const RingInfo& ri) {
    StereoBondSets sets;
    sets.possible.assign(g.edges.size(), false);
    sets.known.assign(g.edges.size(), false);

    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        if (bond->GetOrder() != 2u || bond->IsAromatic()) {
            continue;
        }
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
            continue;
        }
        const auto bi = g.index_by_oeidx.find(begin->GetIdx());
        const auto ei = g.index_by_oeidx.find(end->GetIdx());
        if (bi == g.index_by_oeidx.end() || ei == g.index_by_oeidx.end()) {
            continue;
        }
        const std::size_t a = bi->second;
        const std::size_t b = ei->second;

        // Local gate: total degree (heavy + H) strictly between 1 and 4, and
        // fewer than two hydrogens, on both ends.
        const int tdeg_a = g.heavy_degree[a] + g.h_count[a];
        const int tdeg_b = g.heavy_degree[b] + g.h_count[b];
        if (!(tdeg_a > 1 && tdeg_a < 4 && tdeg_b > 1 && tdeg_b < 4
              && g.h_count[a] < 2 && g.h_count[b] < 2)) {
            continue;
        }

        const auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
        const auto it = g.edge_id.find(key);
        if (it == g.edge_id.end()) {
            continue;
        }
        const std::size_t eid = it->second;

        // Reject bonds in any ring smaller than the threshold (trans double bond
        // is geometrically impossible). The smallest containing ring settles it.
        const std::size_t ring_size = min_ring_size_for_bond(ri, eid);
        if (ring_size != 0u && ring_size < MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO) {
            continue;
        }

        if (bond->HasStereoSpecified(OEChem::OEBondStereo::CisTrans)) {
            sets.known[eid] = true;
        } else {
            sets.possible[eid] = true;
        }
    }
    return sets;
}

// RDKit flagRingStereo (FindStereo.cpp:625-746) for the atom path: commit a ring
// only when it holds >= 2 mutually-supporting candidates. Across-ring divisor-2
// (atom opposite in an even ring) and divisor-3 (three alternating positions of
// a ring whose size is a multiple of three) tests, plus a fused common-edge
// walk. Each across position may be supported either by an out-of-ring
// possible/known stereo bond on the across atom (otherFoundByBondCount, e.g. an
// exocyclic double bond) or, only while no bond support has yet been seen, by
// the across atom itself being a candidate (otherFoundByAtomCount). RDKit tests
// the running bond count, so once any earlier across position is bond-supported
// the atom branch is skipped for the remaining positions.
struct RingStereo {
    std::vector<int> atoms;  // possibleRingStereoAtoms, per heavy atom
    std::vector<int> bonds;  // possibleRingStereoBonds, per edge id
};

RingStereo flag_ring_stereo(const HeavyGraph& g, const RingInfo& ri,
                            const std::vector<bool>& possible,
                            const StereoBondSets& stereo_bonds) {
    RingStereo rs;
    rs.atoms.assign(g.atoms.size(), 0);
    rs.bonds.assign(g.edges.size(), 0);

    for (std::size_t ridx = 0u; ridx < ri.atom_rings.size(); ++ridx) {
        const auto& aring = ri.atom_rings[ridx];
        const auto& bring = ri.bond_rings[ridx];
        const std::size_t sz = aring.size();
        const std::size_t half = sz / 2u + (sz % 2u);
        std::vector<bool> in_ring(g.atoms.size(), false);
        int here = 0;

        for (std::size_t ai = 0u; ai < sz; ++ai) {
            const std::size_t aidx = aring[ai];
            if (!possible[aidx]) {
                continue;
            }
            bool matched = false;
            for (const std::size_t divisor : {std::size_t{2u}, std::size_t{3u}}) {
                if (sz % divisor != 0u) {
                    continue;
                }
                const std::size_t inc = sz / divisor;
                unsigned int found_by_bond = 0u;
                unsigned int found_by_atom = 0u;
                for (std::size_t step = inc; step < sz; step += inc) {
                    const std::size_t other = aring[(ai + step) % sz];
                    // otherFoundByBondCount: an out-of-ring possible/known stereo
                    // bond on the across atom (e.g. an exocyclic double bond)
                    // supports this position without the atom itself being a
                    // candidate.
                    bool bond_support = false;
                    for (const std::size_t e : g.atom_edges[other]) {
                        if ((stereo_bonds.possible[e] || stereo_bonds.known[e])
                            && std::find(bring.begin(), bring.end(), e) == bring.end()) {
                            bond_support = true;
                            break;
                        }
                    }
                    if (bond_support) {
                        ++found_by_bond;
                    }
                    // RDKit only tries the atom branch while the running bond
                    // count is still zero.
                    if (found_by_bond == 0u && possible[other]) {
                        ++found_by_atom;
                    }
                }
                if (found_by_bond == divisor - 1u || found_by_atom == divisor - 1u) {
                    here += 1 + static_cast<int>(found_by_bond);
                    for (std::size_t step = 0u; step < sz; step += inc) {
                        in_ring[aring[(ai + step) % sz]] = true;
                    }
                    matched = true;
                    break;
                }
            }
            if (matched) {
                continue;
            }
            // Fused common-edge walk for bridged/fused quaternary bridgeheads.
            if (ri.num_atom_rings[aidx] > 1) {
                std::size_t prev = aidx;
                for (std::size_t step = 1u; step <= half; ++step) {
                    const std::size_t other = aring[(ai + step) % sz];
                    const auto key = prev < other ? std::make_pair(prev, other)
                                                  : std::make_pair(other, prev);
                    const auto it = g.edge_id.find(key);
                    if (it == g.edge_id.end() || ri.num_bond_rings[it->second] < 2) {
                        break;
                    }
                    if (possible[other]) {
                        here += 2;
                        in_ring[aidx] = true;
                        in_ring[other] = true;
                        break;
                    }
                    prev = other;
                }
            }
        }

        if (here > 1) {
            for (const auto a : aring) {
                if (in_ring[a]) {
                    ++rs.atoms[a];
                }
            }
            for (const auto e : bring) {
                ++rs.bonds[e];
            }
        }
    }
    return rs;
}

// RDKit updateAtoms duplicate rule (FindStereo.cpp:748-869): a candidate is
// disqualified when two controlling (heavy) neighbours share a symmetry class,
// unless the atom is a ring-stereo candidate and the tie is reached through a
// ring-stereo bond (the para-/dependent-stereocenter exception).
bool has_disqualifying_duplicate(const HeavyGraph& g, const RingStereo& rs,
                                 std::size_t i, const std::vector<int>& ranks) {
    std::vector<int> seen;
    seen.reserve(g.adjacency[i].size());
    for (std::size_t k = 0u; k < g.adjacency[i].size(); ++k) {
        const int rnk = ranks[g.adjacency[i][k]];
        if (std::find(seen.begin(), seen.end(), rnk) != seen.end()) {
            if (rs.atoms[i] > 0 && rs.bonds[g.neighbor_edge[i][k]] > 0) {
                continue;  // tolerated dependent-stereo tie; do not record the rank
            }
            return true;
        }
        seen.push_back(rnk);
    }
    return false;
}

// One end of a double bond has distinguishable substituents when, restricted to
// its single/aromatic-bonded non-partner neighbours, it has exactly one such
// neighbour (trivially distinguishable) or two whose symmetry classes differ.
//
// Mirrors RDKit's legacy findPotentialStereoBonds (Chirality.cpp:2910), which
// builds each end's neighbour list with findAtomNeighborsHelper: SINGLE and
// AROMATIC bonds only, excluding the double bond itself. Crucially this also
// excludes any OTHER double bond on the end, so a cumulene centre (allene,
// azide, isocyanate, ketenimine, CO2-like) has no substituent on its cumulated
// side. RDKit then requires BOTH ends' neighbour lists to be non-empty before it
// can flag the bond, so an empty list here means "not distinguishable" and the
// bond is skipped. Plain symmetry classes stand in for assignAtomCIPRanks (equal
// iff topologically equivalent, which is exactly the "ranks differ" test).
bool double_bond_end_distinguishable(const HeavyGraph& g, std::size_t end,
                                     std::size_t partner, const std::vector<int>& ranks) {
    std::vector<int> subs;
    for (std::size_t k = 0u; k < g.adjacency[end].size(); ++k) {
        const std::size_t nbr = g.adjacency[end][k];
        if (nbr == partner) {
            continue;
        }
        // findAtomNeighborsHelper keeps single (key 1) and aromatic (key 100)
        // bonds only; double/triple-bonded neighbours are dropped, which is what
        // strips the cumulated partner of a cumulene centre.
        const int key = g.neighbor_bond_key[end][k];
        if (key == 1 || key == 100) {
            subs.push_back(ranks[nbr]);
        }
    }
    if (subs.empty()) {
        return false;
    }
    if (subs.size() == 1u) {
        return true;
    }
    return subs[0] != subs[1];
}

// Legacy MolOps::findPotentialStereoBonds (Chirality.cpp:2910), which SPS calls
// via rdmolops.FindPotentialStereoBonds. Marks both endpoints of every double
// bond that carries non-STEREONONE stereo afterwards: a non-ring double bond
// whose two-substituent ends are rank-distinguishable (a one-substituent end is
// trivially so), plus any double bond that already carries specified E/Z stereo
// (acyclic or ring size >= MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO). Unmarked ring
// double bonds are skipped entirely, at any size.
std::vector<bool> compute_stereo_bond_atoms(const OEChem::OEMolBase& mol, const HeavyGraph& g, const RingInfo& ri) {
    std::vector<bool> flags(g.atoms.size(), false);
    std::vector<std::string> plain_symbols(g.atoms.size());
    for (std::size_t i = 0u; i < g.atoms.size(); ++i) {
        plain_symbols[i] = base_atom_symbol(g, i);
    }
    const std::vector<int> plain_ranks = rank_by_symbols(g, plain_symbols);

    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        if (bond->GetOrder() != 2u || bond->IsAromatic()) {
            continue;
        }
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
            continue;
        }
        const auto bi = g.index_by_oeidx.find(begin->GetIdx());
        const auto ei = g.index_by_oeidx.find(end->GetIdx());
        if (bi == g.index_by_oeidx.end() || ei == g.index_by_oeidx.end()) {
            continue;
        }
        const std::size_t a = bi->second;
        const std::size_t b = ei->second;
        const bool specified = bond->HasStereoSpecified(OEChem::OEBondStereo::CisTrans);

        bool flag = false;
        if (bond->IsInRing()) {
            // RDKit clears specified stereo on ring double bonds in rings smaller
            // than MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO (trans is geometrically
            // impossible in small rings). Only flag if ring size >= threshold.
            if (specified) {
                const auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
                const auto it = g.edge_id.find(key);
                if (it != g.edge_id.end()) {
                    const std::size_t ring_size = min_ring_size_for_bond(ri, it->second);
                    flag = ring_size >= MIN_RING_SIZE_FOR_DOUBLE_BOND_STEREO;
                }
            }
        } else {
            const bool degree_ok = (g.heavy_degree[a] == 2 || g.heavy_degree[a] == 3)
                                   && (g.heavy_degree[b] == 2 || g.heavy_degree[b] == 3);
            flag = specified
                   || (degree_ok
                       && double_bond_end_distinguishable(g, a, b, plain_ranks)
                       && double_bond_end_distinguishable(g, b, a, plain_ranks));
        }
        if (flag) {
            flags[a] = true;
            flags[b] = true;
        }
    }
    return flags;
}

}  // namespace

RDKitStereogenicity rdkit_potential_stereogenicity(const OEChem::OEMolBase& mol) {
    HeavyGraph g = build_heavy_graph(mol);
    const RingInfo ri = build_ring_info(mol, g);
    const std::size_t n = g.atoms.size();

    // Candidate centres. Nontetrahedral centres participate in the ranking (RDKit
    // puts them in possibleAtoms) but only total-degree-4 ones survive the
    // Atom_Tetrahedral output filter.
    std::vector<bool> candidate(n, false);
    std::vector<bool> tetrahedral(n, false);
    for (std::size_t i = 0u; i < n; ++i) {
        const bool tet = is_potential_tetrahedral_center(g, ri, i);
        const bool nontet = is_potential_nontetrahedral_center(g, i);
        candidate[i] = tet || nontet;
        tetrahedral[i] = tet || (nontet && (g.heavy_degree[i] + g.h_count[i]) == 4);
    }

    // Per-atom seed symbol (RDKit getAtomCompareSymbol) and the parity data for
    // specified centres. base_parity[i] is the tetrahedral parity of atom i with
    // its heavy neighbours in ascending atom-index order (RDKit's index-sorted
    // controllingAtoms); ctrl[i] holds those heavy neighbours in that same order.
    // The Left/Right -> parity mapping is arbitrary: only whether two symmetric
    // centres share a symbol matters, and a global bit-flip preserves every such
    // equality, so any consistent choice reproduces the oracle.
    std::vector<std::string> base(n);
    std::vector<int> base_parity(n, 0);
    std::vector<std::vector<std::size_t>> ctrl(n);
    std::vector<bool> specified(n, false);
    for (std::size_t i = 0u; i < n; ++i) {
        base[i] = base_atom_symbol(g, i);
        if (!candidate[i] || !g.atoms[i]->HasStereoSpecified(OEChem::OEAtomStereo::Tetra)) {
            continue;
        }
        std::vector<std::pair<unsigned int, std::size_t>> ordered;
        ordered.reserve(g.adjacency[i].size());
        for (const auto k : g.adjacency[i]) {
            ordered.emplace_back(g.atoms[k]->GetIdx(), k);
        }
        std::sort(ordered.begin(), ordered.end());
        std::vector<OEChem::OEAtomBase*> oe_nbrs;
        oe_nbrs.reserve(ordered.size());
        ctrl[i].reserve(ordered.size());
        for (const auto& [oeidx, hidx] : ordered) {
            oe_nbrs.push_back(const_cast<OEChem::OEAtomBase*>(g.atoms[hidx]));
            ctrl[i].push_back(hidx);
        }
        const unsigned int st =
            const_cast<OEChem::OEAtomBase*>(g.atoms[i])->GetStereo(oe_nbrs, OEChem::OEAtomStereo::Tetra);
        if (st == OEChem::OEAtomStereo::Left || st == OEChem::OEAtomStereo::Right) {
            specified[i] = true;
            base_parity[i] = (st == OEChem::OEAtomStereo::Right) ? 1 : 0;
        }
    }

    // "_CW"/"_CCW" descriptor rendering (labels are arbitrary; only equality
    // matters). Distinct prefix from the unique "_P" possible-atom label so the
    // two schemes never collide.
    const auto desc_symbol = [&](std::size_t i, int parity) {
        return base[i] + (parity ? "|Scw" : "|Sccw");
    };

    const StereoBondSets stereo_bonds = compute_internal_stereo_bonds(mol, g, ri);

    // RDKit knownAtoms/possibleAtoms partition: specified centres are "known"
    // (carry a parity descriptor), every other candidate is an "unspecified
    // possible" centre (carries a unique per-index label). Unknown ("?"/wiggly)
    // atoms would also be "known" with a unique to_string(idx) label, but that is
    // behaviourally identical to the unspecified unique label for the atom path,
    // so both are folded into the possible branch.
    std::vector<bool> known(n, false);
    std::vector<bool> possible(n, false);
    std::vector<bool> fixed(n, false);
    std::vector<bool> center(n, false);
    std::vector<std::string> symbol(n);
    for (std::size_t i = 0u; i < n; ++i) {
        symbol[i] = base[i];
        if (!candidate[i]) {
            continue;
        }
        if (specified[i]) {
            known[i] = true;
            symbol[i] = desc_symbol(i, base_parity[i]);  // ctrl already index-sorted
        } else {
            possible[i] = true;
            symbol[i] = base[i] + "|P" + std::to_string(i);
        }
    }
    const std::vector<bool> orig_possible = possible;

    RingStereo rs = flag_ring_stereo(g, ri, candidate, stereo_bonds);

    // RDKit updateAtoms (FindStereo.cpp:748-869), atom path. Recomputes each
    // specified centre's descriptor relative to its neighbours' current ranks,
    // drops candidates whose controlling neighbours become indistinguishable
    // (unless a ring-stereo tie tolerates it), and cascades the ring-stereo
    // bookkeeping. Returns whether any symbol/candidate changed.
    //
    // Unlike RDKit's updateAtoms, the duplicate rule is applied to specified
    // centres on EVERY round, not just before they are first fixed. RDKit's new
    // findPotentialStereo trusts a specified tag once fixed because the molecule
    // it sees has already had redundant tags stripped by the legacy parse-time
    // chirality cleanup (MolFromSmiles' SANITIZE_CLEANUPCHIRALITY, which uses the
    // legacy perception by default). OE preserves those tags, so we reproduce the
    // cleanup here: once a centre's controlling neighbours converge to equal
    // ranks it can no longer be a stereocentre and its tag is dropped. The
    // duplicate rule never keeps a centre with equal-rank controlling neighbours,
    // so this only ever removes genuinely redundant (e.g. pseudo-asymmetric meso)
    // tags and matches the SPS oracle (FindMolChiralCenters on the cleaned mol).
    const auto update_atoms = [&](const std::vector<int>& ranks) {
        bool need_another = false;
        center.assign(n, false);
        for (std::size_t i = 0u; i < n; ++i) {
            if (!(known[i] || possible[i])) {
                continue;
            }
            if (!has_disqualifying_duplicate(g, rs, i, ranks)) {
                if (fixed[i]) {
                    center[i] = true;  // frozen survivor; RDKit re-emits its StereoInfo
                    continue;
                }
                std::string new_symbol = symbol[i];
                if (!possible[i]) {  // a "known" (specified) centre
                    if (specified[i]) {
                        std::vector<int> nbr_ranks;
                        nbr_ranks.reserve(ctrl[i].size());
                        for (const auto c : ctrl[i]) {
                            nbr_ranks.push_back(ranks[c]);
                        }
                        const int parity = base_parity[i] ^ neighbor_rank_swap_parity(nbr_ranks);
                        new_symbol = desc_symbol(i, parity);
                    }
                    fixed[i] = true;
                }
                if (symbol[i] != new_symbol) {
                    symbol[i] = new_symbol;
                    need_another = true;
                }
                center[i] = true;
                continue;
            }
            // haveADupe: this centre loses its stereo candidacy this round. This
            // also re-checks already-fixed specified centres, stripping redundant
            // tags OE preserves (see the note above the lambda).
            if (possible[i] || fixed[i]) {
                need_another = true;
            }
            possible[i] = false;
            fixed[i] = false;
            symbol[i] = base[i];
            center[i] = false;
            if (rs.atoms[i] == 0) {
                continue;
            }
            // Ring-stereo cascade (FindStereo.cpp:826-860): drop this atom's ring
            // support, un-fix every ring atom so it is rechecked, and decrement
            // rings that can no longer transmit stereo.
            rs.atoms[i] = 0;
            need_another = true;
            for (std::size_t ridx = 0u; ridx < ri.atom_rings.size(); ++ridx) {
                const auto& aring = ri.atom_rings[ridx];
                int here = 0;
                for (const auto a : aring) {
                    fixed[a] = false;
                    if (rs.atoms[a] > 0) {
                        ++here;
                    }
                }
                if (here > 1) {
                    continue;
                }
                if (here == 1) {
                    for (const auto a : aring) {
                        if (rs.atoms[a] > 0) {
                            --rs.atoms[a];
                            break;
                        }
                    }
                }
                for (const auto e : ri.bond_rings[ridx]) {
                    if (rs.bonds[e] > 0) {
                        --rs.bonds[e];
                    }
                }
            }
        }
        return need_another;
    };

    const auto run_rounds = [&]() {
        bool need_another = true;
        while (need_another) {
            const std::vector<int> ranks = rank_by_symbols(g, symbol);
            need_another = update_atoms(ranks);
        }
    };

    run_rounds();

    // RDKit's second pass (FindStereo.cpp:1163-1204): if any possible centre was
    // dropped, restore the original possible set, demote every non-fixed known
    // centre to a uniquely-labelled possible centre, and re-run. This lets a
    // specified centre that failed as "known" be reconsidered as an ordinary
    // possible (ring-dependent) centre, while the fixed specified survivors keep
    // their parity descriptors.
    if (possible != orig_possible) {
        possible = orig_possible;
        for (std::size_t i = 0u; i < n; ++i) {
            if (!fixed[i] && known[i]) {
                possible[i] = true;
                known[i] = false;
            }
            if (possible[i]) {
                symbol[i] += "|P" + std::to_string(i);
            }
        }
        rs = flag_ring_stereo(g, ri, candidate, stereo_bonds);
        run_rounds();
    }

    const std::vector<bool> stereo_bond_atoms = compute_stereo_bond_atoms(mol, g, ri);

    RDKitStereogenicity result;
    result.atom_is_potential_stereocenter.assign(n, false);
    result.atom_on_potential_stereo_bond.assign(n, false);
    for (std::size_t i = 0u; i < n; ++i) {
        result.atom_is_potential_stereocenter[i] = center[i] && tetrahedral[i];
        result.atom_on_potential_stereo_bond[i] = stereo_bond_atoms[i];
    }
    return result;
}

}  // namespace OEFP
