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

// Equitable-partition (colour) refinement reproducing the symmetry classes that
// RDKit's Canon::rankFragmentAtoms(breakTies=false) yields for FindStereo: only
// equivalence of ranks matters (the duplicate rule tests rank equality), so the
// class labels themselves are arbitrary. Initial colour is (degree, symbol),
// where the symbol is isotope+element+charge plus, for a candidate carrying a
// possible/known stereo tag, a unique per-atom suffix (RDKit's "_idx" label
// under cleanIt=false). Refinement then folds in each atom's sorted multiset of
// (bond key, neighbour class).
std::vector<int> refine_symmetry_classes(const HeavyGraph& g,
                                         const std::vector<bool>& labelled) {
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

    std::vector<std::string> sigs(n);
    for (std::size_t i = 0u; i < n; ++i) {
        std::string s = std::to_string(g.heavy_degree[i]) + '|' + std::to_string(g.isotope[i])
                        + '|' + std::to_string(g.atomic_num[i]) + '|'
                        + std::to_string(g.formal_charge[i]);
        if (labelled[i]) {
            s += "|P" + std::to_string(i);
        }
        sigs[i] = std::move(s);
    }
    auto [color, classes] = compress(sigs);

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

// RDKit flagRingStereo (FindStereo.cpp:625-746) for the atom path: commit a ring
// only when it holds >= 2 mutually-supporting candidates. Across-ring divisor-2
// (atom opposite in an even ring) and divisor-3 (three alternating positions of
// a ring whose size is a multiple of three) tests, plus a fused common-edge
// walk. There are no stereogenic double bonds in the atom path, so RDKit's
// "found by ring-stereo bond" branch is vacuous (bond count always zero).
struct RingStereo {
    std::vector<int> atoms;  // possibleRingStereoAtoms, per heavy atom
    std::vector<int> bonds;  // possibleRingStereoBonds, per edge id
};

RingStereo flag_ring_stereo(const HeavyGraph& g, const RingInfo& ri,
                            const std::vector<bool>& possible) {
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
                unsigned int found_by_atom = 0u;
                for (std::size_t step = inc; step < sz; step += inc) {
                    const std::size_t other = aring[(ai + step) % sz];
                    if (possible[other]) {
                        ++found_by_atom;
                    }
                }
                if (found_by_atom == divisor - 1u) {
                    here += 1;
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

// One end of a double bond has distinguishable substituents when it has a single
// non-partner heavy neighbour (trivially distinguishable) or two whose symmetry
// classes differ. Mirrors the CIP-rank comparison in the legacy
// findPotentialStereoBonds; plain symmetry classes stand in for assignAtomCIPRanks
// (equal iff topologically equivalent, which is exactly the "ranks differ" test).
bool double_bond_end_distinguishable(const HeavyGraph& g, std::size_t end,
                                     std::size_t partner, const std::vector<int>& ranks) {
    std::vector<int> subs;
    for (const auto nbr : g.adjacency[end]) {
        if (nbr != partner) {
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
// (any ring size). Unmarked ring double bonds are skipped entirely, at any size.
std::vector<bool> compute_stereo_bond_atoms(const OEChem::OEMolBase& mol, const HeavyGraph& g) {
    std::vector<bool> flags(g.atoms.size(), false);
    const std::vector<int> plain_ranks = refine_symmetry_classes(g, std::vector<bool>(g.atoms.size(), false));

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
            flag = specified;  // ring double bonds are otherwise ignored, any size
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

    // Candidates seeded into the ranking. Nontetrahedral centres participate in
    // the ranking (RDKit puts them in possibleAtoms) but only total-degree-4
    // ones survive the Atom_Tetrahedral output filter.
    std::vector<bool> possible(n, false);
    std::vector<bool> tetrahedral(n, false);
    for (std::size_t i = 0u; i < n; ++i) {
        const bool tet = is_potential_tetrahedral_center(g, ri, i);
        const bool nontet = is_potential_nontetrahedral_center(g, i);
        possible[i] = tet || nontet;
        tetrahedral[i] = tet || (nontet && (g.heavy_degree[i] + g.h_count[i]) == 4);
    }

    RingStereo rs = flag_ring_stereo(g, ri, possible);

    // Fixed-point disqualification. Each surviving candidate carries a unique
    // label (desymmetrising its own environment); a candidate whose neighbours
    // are indistinguishable is dropped unless a ring-stereo tie tolerates it.
    // Dropping a candidate cascades to its ring partners exactly as RDKit's
    // updateAtoms does, and the possible set only shrinks, so this converges.
    // (The cleanIt=false restore pass RDKit performs afterwards re-seeds the
    // same original candidates and re-runs the identical loop, so it is
    // redundant here.)
    while (true) {
        const std::vector<int> ranks = refine_symmetry_classes(g, possible);
        bool changed = false;
        for (std::size_t i = 0u; i < n; ++i) {
            if (!possible[i] || !has_disqualifying_duplicate(g, rs, i, ranks)) {
                continue;
            }
            possible[i] = false;
            changed = true;
            if (rs.atoms[i] == 0) {
                continue;
            }
            // Ring-stereo cascade: this atom no longer supports ring stereo, so
            // update every ring it shares. Rings dropping below two candidates
            // can no longer transmit stereo.
            rs.atoms[i] = 0;
            for (std::size_t ridx = 0u; ridx < ri.atom_rings.size(); ++ridx) {
                const auto& aring = ri.atom_rings[ridx];
                int here = 0;
                for (const auto a : aring) {
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
        if (!changed) {
            break;
        }
    }

    const std::vector<bool> stereo_bond_atoms = compute_stereo_bond_atoms(mol, g);

    RDKitStereogenicity result;
    result.atom_is_potential_stereocenter.assign(n, false);
    result.atom_on_potential_stereo_bond.assign(n, false);
    for (std::size_t i = 0u; i < n; ++i) {
        result.atom_is_potential_stereocenter[i] = possible[i] && tetrahedral[i];
        result.atom_on_potential_stereo_bond[i] = stereo_bond_atoms[i];
    }
    return result;
}

}  // namespace OEFP
