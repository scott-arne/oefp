#include "oefp/rdkit_stereogenicity.h"

#include "oefp/molecular_properties.h"

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
struct HeavyGraph {
    std::vector<const OEChem::OEAtomBase*> atoms;  // heavy atoms, iteration order
    std::vector<std::uint32_t> atomic_num;
    std::vector<std::uint32_t> isotope;
    std::vector<int> formal_charge;
    std::vector<int> heavy_degree;      // RDKit getAtomNonzeroDegree (heavy only)
    std::vector<int> h_count;           // RDKit getTotalNumHs
    std::vector<unsigned int> hyb;      // OEHybridization
    std::vector<int> explicit_valence;  // sum of heavy bond orders (0 H in the paths that read it)
    std::vector<bool> is_aromatic;

    // Heavy adjacency, with a per-edge bond key parallel to each neighbour list.
    std::vector<std::vector<std::size_t>> adjacency;
    std::vector<std::vector<int>> neighbor_bond_key;
};

// Encode a bond for the symmetry refinement. Aromatic bonds get their own code
// (RDKit orders them by the ":" symbol regardless of the kekulé order), every
// other bond is keyed by its integer order.
int bond_key(const OEChem::OEBondBase& bond) {
    if (bond.IsAromatic()) {
        return 100;
    }
    return static_cast<int>(bond.GetOrder());
}

HeavyGraph build_heavy_graph(const OEChem::OEMolBase& mol) {
    HeavyGraph g;
    std::unordered_map<unsigned int, std::size_t> heavy_index;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (is_hydrogen(*atom)) {
            continue;
        }
        heavy_index.emplace(atom->GetIdx(), g.atoms.size());
        g.atoms.push_back(&*atom);
        g.atomic_num.push_back(atom->GetAtomicNum());
        g.isotope.push_back(atom->GetIsotope());
        g.formal_charge.push_back(atom->GetFormalCharge());
        g.heavy_degree.push_back(static_cast<int>(atom->GetHvyDegree()));
        g.h_count.push_back(static_cast<int>(atom->GetTotalHCount()));
        g.hyb.push_back(atom->GetHyb());
        g.is_aromatic.push_back(atom->IsAromatic());

        int explicit_valence = 0;
        for (OESystem::OEIter<OEChem::OEBondBase> bond = atom->GetBonds(); bond; ++bond) {
            const auto* other = bond->GetNbr(&*atom);
            if (other != nullptr && !is_hydrogen(*other)) {
                explicit_valence += static_cast<int>(bond->GetOrder());
            }
        }
        g.explicit_valence.push_back(explicit_valence);
    }

    g.adjacency.resize(g.atoms.size());
    g.neighbor_bond_key.resize(g.atoms.size());
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
            continue;
        }
        const auto bi = heavy_index.find(begin->GetIdx());
        const auto ei = heavy_index.find(end->GetIdx());
        if (bi == heavy_index.end() || ei == heavy_index.end()) {
            continue;
        }
        const int key = bond_key(*bond);
        g.adjacency[bi->second].push_back(ei->second);
        g.neighbor_bond_key[bi->second].push_back(key);
        g.adjacency[ei->second].push_back(bi->second);
        g.neighbor_bond_key[ei->second].push_back(key);
    }
    return g;
}

// RDKit detail::isAtomPotentialTetrahedralCenter (FindStereo.cpp:50-114),
// remapped to hydrogen-agnostic heavy primitives. The three-coordinate N special
// case (in a 3-ring or bridgehead) is added in the para/ring sub-task and left
// out here, so classic centres are covered while the ring machinery is still
// absent.
bool is_potential_tetrahedral_center(const HeavyGraph& g, std::size_t i) {
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
        return false;
    }
    return false;
}

// Equitable-partition (colour) refinement reproducing the symmetry classes that
// RDKit's Canon::rankFragmentAtoms(breakTies=false) yields for FindStereo: only
// equivalence of ranks matters here (the duplicate rule tests rank equality), so
// the class labels themselves are arbitrary. Initial colour is (degree, symbol),
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

// Whether any two controlling (heavy) neighbours of atom i share a symmetry
// class. RDKit's ring-stereo tolerance is added in the para/ring sub-task; here
// any duplicate disqualifies the candidate.
bool has_duplicate_neighbor(const HeavyGraph& g, std::size_t i,
                            const std::vector<int>& ranks) {
    std::vector<int> seen;
    seen.reserve(g.adjacency[i].size());
    for (const auto nbr : g.adjacency[i]) {
        const int rnk = ranks[nbr];
        if (std::find(seen.begin(), seen.end(), rnk) != seen.end()) {
            return true;
        }
        seen.push_back(rnk);
    }
    return false;
}

}  // namespace

RDKitStereogenicity rdkit_potential_stereogenicity(const OEChem::OEMolBase& mol) {
    const HeavyGraph g = build_heavy_graph(mol);
    const std::size_t n = g.atoms.size();

    std::vector<bool> possible(n, false);
    for (std::size_t i = 0u; i < n; ++i) {
        if (is_potential_tetrahedral_center(g, i)) {
            possible[i] = true;
        }
    }

    // Fixed-point disqualification: give each surviving candidate a unique label
    // (so it desymmetrises its own environment), rank, then drop any candidate
    // whose neighbours are indistinguishable. Removing a label can only merge
    // classes, so the candidate set shrinks monotonically until it stabilises.
    while (true) {
        const std::vector<int> ranks = refine_symmetry_classes(g, possible);
        bool changed = false;
        for (std::size_t i = 0u; i < n; ++i) {
            if (possible[i] && has_duplicate_neighbor(g, i, ranks)) {
                possible[i] = false;
                changed = true;
            }
        }
        if (!changed) {
            break;
        }
    }

    RDKitStereogenicity result;
    result.atom_is_potential_stereocenter.assign(n, false);
    result.atom_on_potential_stereo_bond.assign(n, false);
    for (std::size_t i = 0u; i < n; ++i) {
        result.atom_is_potential_stereocenter[i] = possible[i];
    }
    return result;
}

}  // namespace OEFP
