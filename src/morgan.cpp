#include "oefp/morgan.h"
#include "oefp/stereo.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include <oesystem.h>

namespace OEFP {
namespace {

constexpr const char* MORGAN_COMPAT_VERSION = "Morgan-2026.03.1";
constexpr std::int32_t RDKIT_AROMATIC_BOND_TYPE = 12;
constexpr std::uint32_t OXYGEN_ATOMIC_NUM = 8u;
constexpr std::uint32_t CHLORINE_ATOMIC_NUM = 17u;
constexpr std::uint32_t BROMINE_ATOMIC_NUM = 35u;
constexpr std::uint32_t IODINE_ATOMIC_NUM = 53u;
constexpr std::uint32_t UNFOLDED_MORGAN_IDS = 0u;
constexpr std::uint64_t BITS_PER_WORD = 64u;

using Clock = std::chrono::steady_clock;

struct MorganEvent {
    std::uint32_t atom_id = 0;
    std::uint32_t radius = 0;
    std::uint32_t raw_id = 0;
    std::uint32_t bit_id = 0;
};

struct NeighborInvariant {
    std::int32_t bond = 0;
    std::uint32_t atom = 0;

    bool operator<(const NeighborInvariant& rhs) const {
        return std::tie(bond, atom) < std::tie(rhs.bond, rhs.atom);
    }
};

struct BondRecord {
    std::uint32_t index = 0;
    std::uint32_t sort_index = 0;
    std::uint32_t begin_atom = 0;
    std::uint32_t end_atom = 0;
    std::int32_t invariant = 1;
};

struct AtomRecord {
    const OEChem::OEAtomBase* atom = nullptr;
    std::uint32_t index = 0;
    std::uint32_t compact_index = 0;
    std::int32_t formal_charge_adjustment = 0;
    detail::AtomStereoLabel stereo_label = detail::AtomStereoLabel::None;
    std::vector<std::uint32_t> bond_indices;
};

struct MoleculeGraph {
    std::vector<AtomRecord> atoms;
    std::vector<BondRecord> bonds;
    std::vector<std::uint32_t> atom_id_to_compact;
};

using BondBitset = std::vector<std::uint64_t>;

struct BondBitsetHash {
    std::size_t operator()(const BondBitset& bits) const {
        std::size_t seed = 0;
        for (const auto word : bits) {
            seed ^= static_cast<std::size_t>(word) + 0x9e3779b97f4a7c15ULL + (seed << 6u)
                    + (seed >> 2u);
        }
        return seed;
    }
};

struct BondBitsetStorage {
    std::vector<std::uint64_t> words;
    std::size_t atom_count = 0;
    std::size_t words_per_atom = 0;

    BondBitsetStorage() = default;

    BondBitsetStorage(std::size_t atom_count_value, std::size_t bond_count)
        : words(
              atom_count_value * ((bond_count + BITS_PER_WORD - 1u) / BITS_PER_WORD),
              0ULL),
          atom_count(atom_count_value),
          words_per_atom((bond_count + BITS_PER_WORD - 1u) / BITS_PER_WORD) {
    }

    std::uint64_t* atom_words(std::uint32_t atom_id) {
        return words.data() + static_cast<std::size_t>(atom_id) * words_per_atom;
    }

    const std::uint64_t* atom_words(std::uint32_t atom_id) const {
        return words.data() + static_cast<std::size_t>(atom_id) * words_per_atom;
    }

    BondBitset copy_atom(std::uint32_t atom_id) const {
        if (words_per_atom == 0u) {
            return {};
        }
        const auto* begin = atom_words(atom_id);
        return BondBitset(begin, begin + words_per_atom);
    }
};

struct RoundEnvironment {
    std::uint32_t raw_id = 0;
    std::uint32_t atom_id = 0;
};

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

bool atom_neighborhood_less(
    const BondBitsetStorage& neighborhoods,
    std::uint32_t lhs_atom,
    std::uint32_t rhs_atom) {
    const auto* lhs = neighborhoods.atom_words(lhs_atom);
    const auto* rhs = neighborhoods.atom_words(rhs_atom);
    for (std::size_t reverse_index = neighborhoods.words_per_atom; reverse_index > 0u;
         --reverse_index) {
        const auto word_index = reverse_index - 1u;
        if (lhs[word_index] != rhs[word_index]) {
            return lhs[word_index] < rhs[word_index];
        }
    }
    return false;
}

bool round_environment_less(
    const BondBitsetStorage& neighborhoods,
    const RoundEnvironment& lhs,
    const RoundEnvironment& rhs) {
    if (atom_neighborhood_less(neighborhoods, lhs.atom_id, rhs.atom_id)) {
        return true;
    }
    if (atom_neighborhood_less(neighborhoods, rhs.atom_id, lhs.atom_id)) {
        return false;
    }
    return std::tie(lhs.raw_id, lhs.atom_id) < std::tie(rhs.raw_id, rhs.atom_id);
}

void add_bond_to_neighborhood(
    BondBitsetStorage& neighborhoods,
    std::uint32_t atom_id,
    std::uint32_t bond_id) {
    if (neighborhoods.words_per_atom == 0u) {
        return;
    }
    auto* words = neighborhoods.atom_words(atom_id);
    words[bond_id / BITS_PER_WORD] |= 1ULL << (bond_id % BITS_PER_WORD);
}

void add_neighborhood(
    BondBitsetStorage& neighborhoods,
    std::uint32_t atom_id,
    const BondBitsetStorage& additions,
    std::uint32_t other_atom_id) {
    if (neighborhoods.words_per_atom == 0u) {
        return;
    }
    auto* target = neighborhoods.atom_words(atom_id);
    const auto* source = additions.atom_words(other_atom_id);
    for (std::size_t word_index = 0; word_index < neighborhoods.words_per_atom; ++word_index) {
        target[word_index] |= source[word_index];
    }
}

void validate_options(const MorganOptions& options) {
    if (options.num_bits == 0) {
        throw std::invalid_argument("Morgan num_bits must be greater than zero.");
    }
    if (options.count_simulation && options.count_bounds.empty()) {
        throw std::invalid_argument("Morgan count_bounds cannot be empty when count simulation is enabled.");
    }
    if (options.count_simulation && options.count_bounds.size() >= options.num_bits) {
        throw std::invalid_argument("Morgan count_bounds size must be smaller than num_bits.");
    }
}

const char* bool_parameter(bool value) {
    return value ? "true" : "false";
}

std::string canonical_parameters(const MorganOptions& options) {
    std::ostringstream params;
    params << "radius=" << options.radius
           << ";num_bits=" << options.num_bits
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";use_features=" << bool_parameter(options.use_features)
           << ";use_bond_types=" << bool_parameter(options.use_bond_types)
           << ";only_nonzero_invariants=" << bool_parameter(options.only_nonzero_invariants)
           << ";include_ring_membership=" << bool_parameter(options.include_ring_membership)
           << ";include_redundant_environments="
           << bool_parameter(options.include_redundant_environments)
           << ";count_simulation=" << bool_parameter(options.count_simulation)
           << ";count_bounds=";
    for (std::size_t i = 0; i < options.count_bounds.size(); ++i) {
        if (i != 0u) {
            params << ',';
        }
        params << options.count_bounds[i];
    }
    return params.str();
}

std::string canonical_sparse_count_parameters(const MorganOptions& options) {
    std::ostringstream params;
    params << "radius=" << options.radius
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";use_features=" << bool_parameter(options.use_features)
           << ";use_bond_types=" << bool_parameter(options.use_bond_types)
           << ";only_nonzero_invariants=" << bool_parameter(options.only_nonzero_invariants)
           << ";include_ring_membership=" << bool_parameter(options.include_ring_membership)
           << ";include_redundant_environments="
           << bool_parameter(options.include_redundant_environments)
           << ";output=sparse_count";
    return params.str();
}

std::string canonical_sparse_binary_parameters(const MorganOptions& options) {
    std::ostringstream params;
    params << "radius=" << options.radius
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";use_features=" << bool_parameter(options.use_features)
           << ";use_bond_types=" << bool_parameter(options.use_bond_types)
           << ";only_nonzero_invariants=" << bool_parameter(options.only_nonzero_invariants)
           << ";include_ring_membership=" << bool_parameter(options.include_ring_membership)
           << ";include_redundant_environments="
           << bool_parameter(options.include_redundant_environments)
           << ";output=sparse_binary";
    return params.str();
}

std::string canonical_descriptor_parameters(const MorganOptions& options) {
    std::ostringstream params;
    params << "radius=" << options.radius
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";use_features=" << bool_parameter(options.use_features)
           << ";use_bond_types=" << bool_parameter(options.use_bond_types)
           << ";only_nonzero_invariants=" << bool_parameter(options.only_nonzero_invariants)
           << ";include_ring_membership=" << bool_parameter(options.include_ring_membership)
           << ";include_redundant_environments="
           << bool_parameter(options.include_redundant_environments)
           << ";output=descriptors";
    return params.str();
}

FingerprintSpec morgan_spec(const MorganOptions& options, FingerprintValueType value_type) {
    FingerprintSpec spec;
    spec.size_bits = options.num_bits;
    spec.value_type = value_type;
    spec.source_name = "RDKit-compatible";
    spec.source_type = "Morgan";
    spec.source_version = MORGAN_COMPAT_VERSION;
    spec.parameters = canonical_parameters(options);
    return spec;
}

FingerprintSpec morgan_sparse_count_spec(const MorganOptions& options) {
    FingerprintSpec spec = morgan_spec(options, FingerprintValueType::Counted);
    spec.size_bits = std::numeric_limits<std::uint64_t>::max();
    spec.parameters = canonical_sparse_count_parameters(options);
    return spec;
}

FingerprintSpec morgan_sparse_binary_spec(const MorganOptions& options) {
    FingerprintSpec spec = morgan_spec(options, FingerprintValueType::Binary);
    spec.size_bits = std::numeric_limits<std::uint64_t>::max();
    spec.parameters = canonical_sparse_binary_parameters(options);
    return spec;
}

DescriptorSpec morgan_descriptor_spec(const MorganOptions& options) {
    DescriptorSpec spec;
    spec.value_type = DescriptorValueType::Integer;
    spec.source_name = "OEFP";
    spec.source_type = "Morgan";
    spec.source_version = MORGAN_COMPAT_VERSION;
    spec.parameters = canonical_descriptor_parameters(options);
    return spec;
}

std::shared_ptr<const DescriptorSchema> morgan_descriptor_schema(const MorganOptions& options) {
    const auto spec = morgan_descriptor_spec(options);
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{
        "morgan",
        DescriptorValueKind::CountedIntegerKeys,
        "raw",
        spec.source_name,
        spec.source_type,
        spec.source_version,
        spec.parameters});
    return builder.Build();
}

std::uint32_t event_bit_id(std::uint32_t raw_id, std::uint32_t fold_size) {
    return fold_size == UNFOLDED_MORGAN_IDS ? raw_id : raw_id % fold_size;
}

std::uint32_t hash_combine_value(std::uint32_t seed, std::uint32_t hashed_value) {
    seed ^= hashed_value + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
    return seed;
}

std::uint32_t hash_value(std::uint32_t value) {
    return value;
}

std::uint32_t hash_value(std::int32_t value) {
    return static_cast<std::uint32_t>(value);
}

std::uint32_t combine_hash(std::uint32_t seed, std::uint32_t value) {
    return hash_combine_value(seed, hash_value(value));
}

std::uint32_t hash_value(const NeighborInvariant& value) {
    std::uint32_t seed = 0;
    seed = hash_combine_value(seed, hash_value(value.bond));
    seed = hash_combine_value(seed, hash_value(value.atom));
    return seed;
}

std::uint32_t combine_hash(std::uint32_t seed, const NeighborInvariant& value) {
    return hash_combine_value(seed, hash_value(value));
}

std::uint32_t vector_hash(const std::vector<std::uint32_t>& values) {
    std::uint32_t seed = 0;
    for (const auto value : values) {
        seed = combine_hash(seed, value);
    }
    return seed;
}

std::int32_t isotope_delta(const OEChem::OEAtomBase& atom) {
    const auto isotope = atom.GetIsotope();
    if (isotope == 0) {
        return 0;
    }
    const auto isotope_weight = OEChem::OEGetIsotopicWeight(atom.GetAtomicNum(), isotope);
    const auto average_weight = OEChem::OEGetAverageWeight(atom.GetAtomicNum());
    return static_cast<std::int32_t>(isotope_weight - average_weight);
}

bool is_rdkit_halogen_oxide_center(const AtomRecord& record) {
    if (record.atom == nullptr || record.atom->GetFormalCharge() != 0) {
        return false;
    }
    const auto atomic_num = record.atom->GetAtomicNum();
    return atomic_num == CHLORINE_ATOMIC_NUM || atomic_num == BROMINE_ATOMIC_NUM
           || atomic_num == IODINE_ATOMIC_NUM;
}

bool is_rdkit_halogen_oxide_oxygen(const AtomRecord& record) {
    return record.atom != nullptr && record.atom->GetAtomicNum() == OXYGEN_ATOMIC_NUM
           && record.atom->GetFormalCharge() == 0 && record.atom->GetTotalHCount() == 0
           && record.atom->GetExplicitDegree() == 1;
}

bool rdkit_normalizes_halogen_oxide_bond(
    const AtomRecord& begin,
    const AtomRecord& end,
    const OEChem::OEBondBase& bond) {
    if (bond.GetOrder() != 2 || bond.IsAromatic()) {
        return false;
    }
    return (is_rdkit_halogen_oxide_center(begin) && is_rdkit_halogen_oxide_oxygen(end))
           || (is_rdkit_halogen_oxide_center(end) && is_rdkit_halogen_oxide_oxygen(begin));
}

void apply_rdkit_halogen_oxide_normalization(
    AtomRecord& begin,
    AtomRecord& end,
    const OEChem::OEBondBase& bond) {
    if (!rdkit_normalizes_halogen_oxide_bond(begin, end, bond)) {
        return;
    }

    // RDKit sanitization rewrites neutral halogen oxides such as OCl(=O)=O
    // into charge-separated single bonds. OpenEye keeps the input valence
    // form, so adjust only the Morgan compatibility invariants here.
    if (is_rdkit_halogen_oxide_center(begin)) {
        ++begin.formal_charge_adjustment;
        --end.formal_charge_adjustment;
    } else {
        --begin.formal_charge_adjustment;
        ++end.formal_charge_adjustment;
    }
}

// RDKit's GetFeatureInvariants uses a fixed six-feature Gobbi SMARTS set. The
// FCFP atom seed is the OR of the matched feature bits, used directly (unhashed).
// Bit order matches RDKit: Donor, Acceptor, Aromatic, Halogen, Basic, Acidic.
struct FeaturePattern {
    std::uint32_t bit;
    const char* smarts;
};

const FeaturePattern FEATURE_PATTERNS[] = {
    {1u, "[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]"},
    {2u, "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"},
    {4u, "[a]"},
    {8u, "[F,Cl,Br,I]"},
    {16u, "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]"},
    {32u, "[$([C,S](=[O,S,P])-[O;H1,-1])]"},
};

// Compute a per-compact-index pharmacophore feature mask for the molecule.
//
// OEPrepareSearch requires a mutable molecule, so matching runs against a local
// copy. The copy preserves OpenEye atom indices, so matched target indices map
// back through graph.atom_id_to_compact exactly as the rest of build_graph does.
std::vector<std::uint32_t> compute_feature_invariants(
    const OEChem::OEMolBase& mol,
    const MoleculeGraph& graph) {
    std::vector<std::uint32_t> masks(graph.atoms.size(), 0u);
    OEChem::OEGraphMol search_mol(mol);
    for (const auto& pattern : FEATURE_PATTERNS) {
        OEChem::OESubSearch subsearch(pattern.smarts);
        if (!subsearch) {
            throw std::runtime_error(
                std::string("Invalid OEFP feature SMARTS: ") + pattern.smarts);
        }
        // Feature aromaticity is perceived on the search molecule by
        // OEPrepareSearch; this is intentionally independent of the IsAromatic()
        // reads in the connectivity-invariant path, and matches RDKit's feature
        // invariant model. Do not couple the two perceptions.
        OEChem::OEPrepareSearch(search_mol, subsearch);
        for (OESystem::OEIter<OEChem::OEMatchBase> match = subsearch.Match(search_mol, true);
             match;
             ++match) {
            for (OESystem::OEIter<OEChem::OEMatchPair<OEChem::OEAtomBase>> mp =
                     match->GetAtoms();
                 mp; ++mp) {
                const auto target_idx = mp->target->GetIdx();
                if (target_idx < graph.atom_id_to_compact.size()) {
                    const auto compact = graph.atom_id_to_compact[target_idx];
                    if (compact < masks.size()) {
                        masks[compact] |= pattern.bit;
                    }
                }
            }
        }
    }
    return masks;
}

std::uint32_t atom_invariant(const AtomRecord& atom_record, const MorganOptions& options) {
    const auto& atom = *atom_record.atom;
    std::uint32_t invariant = 0;
    invariant = combine_hash(invariant, atom.GetAtomicNum());
    invariant = combine_hash(invariant, atom.GetDegree());
    invariant = combine_hash(invariant, atom.GetTotalHCount());
    invariant = combine_hash(
        invariant,
        static_cast<std::uint32_t>(
            atom.GetFormalCharge() + atom_record.formal_charge_adjustment));
    invariant = combine_hash(invariant, static_cast<std::uint32_t>(isotope_delta(atom)));
    if (options.include_ring_membership && atom.IsInRing()) {
        invariant = combine_hash(invariant, 1u);
    }
    return invariant;
}

std::int32_t bond_type_value(const OEChem::OEBondBase& bond) {
    if (bond.IsAromatic()) {
        return RDKIT_AROMATIC_BOND_TYPE;
    }
    return static_cast<std::int32_t>(bond.GetOrder());
}

std::int32_t bond_invariant(
    const OEChem::OEMolBase& mol,
    const OEChem::OEBondBase& bond,
    const AtomRecord& begin,
    const AtomRecord& end,
    const MorganOptions& options) {
    if (!options.use_bond_types) {
        return 1;
    }
    if (rdkit_normalizes_halogen_oxide_bond(begin, end, bond)) {
        return 1;
    }
    if (options.use_chirality && bond.GetOrder() == 2 && !bond.IsAromatic()) {
        const auto stereo =
            detail::MorganDoubleBondStereoValue(detail::PerceiveBondStereo(mol, bond));
        if (stereo != 0) {
            return 100 + 10 * bond_type_value(bond) + stereo;
        }
    }
    return bond_type_value(bond);
}

void assign_rdkit_bond_order(MoleculeGraph& graph, const std::vector<std::uint32_t>& bond_ids) {
    const auto missing_bond = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> parent_bonds(graph.atoms.size(), missing_bond);
    std::vector<std::uint32_t> parent_atoms(graph.atoms.size(), 0u);

    for (const auto bond_id : bond_ids) {
        const auto& bond = graph.bonds[bond_id];
        const auto child = std::max(bond.begin_atom, bond.end_atom);
        const auto parent = std::min(bond.begin_atom, bond.end_atom);
        if (child >= parent_bonds.size()) {
            continue;
        }
        if (parent_bonds[child] == missing_bond || parent > parent_atoms[child]) {
            parent_bonds[child] = bond_id;
            parent_atoms[child] = parent;
        }
    }

    std::vector<std::tuple<std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t>>
        ordered_bonds;
    ordered_bonds.reserve(bond_ids.size());
    for (const auto bond_id : bond_ids) {
        const auto& bond = graph.bonds[bond_id];
        const auto child = std::max(bond.begin_atom, bond.end_atom);
        const auto parent = std::min(bond.begin_atom, bond.end_atom);
        const auto order_group =
            child < parent_bonds.size() && parent_bonds[child] == bond_id ? 0u : 1u;
        ordered_bonds.emplace_back(order_group, child, parent, bond_id);
    }

    // RDKit numbers parent/tree bonds before ring-closure bonds. OpenEye can
    // expose closure bonds first, so use a separate sort index for provenance.
    std::sort(ordered_bonds.begin(), ordered_bonds.end());
    for (std::uint32_t order = 0; order < ordered_bonds.size(); ++order) {
        graph.bonds[std::get<3>(ordered_bonds[order])].sort_index = order;
    }
}

MoleculeGraph build_graph(const OEChem::OEMolBase& mol, const MorganOptions& options) {
    MoleculeGraph graph;
    const auto missing_index = std::numeric_limits<std::uint32_t>::max();
    graph.atoms.reserve(mol.NumAtoms());
    graph.bonds.reserve(mol.NumBonds());
    graph.atom_id_to_compact.assign(mol.GetMaxAtomIdx(), missing_index);

    std::vector<const OEChem::OEAtomBase*> atoms;
    atoms.reserve(mol.NumAtoms());
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        atoms.push_back(atom);
    }
    std::sort(
        atoms.begin(),
        atoms.end(),
        [](const OEChem::OEAtomBase* lhs, const OEChem::OEAtomBase* rhs) {
            return lhs->GetIdx() < rhs->GetIdx();
        });

    for (const auto* atom : atoms) {
        const auto idx = atom->GetIdx();
        if (idx >= graph.atom_id_to_compact.size()) {
            throw std::runtime_error("OpenEye atom index exceeds molecule atom storage.");
        }
        const auto compact_index = static_cast<std::uint32_t>(graph.atoms.size());
        graph.atom_id_to_compact[idx] = compact_index;
        AtomRecord record;
        record.atom = atom;
        record.index = idx;
        record.compact_index = compact_index;
        if (options.use_chirality) {
            record.stereo_label = detail::PerceiveAtomStereo(mol, *atom);
        }
        record.bond_indices.reserve(atom->GetDegree());
        graph.atoms.push_back(std::move(record));
    }

    std::vector<std::uint32_t> bond_ids;
    bond_ids.reserve(mol.NumBonds());
    std::vector<const OEChem::OEBondBase*> bonds;
    bonds.reserve(mol.NumBonds());
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        bonds.push_back(bond);
    }
    std::sort(
        bonds.begin(),
        bonds.end(),
        [](const OEChem::OEBondBase* lhs, const OEChem::OEBondBase* rhs) {
            return lhs->GetIdx() < rhs->GetIdx();
        });

    for (const auto* bond : bonds) {
        const auto idx = bond->GetIdx();
        const auto begin_idx = bond->GetBgnIdx();
        const auto end_idx = bond->GetEndIdx();
        if (begin_idx >= graph.atom_id_to_compact.size()
            || end_idx >= graph.atom_id_to_compact.size()
            || graph.atom_id_to_compact[begin_idx] == missing_index
            || graph.atom_id_to_compact[end_idx] == missing_index) {
            throw std::runtime_error("OpenEye bond references an atom outside molecule storage.");
        }
        const auto compact_bond_id = static_cast<std::uint32_t>(graph.bonds.size());
        bond_ids.push_back(compact_bond_id);
        BondRecord record;
        record.index = idx;
        record.begin_atom = graph.atom_id_to_compact[begin_idx];
        record.end_atom = graph.atom_id_to_compact[end_idx];
        auto& begin = graph.atoms[record.begin_atom];
        auto& end = graph.atoms[record.end_atom];
        apply_rdkit_halogen_oxide_normalization(begin, end, *bond);
        record.invariant = bond_invariant(mol, *bond, begin, end, options);
        graph.atoms[record.begin_atom].bond_indices.push_back(compact_bond_id);
        graph.atoms[record.end_atom].bond_indices.push_back(compact_bond_id);
        graph.bonds.push_back(record);
    }

    assign_rdkit_bond_order(graph, bond_ids);
    return graph;
}

std::uint32_t other_atom(const BondRecord& bond, std::uint32_t atom_id) {
    return bond.begin_atom == atom_id ? bond.end_atom : bond.begin_atom;
}

std::vector<std::uint32_t> atom_iteration_order(
    const std::vector<std::uint32_t>& current_invariants,
    const MorganOptions& options) {
    std::vector<std::uint32_t> atom_order(current_invariants.size());
    for (std::uint32_t i = 0; i < atom_order.size(); ++i) {
        atom_order[i] = i;
    }
    if (!options.only_nonzero_invariants) {
        return atom_order;
    }

    std::sort(
        atom_order.begin(),
        atom_order.end(),
        [&current_invariants](std::uint32_t lhs, std::uint32_t rhs) {
            const auto lhs_zero = current_invariants[lhs] == 0u ? 1u : 0u;
            const auto rhs_zero = current_invariants[rhs] == 0u ? 1u : 0u;
            return std::tie(lhs_zero, lhs) < std::tie(rhs_zero, rhs);
        });
    return atom_order;
}

template <typename EventSink>
void enumerate_events_into(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options,
    std::uint32_t fold_size,
    EventSink&& emit_event,
    MorganGenerationProfile* profile = nullptr) {
    const auto graph_start = Clock::now();
    const auto graph = build_graph(mol, options);
    const auto graph_end = Clock::now();
    if (profile != nullptr) {
        profile->graph_seconds += elapsed_seconds(graph_start, graph_end);
        profile->atom_count = static_cast<std::uint32_t>(mol.NumAtoms());
        profile->bond_count = static_cast<std::uint32_t>(mol.NumBonds());
    }

    const auto invariant_start = Clock::now();
    std::vector<std::uint32_t> atom_invariants(graph.atoms.size(), 0u);
    std::vector<std::uint32_t> feature_masks;
    if (options.use_features) {
        feature_masks = compute_feature_invariants(mol, graph);
    }
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom != nullptr) {
            atom_invariants[atom_record.compact_index] =
                options.use_features
                    ? feature_masks[atom_record.compact_index]
                    : atom_invariant(atom_record, options);
        }
    }

    std::vector<std::uint32_t> current = atom_invariants;
    std::vector<std::uint32_t> next(graph.atoms.size(), 0u);
    BondBitsetStorage neighborhoods(graph.atoms.size(), graph.bonds.size());
    BondBitsetStorage round_neighborhoods(graph.atoms.size(), graph.bonds.size());
    std::vector<bool> dead_atoms(graph.atoms.size(), false);
    std::vector<bool> chiral_atoms(graph.atoms.size(), false);
    std::unordered_set<std::uint64_t> seen_single_word_neighborhoods;
    std::unordered_set<BondBitset, BondBitsetHash> seen_multi_word_neighborhoods;
    const auto atom_order = atom_iteration_order(current, options);
    const auto invariant_end = Clock::now();
    if (profile != nullptr) {
        profile->invariant_seconds += elapsed_seconds(invariant_start, invariant_end);
    }

    const auto radius_zero_start = Clock::now();
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom == nullptr) {
            continue;
        }
        if (!options.only_nonzero_invariants || current[atom_record.compact_index] != 0u) {
            if (profile != nullptr) {
                ++profile->event_count;
            }
            emit_event(
                MorganEvent{atom_record.index, 0u, current[atom_record.compact_index],
                            event_bit_id(current[atom_record.compact_index], fold_size)});
        }
    }
    const auto radius_zero_end = Clock::now();
    if (profile != nullptr) {
        profile->radius_zero_seconds += elapsed_seconds(radius_zero_start, radius_zero_end);
    }

    for (std::uint32_t layer = 0; layer < options.radius; ++layer) {
        std::vector<RoundEnvironment> this_round;
        this_round.reserve(graph.atoms.size());
        std::vector<NeighborInvariant> neighbors;

        const auto neighborhood_start = Clock::now();
        for (const auto atom_id : atom_order) {
            if (atom_id >= graph.atoms.size()) {
                continue;
            }
            const auto& atom_record = graph.atoms[atom_id];
            if (atom_record.atom == nullptr || dead_atoms[atom_id]) {
                continue;
            }
            if (atom_record.bond_indices.empty()) {
                dead_atoms[atom_id] = true;
                continue;
            }

            neighbors.clear();
            neighbors.reserve(atom_record.bond_indices.size());

            for (const auto bond_id : atom_record.bond_indices) {
                const auto& bond = graph.bonds[bond_id];
                add_bond_to_neighborhood(round_neighborhoods, atom_id, bond.sort_index);
                const auto nbr = other_atom(bond, atom_id);
                add_neighborhood(round_neighborhoods, atom_id, neighborhoods, nbr);
                neighbors.push_back(NeighborInvariant{bond.invariant, current[nbr]});
            }

            std::sort(neighbors.begin(), neighbors.end());

            auto looks_chiral = options.use_chirality
                                && atom_record.stereo_label != detail::AtomStereoLabel::None;
            std::uint32_t invariant = layer;
            invariant = combine_hash(invariant, current[atom_id]);
            for (std::size_t neighbor_index = 0; neighbor_index < neighbors.size();
                 ++neighbor_index) {
                const auto& neighbor = neighbors[neighbor_index];
                invariant = combine_hash(invariant, neighbor);
                if (looks_chiral && !chiral_atoms[atom_id]) {
                    if (neighbor.bond != 1) {
                        looks_chiral = false;
                    } else if (
                        neighbor_index != 0u
                        && neighbor.atom == neighbors[neighbor_index - 1u].atom) {
                        looks_chiral = false;
                    }
                }
            }
            if (looks_chiral) {
                chiral_atoms[atom_id] = true;
                invariant = combine_hash(
                    invariant,
                    detail::MorganAtomChiralityValue(atom_record.stereo_label));
            }

            next[atom_id] = invariant;
            this_round.push_back(RoundEnvironment{invariant, atom_id});
        }
        const auto neighborhood_end = Clock::now();
        if (profile != nullptr) {
            profile->neighborhood_seconds +=
                elapsed_seconds(neighborhood_start, neighborhood_end);
        }

        // RDKit sorts boost::dynamic_bitset neighborhoods before duplicate
        // suppression, so compare the stored bond ids from the highest bit down.
        const auto duplicate_start = Clock::now();
        std::sort(
            this_round.begin(),
            this_round.end(),
            [&round_neighborhoods](const RoundEnvironment& lhs, const RoundEnvironment& rhs) {
                return round_environment_less(round_neighborhoods, lhs, rhs);
            });
        for (const auto& item : this_round) {
            const auto raw_id = item.raw_id;
            const auto atom_id = item.atom_id;
            const auto output_atom_id = graph.atoms[atom_id].index;
            bool seen = false;
            BondBitset multi_word_neighborhood;
            std::uint64_t single_word_neighborhood = 0ULL;
            if (!options.include_redundant_environments) {
                if (round_neighborhoods.words_per_atom == 1u) {
                    single_word_neighborhood = *round_neighborhoods.atom_words(atom_id);
                    seen = seen_single_word_neighborhoods.count(single_word_neighborhood) != 0u;
                } else {
                    multi_word_neighborhood = round_neighborhoods.copy_atom(atom_id);
                    seen = seen_multi_word_neighborhoods.count(multi_word_neighborhood) != 0u;
                }
            }

            if (options.include_redundant_environments || !seen) {
                if (!options.only_nonzero_invariants || atom_invariants[atom_id] != 0u) {
                    if (profile != nullptr) {
                        ++profile->event_count;
                    }
                    emit_event(MorganEvent{
                        output_atom_id,
                        layer + 1u,
                        raw_id,
                        event_bit_id(raw_id, fold_size),
                    });
                    if (!options.include_redundant_environments) {
                        if (round_neighborhoods.words_per_atom == 1u) {
                            seen_single_word_neighborhoods.insert(single_word_neighborhood);
                        } else {
                            seen_multi_word_neighborhoods.insert(std::move(multi_word_neighborhood));
                        }
                    }
                }
            } else {
                dead_atoms[atom_id] = true;
            }
        }
        const auto duplicate_end = Clock::now();
        if (profile != nullptr) {
            profile->duplicate_seconds += elapsed_seconds(duplicate_start, duplicate_end);
        }

        current.swap(next);
        std::fill(next.begin(), next.end(), 0u);
        neighborhoods.words = round_neighborhoods.words;
    }
}

std::vector<MorganEvent> enumerate_events(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options,
    std::uint32_t fold_size) {
    std::vector<MorganEvent> events;
    events.reserve((static_cast<std::size_t>(options.radius) + 1u) * mol.NumAtoms());
    enumerate_events_into(
        mol,
        options,
        fold_size,
        [&events](MorganEvent event) {
            events.push_back(event);
        });

    return events;
}

OEFP make_count_simulated_fingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    const auto bound_count = options.count_bounds.size();
    const auto effective_size = options.num_bits / bound_count;
    std::map<std::uint32_t, std::uint32_t> effective_counts;
    for (const auto& event : enumerate_events(mol, options, effective_size)) {
        auto& count = effective_counts[event.bit_id];
        if (count == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Morgan count simulation count exceeds uint32 storage.");
        }
        ++count;
    }

    OEFP fingerprint(morgan_spec(options, FingerprintValueType::Binary));
    for (const auto& [base_bit, count] : effective_counts) {
        for (std::size_t i = 0; i < bound_count; ++i) {
            if (count >= options.count_bounds[i]) {
                fingerprint.SetBit(
                    static_cast<std::uint64_t>(base_bit) * bound_count
                    + static_cast<std::uint64_t>(i));
            }
        }
    }
    return fingerprint;
}

OEFP make_binary_fingerprint_from_events(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options,
    const FingerprintSpec& spec) {
    std::vector<std::uint64_t> words(DenseWordCount(spec.size_bits), 0ULL);
    enumerate_events_into(
        mol,
        options,
        options.num_bits,
        [&words](const MorganEvent& event) {
            words[event.bit_id / BITS_PER_WORD] |= 1ULL << (event.bit_id % BITS_PER_WORD);
        });
    return OEFP(spec, std::move(words));
}

OEFP make_profiled_fingerprint_from_events(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options,
    MorganGenerationProfile& profile) {
    const auto spec = morgan_spec(options, FingerprintValueType::Binary);
    const auto fold_size =
        options.count_simulation
            ? options.num_bits / static_cast<std::uint32_t>(options.count_bounds.size())
            : options.num_bits;
    std::vector<MorganEvent> events;
    events.reserve((static_cast<std::size_t>(options.radius) + 1u) * mol.NumAtoms());
    enumerate_events_into(
        mol,
        options,
        fold_size,
        [&events](MorganEvent event) {
            events.push_back(event);
        },
        &profile);

    const auto bit_folding_start = Clock::now();
    std::vector<std::uint64_t> words(DenseWordCount(spec.size_bits), 0ULL);
    if (options.count_simulation) {
        const auto bound_count = options.count_bounds.size();
        std::map<std::uint32_t, std::uint32_t> effective_counts;
        for (const auto& event : events) {
            auto& count = effective_counts[event.bit_id];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("Morgan count simulation count exceeds uint32 storage.");
            }
            ++count;
        }
        for (const auto& [base_bit, count] : effective_counts) {
            for (std::size_t i = 0; i < bound_count; ++i) {
                if (count >= options.count_bounds[i]) {
                    const auto bit_id =
                        static_cast<std::uint64_t>(base_bit) * bound_count
                        + static_cast<std::uint64_t>(i);
                    words[bit_id / BITS_PER_WORD] |= 1ULL << (bit_id % BITS_PER_WORD);
                }
            }
        }
    } else {
        for (const auto& event : events) {
            words[event.bit_id / BITS_PER_WORD] |= 1ULL << (event.bit_id % BITS_PER_WORD);
        }
    }
    const auto fingerprint = OEFP(spec, std::move(words));
    const auto bit_folding_end = Clock::now();
    profile.bit_folding_seconds += elapsed_seconds(bit_folding_start, bit_folding_end);
    profile.on_bit_count = static_cast<std::uint32_t>(fingerprint.CountOnBits());
    return fingerprint;
}

MorganFingerprintResult make_count_simulated_fingerprint_with_mapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    const auto bound_count = options.count_bounds.size();
    const auto effective_size = options.num_bits / bound_count;
    std::map<std::uint32_t, std::vector<MorganEvent>> events_by_base_bit;
    for (const auto& event : enumerate_events(mol, options, effective_size)) {
        auto& events = events_by_base_bit[event.bit_id];
        if (events.size() == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Morgan count simulation count exceeds uint32 storage.");
        }
        events.push_back(event);
    }

    OEFP fingerprint(morgan_spec(options, FingerprintValueType::Binary));
    OEFPMappingSet mapping;
    for (const auto& [base_bit, events] : events_by_base_bit) {
        const auto count = static_cast<std::uint32_t>(events.size());
        for (std::size_t i = 0; i < bound_count; ++i) {
            if (count >= options.count_bounds[i]) {
                const auto simulated_bit =
                    static_cast<std::uint64_t>(base_bit) * bound_count
                    + static_cast<std::uint64_t>(i);
                fingerprint.SetBit(simulated_bit);
                for (const auto& event : events) {
                    mapping.AddEnvironmentMapping(
                        0,
                        simulated_bit,
                        event.atom_id,
                        event.radius);
                }
            }
        }
    }
    return MorganFingerprintResult(fingerprint, mapping);
}

OEFPMappingSet mapping_from_events(const std::vector<MorganEvent>& events) {
    OEFPMappingSet mapping;
    for (const auto& event : events) {
        mapping.AddEnvironmentMapping(0, event.bit_id, event.atom_id, event.radius);
    }
    return mapping;
}

OEFPCount count_fingerprint_from_events(
    FingerprintSpec spec,
    const std::vector<MorganEvent>& events,
    const char* overflow_message) {
    std::map<std::uint32_t, std::uint32_t> folded_counts;
    for (const auto& event : events) {
        auto& count = folded_counts[event.bit_id];
        if (count == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(overflow_message);
        }
        ++count;
    }

    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> counts;
    indices.reserve(folded_counts.size());
    counts.reserve(folded_counts.size());
    for (const auto& [index, count] : folded_counts) {
        indices.push_back(index);
        counts.push_back(count);
    }

    return OEFPCount(std::move(spec), std::move(indices), std::move(counts));
}

DescriptorSet descriptors_from_events(
    std::shared_ptr<const DescriptorSchema> schema,
    const std::vector<MorganEvent>& events,
    const char* overflow_message) {
    std::map<std::uint32_t, std::uint32_t> raw_counts;
    for (const auto& event : events) {
        auto& count = raw_counts[event.raw_id];
        if (count == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(overflow_message);
        }
        ++count;
    }

    std::vector<std::int64_t> keys;
    std::vector<std::uint32_t> counts;
    keys.reserve(raw_counts.size());
    counts.reserve(raw_counts.size());
    for (const auto& [raw_id, count] : raw_counts) {
        keys.push_back(static_cast<std::int64_t>(raw_id));
        counts.push_back(count);
    }

    DescriptorSetBuilder builder(std::move(schema));
    builder.Set("morgan", DescriptorValue::CountedIntegerKeys(std::move(keys), std::move(counts)));
    return builder.Build();
}

} // namespace

double MorganGenerationProfile::TotalSeconds() const {
    return graph_seconds + invariant_seconds + radius_zero_seconds + neighborhood_seconds
           + duplicate_seconds + bit_folding_seconds;
}

MorganFingerprintResult::MorganFingerprintResult(OEFP fingerprint, OEFPMappingSet mapping)
    : fingerprint_(std::move(fingerprint)), mapping_(std::move(mapping)) {
}

OEFP MorganFingerprintResult::Fingerprint() const {
    return fingerprint_;
}

OEFPMappingSet MorganFingerprintResult::Mapping() const {
    return mapping_;
}

MorganSparseFingerprintResult::MorganSparseFingerprintResult(
    OEFPSparse fingerprint,
    OEFPMappingSet mapping)
    : fingerprint_(std::move(fingerprint)), mapping_(std::move(mapping)) {
}

OEFPSparse MorganSparseFingerprintResult::Fingerprint() const {
    return fingerprint_;
}

OEFPMappingSet MorganSparseFingerprintResult::Mapping() const {
    return mapping_;
}

MorganCountFingerprintResult::MorganCountFingerprintResult(
    OEFPCount fingerprint,
    OEFPMappingSet mapping)
    : fingerprint_(std::move(fingerprint)), mapping_(std::move(mapping)) {
}

OEFPCount MorganCountFingerprintResult::Fingerprint() const {
    return fingerprint_;
}

OEFPMappingSet MorganCountFingerprintResult::Mapping() const {
    return mapping_;
}

MorganSparseCountFingerprintResult::MorganSparseCountFingerprintResult(
    OEFPCount fingerprint,
    OEFPMappingSet mapping)
    : fingerprint_(std::move(fingerprint)), mapping_(std::move(mapping)) {
}

OEFPCount MorganSparseCountFingerprintResult::Fingerprint() const {
    return fingerprint_;
}

OEFPMappingSet MorganSparseCountFingerprintResult::Mapping() const {
    return mapping_;
}

MorganGenerator::MorganGenerator(MorganOptions options)
    : options_(std::move(options)) {
    validate_options(options_);
    binary_spec_ = morgan_spec(options_, FingerprintValueType::Binary);
}

OEFP MorganGenerator::Fingerprint(const OEChem::OEMolBase& mol) const {
    if (options_.count_simulation) {
        return make_count_simulated_fingerprint(mol, options_);
    }
    return make_binary_fingerprint_from_events(mol, options_, binary_spec_);
}

const MorganOptions& MorganGenerator::Options() const {
    return options_;
}

OEFP MakeMorganFingerprint(const OEChem::OEMolBase& mol, const MorganOptions& options) {
    return MorganGenerator(options).Fingerprint(mol);
}

MorganFingerprintResult MakeMorganFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        return make_count_simulated_fingerprint_with_mapping(mol, options);
    }

    const auto events = enumerate_events(mol, options, options.num_bits);
    OEFP fingerprint(morgan_spec(options, FingerprintValueType::Binary));
    for (const auto& event : events) {
        fingerprint.SetBit(event.bit_id);
    }

    return MorganFingerprintResult(fingerprint, mapping_from_events(events));
}

OEFPCount MakeMorganCountFingerprint(const OEChem::OEMolBase& mol, const MorganOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        throw std::invalid_argument("Morgan count simulation is only supported for binary fingerprints.");
    }

    const auto events = enumerate_events(mol, options, options.num_bits);
    return count_fingerprint_from_events(
        morgan_spec(options, FingerprintValueType::Counted),
        events,
        "Morgan count fingerprint count exceeds uint32 storage.");
}

MorganCountFingerprintResult MakeMorganCountFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        throw std::invalid_argument("Morgan count simulation is only supported for binary fingerprints.");
    }

    const auto events = enumerate_events(mol, options, options.num_bits);
    return MorganCountFingerprintResult(
        count_fingerprint_from_events(
            morgan_spec(options, FingerprintValueType::Counted),
            events,
            "Morgan count fingerprint count exceeds uint32 storage."),
        mapping_from_events(events));
}

OEFPCount MakeMorganSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        throw std::invalid_argument("Morgan count simulation is only supported for binary fingerprints.");
    }

    const auto events = enumerate_events(mol, options, UNFOLDED_MORGAN_IDS);
    return count_fingerprint_from_events(
        morgan_sparse_count_spec(options),
        events,
        "Morgan sparse count fingerprint count exceeds uint32 storage.");
}

MorganSparseCountFingerprintResult MakeMorganSparseCountFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        throw std::invalid_argument("Morgan count simulation is only supported for binary fingerprints.");
    }

    const auto events = enumerate_events(mol, options, UNFOLDED_MORGAN_IDS);
    return MorganSparseCountFingerprintResult(
        count_fingerprint_from_events(
            morgan_sparse_count_spec(options),
            events,
            "Morgan sparse count fingerprint count exceeds uint32 storage."),
        mapping_from_events(events));
}

DescriptorSet MakeMorganDescriptors(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        throw std::invalid_argument("Morgan count simulation is only supported for binary fingerprints.");
    }

    const auto events = enumerate_events(mol, options, UNFOLDED_MORGAN_IDS);
    return descriptors_from_events(
        morgan_descriptor_schema(options),
        events,
        "Morgan descriptor count exceeds uint32 storage.");
}

OEFPSparse MakeMorganSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        throw std::invalid_argument("Morgan count simulation is only supported for binary fingerprints.");
    }

    std::set<std::uint32_t> raw_ids;
    for (const auto& event : enumerate_events(mol, options, UNFOLDED_MORGAN_IDS)) {
        raw_ids.insert(event.raw_id);
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(raw_ids.size());
    for (const auto raw_id : raw_ids) {
        indices.push_back(raw_id);
    }

    return OEFPSparse(morgan_sparse_binary_spec(options), std::move(indices));
}

MorganSparseFingerprintResult MakeMorganSparseFingerprintWithMapping(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        throw std::invalid_argument("Morgan count simulation is only supported for binary fingerprints.");
    }

    const auto events = enumerate_events(mol, options, UNFOLDED_MORGAN_IDS);
    std::set<std::uint32_t> raw_ids;
    for (const auto& event : events) {
        raw_ids.insert(event.raw_id);
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(raw_ids.size());
    for (const auto raw_id : raw_ids) {
        indices.push_back(raw_id);
    }

    return MorganSparseFingerprintResult(
        OEFPSparse(morgan_sparse_binary_spec(options), std::move(indices)),
        mapping_from_events(events));
}

MorganGenerationProfile ProfileMorganFingerprint(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options) {
    validate_options(options);
    MorganGenerationProfile profile;
    static_cast<void>(make_profiled_fingerprint_from_events(mol, options, profile));
    return profile;
}

} // namespace OEFP
