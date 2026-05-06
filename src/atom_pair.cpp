#include "oefp/atom_pair.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <oesystem.h>

namespace OEFP {
namespace {

constexpr const char* ATOM_PAIR_COMPAT_VERSION = "AtomPair-1.1.0";
constexpr std::uint32_t NUM_TYPE_BITS = 4;
constexpr std::uint32_t ATOM_NUMBER_TYPE_COUNT = 1u << NUM_TYPE_BITS;
constexpr std::uint32_t ATOM_NUMBER_TYPES[ATOM_NUMBER_TYPE_COUNT] = {
    5u, 6u, 7u, 8u, 9u, 14u, 15u, 16u, 17u, 33u, 34u, 35u, 51u, 52u, 53u};
constexpr std::uint32_t NUM_PI_BITS = 2;
constexpr std::uint32_t MAX_NUM_PI = (1u << NUM_PI_BITS) - 1u;
constexpr std::uint32_t NUM_BRANCH_BITS = 3;
constexpr std::uint32_t MAX_NUM_BRANCHES = (1u << NUM_BRANCH_BITS) - 1u;
constexpr std::uint32_t CODE_SIZE = NUM_TYPE_BITS + NUM_PI_BITS + NUM_BRANCH_BITS;
constexpr std::uint32_t NUM_PATH_BITS = 5;
constexpr std::uint32_t MAX_PATH_LENGTH = (1u << NUM_PATH_BITS) - 1u;

struct AtomRecord {
    const OEChem::OEAtomBase* atom = nullptr;
    std::uint32_t index = 0;
    std::vector<std::uint32_t> neighbors;
};

struct MoleculeGraph {
    std::vector<AtomRecord> atoms;
};

struct AtomPairEvent {
    std::uint32_t raw_id = 0;
    std::uint32_t bit_id = 0;
};

const char* bool_parameter(bool value) {
    return value ? "true" : "false";
}

std::string canonical_parameters(const AtomPairOptions& options) {
    std::ostringstream params;
    params << "min_distance=" << options.min_distance
           << ";max_distance=" << options.max_distance
           << ";num_bits=" << options.num_bits
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";use_2d=" << bool_parameter(options.use_2d)
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

FingerprintSpec atom_pair_spec(const AtomPairOptions& options) {
    FingerprintSpec spec;
    spec.size_bits = options.num_bits;
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "RDKit-compatible";
    spec.source_type = "AtomPair";
    spec.source_version = ATOM_PAIR_COMPAT_VERSION;
    spec.parameters = canonical_parameters(options);
    return spec;
}

void validate_options(const AtomPairOptions& options) {
    if (options.num_bits == 0) {
        throw std::invalid_argument("Atom Pair num_bits must be greater than zero.");
    }
    if (options.min_distance > options.max_distance) {
        throw std::invalid_argument("Atom Pair min_distance cannot exceed max_distance.");
    }
    if (options.max_distance >= MAX_PATH_LENGTH) {
        throw std::invalid_argument("Atom Pair max_distance must be smaller than 31.");
    }
    if (options.use_chirality) {
        throw std::invalid_argument("Atom Pair chirality conformance is not implemented yet.");
    }
    if (!options.use_2d) {
        throw std::invalid_argument("Atom Pair 3D distance conformance is not implemented yet.");
    }
    if (options.count_simulation && options.count_bounds.empty()) {
        throw std::invalid_argument("Atom Pair count_bounds cannot be empty when count simulation is enabled.");
    }
    if (options.count_simulation && options.count_bounds.size() >= options.num_bits) {
        throw std::invalid_argument("Atom Pair count_bounds size must be smaller than num_bits.");
    }
}

std::uint32_t hash_combine_value(std::uint32_t seed, std::uint32_t hashed_value) {
    seed ^= hashed_value + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
    return seed;
}

std::uint32_t combine_hash(std::uint32_t seed, std::uint32_t value) {
    return hash_combine_value(seed, value);
}

std::uint32_t event_bit_id(std::uint32_t raw_id, std::uint32_t fold_size) {
    return raw_id % fold_size;
}

std::uint32_t num_pi_electrons(const OEChem::OEAtomBase& atom) {
    if (atom.IsAromatic()) {
        return 1;
    }
    if (atom.GetHyb() == OEChem::OEHybridization::sp3) {
        return 0;
    }

    const auto explicit_valence = atom.GetExplicitValence();
    const auto physical_bonds = atom.GetExplicitHCount() + atom.GetExplicitDegree();
    if (explicit_valence < physical_bonds) {
        return 0;
    }
    return explicit_valence - physical_bonds;
}

std::uint32_t atom_code(const OEChem::OEAtomBase& atom) {
    std::uint32_t code = atom.GetExplicitDegree() % MAX_NUM_BRANCHES;
    code |= (num_pi_electrons(atom) % MAX_NUM_PI) << NUM_BRANCH_BITS;

    std::uint32_t type_idx = 0;
    while (type_idx < ATOM_NUMBER_TYPE_COUNT) {
        if (ATOM_NUMBER_TYPES[type_idx] == atom.GetAtomicNum()) {
            break;
        }
        if (ATOM_NUMBER_TYPES[type_idx] > atom.GetAtomicNum()) {
            type_idx = ATOM_NUMBER_TYPE_COUNT;
            break;
        }
        ++type_idx;
    }
    if (type_idx == ATOM_NUMBER_TYPE_COUNT) {
        --type_idx;
    }

    code |= type_idx << (NUM_BRANCH_BITS + NUM_PI_BITS);
    return code;
}

std::uint32_t atom_pair_hash(
    std::uint32_t first_code,
    std::uint32_t second_code,
    std::uint32_t distance) {
    std::uint32_t seed = 0;
    seed = combine_hash(seed, std::min(first_code, second_code));
    seed = combine_hash(seed, distance);
    seed = combine_hash(seed, std::max(first_code, second_code));
    return seed;
}

MoleculeGraph build_graph(const OEChem::OEMolBase& mol) {
    MoleculeGraph graph;
    graph.atoms.resize(mol.GetMaxAtomIdx());

    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto idx = atom->GetIdx();
        if (idx >= graph.atoms.size()) {
            throw std::runtime_error("OpenEye atom index exceeds molecule atom storage.");
        }
        graph.atoms[idx].atom = atom;
        graph.atoms[idx].index = idx;
    }

    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto begin = bond->GetBgnIdx();
        const auto end = bond->GetEndIdx();
        if (begin >= graph.atoms.size() || end >= graph.atoms.size()) {
            throw std::runtime_error("OpenEye bond references an atom outside molecule storage.");
        }
        graph.atoms[begin].neighbors.push_back(end);
        graph.atoms[end].neighbors.push_back(begin);
    }

    return graph;
}

std::vector<std::uint32_t> shortest_distances(
    const MoleculeGraph& graph,
    std::uint32_t start_atom) {
    const auto unreachable = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> distances(graph.atoms.size(), unreachable);
    std::queue<std::uint32_t> queue;
    distances[start_atom] = 0;
    queue.push(start_atom);

    while (!queue.empty()) {
        const auto atom_id = queue.front();
        queue.pop();
        for (const auto neighbor : graph.atoms[atom_id].neighbors) {
            if (distances[neighbor] != unreachable) {
                continue;
            }
            distances[neighbor] = distances[atom_id] + 1u;
            queue.push(neighbor);
        }
    }
    return distances;
}

std::vector<AtomPairEvent> enumerate_events(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options,
    std::uint32_t fold_size) {
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEAssignHybridization(working_mol);
    const auto graph = build_graph(working_mol);
    const auto code_size_limit = (1u << CODE_SIZE) - 1u;

    std::vector<std::uint32_t> atom_codes(graph.atoms.size(), 0u);
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom != nullptr) {
            atom_codes[atom_record.index] = atom_code(*atom_record.atom) % code_size_limit;
        }
    }

    std::vector<AtomPairEvent> events;
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom == nullptr) {
            continue;
        }
        const auto distances = shortest_distances(graph, atom_record.index);
        for (std::uint32_t other = atom_record.index + 1u; other < graph.atoms.size(); ++other) {
            if (graph.atoms[other].atom == nullptr) {
                continue;
            }
            const auto distance = distances[other];
            if (distance < options.min_distance || distance > options.max_distance) {
                continue;
            }
            const auto raw_id = atom_pair_hash(
                atom_codes[atom_record.index],
                atom_codes[other],
                distance);
            events.push_back(AtomPairEvent{raw_id, event_bit_id(raw_id, fold_size)});
        }
    }
    return events;
}

OEFP make_standard_fingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    OEFP fingerprint(atom_pair_spec(options));
    for (const auto& event : enumerate_events(mol, options, options.num_bits)) {
        fingerprint.SetBit(event.bit_id);
    }
    return fingerprint;
}

OEFP make_count_simulated_fingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    const auto bound_count = options.count_bounds.size();
    const auto effective_size = options.num_bits / bound_count;
    std::map<std::uint32_t, std::uint32_t> effective_counts;
    for (const auto& event : enumerate_events(mol, options, effective_size)) {
        auto& count = effective_counts[event.bit_id];
        if (count == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Atom Pair count simulation count exceeds uint32 storage.");
        }
        ++count;
    }

    OEFP fingerprint(atom_pair_spec(options));
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

} // namespace

OEFP MakeAtomPairFingerprint(const OEChem::OEMolBase& mol, const AtomPairOptions& options) {
    validate_options(options);
    if (options.count_simulation) {
        return make_count_simulated_fingerprint(mol, options);
    }
    return make_standard_fingerprint(mol, options);
}

} // namespace OEFP
