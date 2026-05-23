#include "oefp/atom_pair.h"

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
#include <utility>
#include <vector>

#include <oesystem.h>

namespace OEFP {
namespace {

constexpr const char* TOPOLOGICAL_ATOM_PAIR_COMPAT_VERSION = "TopologicalAtomPair-1.1.0";
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
constexpr std::uint32_t ATOM_PAIR_SPARSE_SIZE = 1u << (NUM_PATH_BITS + 2u * CODE_SIZE);
constexpr std::uint64_t BITS_PER_WORD = 64u;

using Clock = std::chrono::steady_clock;

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

struct AtomPairCodeEvent {
    std::uint32_t first_code = 0;
    std::uint32_t second_code = 0;
    std::uint32_t distance = 0;
};

enum class AtomPairIdentifierKind {
    Hashed,
    Encoded,
};

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

const char* bool_parameter(bool value) {
    return value ? "true" : "false";
}

std::string canonical_parameters(const AtomPairOptions& options) {
    std::ostringstream params;
    params << "min_distance=" << options.min_distance
           << ";max_distance=" << options.max_distance
           << ";num_bits=" << options.num_bits
           << ";use_chirality=" << bool_parameter(options.use_chirality)
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

std::string canonical_sparse_binary_parameters(const AtomPairOptions& options) {
    std::ostringstream params;
    params << "min_distance=" << options.min_distance
           << ";max_distance=" << options.max_distance
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";count_simulation=" << bool_parameter(options.count_simulation)
           << ";count_bounds=";
    for (std::size_t i = 0; i < options.count_bounds.size(); ++i) {
        if (i != 0u) {
            params << ',';
        }
        params << options.count_bounds[i];
    }
    params << ";output=sparse_binary";
    return params.str();
}

std::string canonical_sparse_count_parameters(const AtomPairOptions& options) {
    std::ostringstream params;
    params << "min_distance=" << options.min_distance
           << ";max_distance=" << options.max_distance
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";output=sparse_count";
    return params.str();
}

std::string canonical_descriptor_parameters(const AtomPairOptions& options) {
    std::ostringstream params;
    params << "min_distance=" << options.min_distance
           << ";max_distance=" << options.max_distance
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";output=descriptors";
    return params.str();
}

FingerprintSpec atom_pair_spec(const AtomPairOptions& options) {
    FingerprintSpec spec;
    spec.size_bits = options.num_bits;
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "RDKit-compatible";
    spec.source_type = "TopologicalAtomPair";
    spec.source_version = TOPOLOGICAL_ATOM_PAIR_COMPAT_VERSION;
    spec.parameters = canonical_parameters(options);
    return spec;
}

FingerprintSpec atom_pair_sparse_binary_spec(const AtomPairOptions& options) {
    FingerprintSpec spec = atom_pair_spec(options);
    spec.size_bits = ATOM_PAIR_SPARSE_SIZE;
    spec.parameters = canonical_sparse_binary_parameters(options);
    return spec;
}

FingerprintSpec atom_pair_sparse_count_spec(const AtomPairOptions& options) {
    FingerprintSpec spec = atom_pair_spec(options);
    spec.size_bits = ATOM_PAIR_SPARSE_SIZE;
    spec.value_type = FingerprintValueType::Counted;
    spec.parameters = canonical_sparse_count_parameters(options);
    return spec;
}

DescriptorSpec atom_pair_descriptor_spec(const AtomPairOptions& options) {
    DescriptorSpec spec;
    spec.value_type = DescriptorValueType::String;
    spec.source_name = "OEFP";
    spec.source_type = "TopologicalAtomPair";
    spec.source_version = TOPOLOGICAL_ATOM_PAIR_COMPAT_VERSION;
    spec.parameters = canonical_descriptor_parameters(options);
    return spec;
}

std::shared_ptr<const DescriptorSchema> atom_pair_descriptor_schema(
    const AtomPairOptions& options) {
    const auto spec = atom_pair_descriptor_spec(options);
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{
        "atom_pair",
        DescriptorValueKind::CountedStringKeys,
        "raw",
        spec.source_name,
        spec.source_type,
        spec.source_version,
        spec.parameters,
        "",
        "Topological Atom Pair counted string keys.",
        kTopologicalAtomPairPrerequisites});
    return builder.Build();
}

AtomPairOptions count_options(const AtomPairOptions& options) {
    AtomPairOptions normalized = options;
    normalized.count_simulation = false;
    return normalized;
}

FingerprintSpec atom_pair_count_spec(const AtomPairOptions& options) {
    FingerprintSpec spec = atom_pair_spec(count_options(options));
    spec.value_type = FingerprintValueType::Counted;
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
        throw std::invalid_argument(
            "Distance Atom Pair requires existing 3D coordinates and is not implemented yet.");
    }
    if (options.count_simulation && options.count_bounds.empty()) {
        throw std::invalid_argument("Atom Pair count_bounds cannot be empty when count simulation is enabled.");
    }
    if (options.count_simulation && options.count_bounds.size() >= options.num_bits) {
        throw std::invalid_argument("Atom Pair count_bounds size must be smaller than num_bits.");
    }
}

void validate_count_options(const AtomPairOptions& options) {
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
        throw std::invalid_argument(
            "Distance Atom Pair requires existing 3D coordinates and is not implemented yet.");
    }
}

void validate_sparse_options(const AtomPairOptions& options) {
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
        throw std::invalid_argument(
            "Distance Atom Pair requires existing 3D coordinates and is not implemented yet.");
    }
    if (options.count_simulation && options.count_bounds.empty()) {
        throw std::invalid_argument("Atom Pair count_bounds cannot be empty when count simulation is enabled.");
    }
    if (options.count_simulation && options.count_bounds.size() >= ATOM_PAIR_SPARSE_SIZE) {
        throw std::invalid_argument("Atom Pair count_bounds size must be smaller than sparse fingerprint size.");
    }
}

void validate_sparse_count_options(const AtomPairOptions& options) {
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
        throw std::invalid_argument(
            "Distance Atom Pair requires existing 3D coordinates and is not implemented yet.");
    }
}

void validate_descriptor_options(const AtomPairOptions& options) {
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
        throw std::invalid_argument(
            "Distance Atom Pair requires existing 3D coordinates and is not implemented yet.");
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

std::uint32_t atom_pair_code(
    std::uint32_t first_code,
    std::uint32_t second_code,
    std::uint32_t distance) {
    const auto smaller_code = std::min(first_code, second_code);
    const auto larger_code = std::max(first_code, second_code);
    return distance
           | (smaller_code << NUM_PATH_BITS)
           | (larger_code << (NUM_PATH_BITS + CODE_SIZE));
}

std::uint32_t atom_pair_identifier(
    std::uint32_t first_code,
    std::uint32_t second_code,
    std::uint32_t distance,
    AtomPairIdentifierKind identifier_kind) {
    if (identifier_kind == AtomPairIdentifierKind::Encoded) {
        return atom_pair_code(first_code, second_code, distance);
    }
    return atom_pair_hash(first_code, second_code, distance);
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
        graph.atoms[idx].neighbors.reserve(atom->GetDegree());
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

void fill_shortest_distances(
    const MoleculeGraph& graph,
    std::uint32_t start_atom,
    std::vector<std::uint32_t>& distances,
    std::vector<std::uint32_t>& queue) {
    const auto unreachable = std::numeric_limits<std::uint32_t>::max();
    std::fill(distances.begin(), distances.end(), unreachable);
    queue.clear();
    distances[start_atom] = 0;
    queue.push_back(start_atom);

    std::size_t queue_head = 0;
    while (queue_head < queue.size()) {
        const auto atom_id = queue[queue_head];
        ++queue_head;
        for (const auto neighbor : graph.atoms[atom_id].neighbors) {
            if (distances[neighbor] != unreachable) {
                continue;
            }
            distances[neighbor] = distances[atom_id] + 1u;
            queue.push_back(neighbor);
        }
    }
}

template <typename EventSink>
void enumerate_code_events_into(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options,
    EventSink&& emit_event) {
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

    std::vector<std::uint32_t> distances(graph.atoms.size(), 0u);
    std::vector<std::uint32_t> distance_queue;
    distance_queue.reserve(graph.atoms.size());
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom == nullptr) {
            continue;
        }
        fill_shortest_distances(graph, atom_record.index, distances, distance_queue);
        for (std::uint32_t other = atom_record.index + 1u; other < graph.atoms.size(); ++other) {
            if (graph.atoms[other].atom == nullptr) {
                continue;
            }
            const auto distance = distances[other];
            if (distance < options.min_distance || distance > options.max_distance) {
                continue;
            }
            emit_event(AtomPairCodeEvent{
                atom_codes[atom_record.index],
                atom_codes[other],
                distance});
        }
    }
}

template <typename EventSink>
void enumerate_events_into(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options,
    std::uint32_t fold_size,
    AtomPairIdentifierKind identifier_kind,
    EventSink&& emit_event) {
    enumerate_code_events_into(
        mol,
        options,
        [fold_size, identifier_kind, &emit_event](const AtomPairCodeEvent& code_event) {
            const auto raw_id = atom_pair_identifier(
                code_event.first_code,
                code_event.second_code,
                code_event.distance,
                identifier_kind);
            emit_event(AtomPairEvent{raw_id, event_bit_id(raw_id, fold_size)});
        });
}

std::vector<AtomPairEvent> enumerate_events(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options,
    std::uint32_t fold_size,
    AtomPairIdentifierKind identifier_kind) {
    std::vector<AtomPairEvent> events;
    events.reserve(mol.NumAtoms() * (mol.NumAtoms() - 1u) / 2u);
    enumerate_events_into(
        mol,
        options,
        fold_size,
        identifier_kind,
        [&events](AtomPairEvent event) {
            events.push_back(event);
        });
    return events;
}

OEFP make_standard_fingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options,
    const FingerprintSpec& spec) {
    std::vector<std::uint64_t> words(DenseWordCount(spec.size_bits), 0ULL);
    enumerate_events_into(
        mol,
        options,
        options.num_bits,
        AtomPairIdentifierKind::Hashed,
        [&words](const AtomPairEvent& event) {
            words[event.bit_id / BITS_PER_WORD] |= 1ULL << (event.bit_id % BITS_PER_WORD);
        });
    return OEFP(spec, std::move(words));
}

OEFP make_count_simulated_fingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options,
    const FingerprintSpec& spec) {
    const auto bound_count = options.count_bounds.size();
    const auto effective_size = options.num_bits / bound_count;
    std::vector<std::uint32_t> effective_counts(effective_size, 0u);
    enumerate_events_into(
        mol,
        options,
        effective_size,
        AtomPairIdentifierKind::Hashed,
        [&effective_counts](const AtomPairEvent& event) {
            auto& count = effective_counts[event.bit_id];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("Atom Pair count simulation count exceeds uint32 storage.");
            }
            ++count;
        });

    std::vector<std::uint64_t> words(DenseWordCount(spec.size_bits), 0ULL);
    for (std::uint32_t base_bit = 0; base_bit < effective_counts.size(); ++base_bit) {
        const auto count = effective_counts[base_bit];
        if (count == 0u) {
            continue;
        }
        for (std::size_t i = 0; i < bound_count; ++i) {
            if (count >= options.count_bounds[i]) {
                const auto bit_id =
                    static_cast<std::uint64_t>(base_bit) * bound_count
                    + static_cast<std::uint64_t>(i);
                words[bit_id / BITS_PER_WORD] |= 1ULL << (bit_id % BITS_PER_WORD);
            }
        }
    }
    return OEFP(spec, std::move(words));
}

OEFP make_profiled_fingerprint_from_events(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options,
    AtomPairGenerationProfile& profile) {
    const auto spec = atom_pair_spec(options);
    const auto fold_size =
        options.count_simulation
            ? options.num_bits / static_cast<std::uint32_t>(options.count_bounds.size())
            : options.num_bits;
    const auto molecule_start = Clock::now();
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEAssignHybridization(working_mol);
    const auto molecule_end = Clock::now();
    profile.molecule_preparation_seconds += elapsed_seconds(molecule_start, molecule_end);

    const auto graph_start = Clock::now();
    const auto graph = build_graph(working_mol);
    const auto graph_end = Clock::now();
    profile.graph_seconds += elapsed_seconds(graph_start, graph_end);
    profile.atom_count = static_cast<std::uint32_t>(working_mol.NumAtoms());

    const auto atom_code_start = Clock::now();
    const auto code_size_limit = (1u << CODE_SIZE) - 1u;
    std::vector<std::uint32_t> atom_codes(graph.atoms.size(), 0u);
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom != nullptr) {
            atom_codes[atom_record.index] = atom_code(*atom_record.atom) % code_size_limit;
        }
    }
    const auto atom_code_end = Clock::now();
    profile.atom_code_seconds += elapsed_seconds(atom_code_start, atom_code_end);

    std::vector<AtomPairEvent> events;
    events.reserve(working_mol.NumAtoms() * (working_mol.NumAtoms() - 1u) / 2u);
    std::vector<std::uint32_t> distances(graph.atoms.size(), 0u);
    std::vector<std::uint32_t> distance_queue;
    distance_queue.reserve(graph.atoms.size());
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom == nullptr) {
            continue;
        }

        const auto distance_start = Clock::now();
        fill_shortest_distances(graph, atom_record.index, distances, distance_queue);
        const auto distance_end = Clock::now();
        profile.distance_seconds += elapsed_seconds(distance_start, distance_end);

        const auto pair_start = Clock::now();
        for (std::uint32_t other = atom_record.index + 1u; other < graph.atoms.size(); ++other) {
            if (graph.atoms[other].atom == nullptr) {
                continue;
            }
            const auto distance = distances[other];
            if (distance < options.min_distance || distance > options.max_distance) {
                continue;
            }
            const auto raw_id = atom_pair_identifier(
                atom_codes[atom_record.index],
                atom_codes[other],
                distance,
                AtomPairIdentifierKind::Hashed);
            events.push_back(AtomPairEvent{raw_id, event_bit_id(raw_id, fold_size)});
            ++profile.event_count;
        }
        const auto pair_end = Clock::now();
        profile.pair_enumeration_seconds += elapsed_seconds(pair_start, pair_end);
    }

    const auto bit_folding_start = Clock::now();
    std::vector<std::uint64_t> words(DenseWordCount(spec.size_bits), 0ULL);
    if (options.count_simulation) {
        const auto bound_count = options.count_bounds.size();
        std::vector<std::uint32_t> effective_counts(fold_size, 0u);
        for (const auto& event : events) {
            auto& count = effective_counts[event.bit_id];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("Atom Pair count simulation count exceeds uint32 storage.");
            }
            ++count;
        }
        for (std::uint32_t base_bit = 0; base_bit < effective_counts.size(); ++base_bit) {
            const auto count = effective_counts[base_bit];
            if (count == 0u) {
                continue;
            }
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

OEFPCount make_count_fingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    std::map<std::uint32_t, std::uint32_t> folded_counts;
    for (const auto& event :
         enumerate_events(mol, options, options.num_bits, AtomPairIdentifierKind::Hashed)) {
        auto& count = folded_counts[event.bit_id];
        if (count == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Atom Pair count fingerprint count exceeds uint32 storage.");
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

    return OEFPCount(atom_pair_count_spec(options), std::move(indices), std::move(counts));
}

OEFPSparse make_sparse_fingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    std::set<std::uint32_t> on_bits;
    for (const auto& event :
         enumerate_events(mol, options, ATOM_PAIR_SPARSE_SIZE, AtomPairIdentifierKind::Hashed)) {
        on_bits.insert(event.bit_id);
    }

    std::vector<std::uint32_t> indices;
    indices.reserve(on_bits.size());
    for (const auto bit_id : on_bits) {
        indices.push_back(bit_id);
    }

    return OEFPSparse(atom_pair_sparse_binary_spec(options), std::move(indices));
}

OEFPSparse make_count_simulated_sparse_fingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    const auto bound_count = options.count_bounds.size();
    const auto effective_size = ATOM_PAIR_SPARSE_SIZE / bound_count;
    std::map<std::uint32_t, std::uint32_t> effective_counts;
    for (const auto& event :
         enumerate_events(mol, options, effective_size, AtomPairIdentifierKind::Hashed)) {
        auto& count = effective_counts[event.bit_id];
        if (count == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Atom Pair sparse count simulation count exceeds uint32 storage.");
        }
        ++count;
    }

    std::vector<std::uint32_t> indices;
    for (const auto& [base_bit, count] : effective_counts) {
        for (std::size_t i = 0; i < bound_count; ++i) {
            if (count >= options.count_bounds[i]) {
                indices.push_back(
                    base_bit * static_cast<std::uint32_t>(bound_count)
                    + static_cast<std::uint32_t>(i));
            }
        }
    }

    return OEFPSparse(atom_pair_sparse_binary_spec(options), std::move(indices));
}

OEFPCount make_sparse_count_fingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    std::map<std::uint32_t, std::uint32_t> sparse_counts;
    for (const auto& event :
         enumerate_events(mol, options, ATOM_PAIR_SPARSE_SIZE, AtomPairIdentifierKind::Encoded)) {
        auto& count = sparse_counts[event.raw_id];
        if (count == std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error("Atom Pair sparse count fingerprint count exceeds uint32 storage.");
        }
        ++count;
    }

    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> counts;
    indices.reserve(sparse_counts.size());
    counts.reserve(sparse_counts.size());
    for (const auto& [index, count] : sparse_counts) {
        indices.push_back(index);
        counts.push_back(count);
    }

    return OEFPCount(atom_pair_sparse_count_spec(options), std::move(indices), std::move(counts));
}

std::string atom_pair_descriptor_key(
    std::uint32_t first_code,
    std::uint32_t second_code,
    std::uint32_t distance) {
    const auto smaller_code = std::min(first_code, second_code);
    const auto larger_code = std::max(first_code, second_code);
    std::ostringstream key;
    key << smaller_code << '_' << distance << '_' << larger_code;
    return key.str();
}

DescriptorSet make_descriptors(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    std::map<std::string, std::uint32_t> counts_by_key;
    enumerate_code_events_into(
        mol,
        options,
        [&counts_by_key](const AtomPairCodeEvent& event) {
            auto& count = counts_by_key[
                atom_pair_descriptor_key(event.first_code, event.second_code, event.distance)];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error("Atom Pair descriptor count exceeds uint32 storage.");
            }
            ++count;
        });

    std::vector<std::string> keys;
    std::vector<std::uint32_t> counts;
    keys.reserve(counts_by_key.size());
    counts.reserve(counts_by_key.size());
    for (const auto& [key, count] : counts_by_key) {
        keys.push_back(key);
        counts.push_back(count);
    }

    const auto schema = atom_pair_descriptor_schema(options);
    DescriptorSetBuilder builder(schema);
    builder.Set("atom_pair", DescriptorValue::CountedStringKeys(std::move(keys), std::move(counts)));
    return builder.Build();
}

} // namespace

double AtomPairGenerationProfile::TotalSeconds() const {
    return molecule_preparation_seconds + graph_seconds + atom_code_seconds + distance_seconds
           + pair_enumeration_seconds + bit_folding_seconds;
}

AtomPairGenerator::AtomPairGenerator(AtomPairOptions options)
    : options_(std::move(options)) {
    validate_options(options_);
    binary_spec_ = atom_pair_spec(options_);
}

OEFP AtomPairGenerator::Fingerprint(const OEChem::OEMolBase& mol) const {
    if (options_.count_simulation) {
        return make_count_simulated_fingerprint(mol, options_, binary_spec_);
    }
    return make_standard_fingerprint(mol, options_, binary_spec_);
}

const AtomPairOptions& AtomPairGenerator::Options() const {
    return options_;
}

OEFP MakeAtomPairFingerprint(const OEChem::OEMolBase& mol, const AtomPairOptions& options) {
    return AtomPairGenerator(options).Fingerprint(mol);
}

OEFPCount MakeAtomPairCountFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    validate_count_options(options);
    return make_count_fingerprint(mol, count_options(options));
}

OEFPSparse MakeAtomPairSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    validate_sparse_options(options);
    if (options.count_simulation) {
        return make_count_simulated_sparse_fingerprint(mol, options);
    }
    return make_sparse_fingerprint(mol, options);
}

OEFPCount MakeAtomPairSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    validate_sparse_count_options(options);
    return make_sparse_count_fingerprint(mol, options);
}

DescriptorSet MakeAtomPairDescriptors(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    validate_descriptor_options(options);
    return make_descriptors(mol, options);
}

AtomPairGenerationProfile ProfileAtomPairFingerprint(
    const OEChem::OEMolBase& mol,
    const AtomPairOptions& options) {
    validate_options(options);
    AtomPairGenerationProfile profile;
    static_cast<void>(make_profiled_fingerprint_from_events(mol, options, profile));
    return profile;
}

} // namespace OEFP
