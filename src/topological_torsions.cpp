#include "oefp/topological_torsions.h"

#include "oefp/stereo.h"

#include <algorithm>
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

constexpr const char* TOPOLOGICAL_TORSIONS_COMPAT_VERSION = "TopologicalTorsions-1.0.0";
constexpr std::uint32_t NUM_TYPE_BITS = 4;
constexpr std::uint32_t ATOM_NUMBER_TYPE_COUNT = 1u << NUM_TYPE_BITS;
constexpr std::uint32_t ATOM_NUMBER_TYPES[ATOM_NUMBER_TYPE_COUNT] = {
    5u, 6u, 7u, 8u, 9u, 14u, 15u, 16u, 17u, 33u, 34u, 35u, 51u, 52u, 53u};
constexpr std::uint32_t NUM_PI_BITS = 2;
constexpr std::uint32_t MAX_NUM_PI = (1u << NUM_PI_BITS) - 1u;
constexpr std::uint32_t NUM_BRANCH_BITS = 3;
constexpr std::uint32_t MAX_NUM_BRANCHES = (1u << NUM_BRANCH_BITS) - 1u;
constexpr std::uint32_t CODE_SIZE = NUM_TYPE_BITS + NUM_PI_BITS + NUM_BRANCH_BITS;
// RDKit appends two chirality bits (R=1, S=2, none=0) above the base atom code
// when includeChirality is set, matching the Atom Pair encoding.
constexpr std::uint32_t NUM_CHIRAL_BITS = 2;
constexpr std::uint64_t BITS_PER_WORD = 64u;
constexpr std::uint32_t MAX_TORSION_ATOM_COUNT = 7u;

// Per-atom path-code width, widened by the chirality bits when enabled. The
// base CODE_SIZE bits are computed identically regardless of chirality, so the
// use_chirality=false output is bit-for-bit unchanged.
std::uint32_t path_code_size(const TopologicalTorsionsOptions& options) {
    return CODE_SIZE + (options.use_chirality ? NUM_CHIRAL_BITS : 0u);
}

struct BondRef {
    std::uint32_t atom_index = 0;
    std::uint32_t bond_index = 0;
};

struct AtomRecord {
    const OEChem::OEAtomBase* atom = nullptr;
    std::uint32_t index = 0;
    std::vector<BondRef> neighbors;
};

struct MoleculeGraph {
    std::vector<AtomRecord> atoms;
};

struct TorsionCodeEvent {
    std::vector<std::uint32_t> path_codes;
};

struct TorsionEvent {
    std::uint32_t raw_hash = 0;
    std::uint32_t bit_id = 0;
};

const char* bool_parameter(bool value) {
    return value ? "true" : "false";
}

std::string count_bounds_parameter(const std::vector<std::uint32_t>& count_bounds) {
    std::ostringstream values;
    for (std::size_t i = 0; i < count_bounds.size(); ++i) {
        if (i != 0u) {
            values << ',';
        }
        values << count_bounds[i];
    }
    return values.str();
}

std::string canonical_parameters(const TopologicalTorsionsOptions& options) {
    std::ostringstream params;
    params << "torsion_atom_count=" << options.torsion_atom_count
           << ";num_bits=" << options.num_bits
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";count_simulation=" << bool_parameter(options.count_simulation)
           << ";count_bounds=" << count_bounds_parameter(options.count_bounds);
    return params.str();
}

std::string canonical_sparse_binary_parameters(const TopologicalTorsionsOptions& options) {
    std::ostringstream params;
    params << "torsion_atom_count=" << options.torsion_atom_count
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";count_simulation=" << bool_parameter(options.count_simulation)
           << ";count_bounds=" << count_bounds_parameter(options.count_bounds)
           << ";output=sparse_binary";
    return params.str();
}

std::string canonical_sparse_count_parameters(const TopologicalTorsionsOptions& options) {
    std::ostringstream params;
    params << "torsion_atom_count=" << options.torsion_atom_count
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";output=sparse_count";
    return params.str();
}

std::string canonical_descriptor_parameters(const TopologicalTorsionsOptions& options) {
    std::ostringstream params;
    params << "torsion_atom_count=" << options.torsion_atom_count
           << ";use_chirality=" << bool_parameter(options.use_chirality)
           << ";output=descriptors";
    return params.str();
}

TopologicalTorsionsOptions count_options(const TopologicalTorsionsOptions& options) {
    TopologicalTorsionsOptions normalized = options;
    normalized.count_simulation = false;
    return normalized;
}

FingerprintSpec topological_torsions_spec(const TopologicalTorsionsOptions& options) {
    FingerprintSpec spec;
    spec.size_bits = options.num_bits;
    spec.value_type = FingerprintValueType::Binary;
    spec.source_name = "RDKit-compatible";
    spec.source_type = "TopologicalTorsions";
    spec.source_version = TOPOLOGICAL_TORSIONS_COMPAT_VERSION;
    spec.parameters = canonical_parameters(options);
    return spec;
}

std::uint64_t raw_sparse_count_result_size(const TopologicalTorsionsOptions& options) {
    const auto raw_bits =
        static_cast<std::uint64_t>(options.torsion_atom_count) * path_code_size(options);
    return std::uint64_t{1} << raw_bits;
}

std::uint32_t sparse_result_size(const TopologicalTorsionsOptions& options) {
    const auto raw_bits =
        static_cast<std::uint64_t>(options.torsion_atom_count) * path_code_size(options);
    const auto raw_size = std::uint64_t{1} << raw_bits;
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(std::numeric_limits<std::uint32_t>::max(), raw_size));
}

FingerprintSpec topological_torsions_sparse_binary_spec(
    const TopologicalTorsionsOptions& options) {
    FingerprintSpec spec = topological_torsions_spec(options);
    spec.size_bits = sparse_result_size(options);
    spec.parameters = canonical_sparse_binary_parameters(options);
    return spec;
}

FingerprintSpec topological_torsions_count_spec(
    const TopologicalTorsionsOptions& options) {
    FingerprintSpec spec = topological_torsions_spec(count_options(options));
    spec.value_type = FingerprintValueType::Counted;
    return spec;
}

FingerprintSpec topological_torsions_sparse_count_spec(
    const TopologicalTorsionsOptions& options) {
    FingerprintSpec spec = topological_torsions_spec(options);
    spec.size_bits = raw_sparse_count_result_size(options);
    spec.value_type = FingerprintValueType::Counted;
    spec.parameters = canonical_sparse_count_parameters(options);
    return spec;
}

DescriptorSpec topological_torsions_descriptor_spec(
    const TopologicalTorsionsOptions& options) {
    DescriptorSpec spec;
    spec.value_type = DescriptorValueType::String;
    spec.source_name = "OEFP";
    spec.source_type = "TopologicalTorsions";
    spec.source_version = TOPOLOGICAL_TORSIONS_COMPAT_VERSION;
    spec.parameters = canonical_descriptor_parameters(options);
    return spec;
}

std::shared_ptr<const DescriptorSchema> topological_torsions_descriptor_schema(
    const TopologicalTorsionsOptions& options) {
    const auto spec = topological_torsions_descriptor_spec(options);
    DescriptorSchemaBuilder builder;
    builder.Add(DescriptorDefinition{
        "topological_torsions",
        DescriptorValueKind::CountedStringKeys,
        "raw",
        spec.source_name,
        spec.source_type,
        spec.source_version,
        spec.parameters,
        "",
        "Topological Torsions counted path-code keys.",
        kTopologicalTorsionsPrerequisites});
    return builder.Build();
}

void validate_common_options(const TopologicalTorsionsOptions& options) {
    if (options.torsion_atom_count == 0u) {
        throw std::invalid_argument(
            "Topological Torsions torsion_atom_count must be greater than zero.");
    }
    if (options.torsion_atom_count > MAX_TORSION_ATOM_COUNT) {
        throw std::invalid_argument(
            "Topological Torsions torsion_atom_count must be smaller than 8.");
    }
    // The raw sparse-count code packs torsion_atom_count atom codes into one
    // uint64. The base 9-bit code always fits (max 7*9=63), but the two extra
    // chirality bits per atom can exceed 64 bits at large torsion lengths.
    if (options.use_chirality
        && static_cast<std::uint64_t>(options.torsion_atom_count) * path_code_size(options)
               > 64u) {
        throw std::invalid_argument(
            "Topological Torsions chirality requires torsion_atom_count small enough that "
            "torsion_atom_count * 11 does not exceed 64 bits.");
    }
}

void validate_options(const TopologicalTorsionsOptions& options) {
    validate_common_options(options);
    if (options.num_bits == 0u) {
        throw std::invalid_argument("Topological Torsions num_bits must be greater than zero.");
    }
    if (options.count_simulation && options.count_bounds.empty()) {
        throw std::invalid_argument(
            "Topological Torsions count_bounds cannot be empty when count simulation is enabled.");
    }
    if (options.count_simulation && options.count_bounds.size() >= options.num_bits) {
        throw std::invalid_argument(
            "Topological Torsions count_bounds size must be smaller than num_bits.");
    }
}

void validate_count_options(const TopologicalTorsionsOptions& options) {
    validate_common_options(options);
    if (options.num_bits == 0u) {
        throw std::invalid_argument("Topological Torsions num_bits must be greater than zero.");
    }
}

void validate_sparse_options(const TopologicalTorsionsOptions& options) {
    validate_common_options(options);
    if (options.count_simulation && options.count_bounds.empty()) {
        throw std::invalid_argument(
            "Topological Torsions count_bounds cannot be empty when count simulation is enabled.");
    }
    if (options.count_simulation
        && options.count_bounds.size() >= sparse_result_size(options)) {
        throw std::invalid_argument(
            "Topological Torsions count_bounds size must be smaller than sparse fingerprint size.");
    }
}

void validate_sparse_count_options(const TopologicalTorsionsOptions& options) {
    validate_common_options(options);
}

void validate_descriptor_options(const TopologicalTorsionsOptions& options) {
    validate_common_options(options);
}

std::uint32_t hash_combine_value(std::uint32_t seed, std::uint32_t hashed_value) {
    seed ^= hashed_value + 0x9e3779b9u + (seed << 6u) + (seed >> 2u);
    return seed;
}

std::uint32_t event_bit_id(std::uint32_t raw_hash, std::uint32_t fold_size) {
    return static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(raw_hash) % static_cast<std::uint64_t>(fold_size));
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

    std::uint32_t bond_index = 0;
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto begin = bond->GetBgnIdx();
        const auto end = bond->GetEndIdx();
        if (begin >= graph.atoms.size() || end >= graph.atoms.size()) {
            throw std::runtime_error("OpenEye bond references an atom outside molecule storage.");
        }
        if (graph.atoms[begin].atom == nullptr || graph.atoms[end].atom == nullptr) {
            throw std::runtime_error("OpenEye bond references a missing atom record.");
        }
        // RDKit Topological Torsions call findAllPathsOfLengthN(..., useHs=false).
        if (graph.atoms[begin].atom->GetAtomicNum() == 1
            || graph.atoms[end].atom->GetAtomicNum() == 1) {
            ++bond_index;
            continue;
        }
        graph.atoms[begin].neighbors.push_back(BondRef{end, bond_index});
        graph.atoms[end].neighbors.push_back(BondRef{begin, bond_index});
        ++bond_index;
    }

    return graph;
}

std::vector<std::uint32_t> bond_signature(
    const MoleculeGraph& graph,
    const std::vector<std::uint32_t>& path) {
    std::vector<std::uint32_t> signature;
    if (path.size() <= 1u) {
        return signature;
    }
    signature.reserve(path.size() - 1u);
    for (std::size_t i = 0; i + 1u < path.size(); ++i) {
        const auto begin = path[i];
        const auto end = path[i + 1u];
        auto found = std::find_if(
            graph.atoms[begin].neighbors.begin(),
            graph.atoms[begin].neighbors.end(),
            [end](const BondRef& ref) {
                return ref.atom_index == end;
            });
        if (found == graph.atoms[begin].neighbors.end()) {
            throw std::runtime_error("Topological Torsions path contains a non-bonded atom pair.");
        }
        signature.push_back(found->bond_index);
    }
    std::sort(signature.begin(), signature.end());
    return signature;
}

bool path_has_invalid_repeat(const std::vector<std::uint32_t>& path) {
    std::set<std::uint32_t> seen;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const auto atom_id = path[i];
        if (i != 0u && atom_id != path.front() && seen.count(atom_id) != 0u) {
            return true;
        }
        seen.insert(atom_id);
    }
    return false;
}

template <typename PathSink>
void emit_unique_path(
    const MoleculeGraph& graph,
    const std::vector<std::uint32_t>& path,
    std::set<std::vector<std::uint32_t>>& seen_bond_signatures,
    PathSink&& emit_path) {
    if (path.size() > 1u) {
        auto signature = bond_signature(graph, path);
        if (!seen_bond_signatures.insert(std::move(signature)).second) {
            return;
        }
    }
    if (path_has_invalid_repeat(path)) {
        return;
    }
    emit_path(path);
}

template <typename PathSink>
void extend_atom_paths(
    const MoleculeGraph& graph,
    std::uint32_t target_size,
    std::vector<std::uint32_t>& path,
    std::set<std::vector<std::uint32_t>>& seen_bond_signatures,
    PathSink&& emit_path) {
    if (path.size() == target_size) {
        emit_unique_path(graph, path, seen_bond_signatures, emit_path);
        return;
    }

    const auto end_atom = path.back();
    for (const auto& neighbor : graph.atoms[end_atom].neighbors) {
        const auto next_atom = neighbor.atom_index;
        const auto found = std::find(path.begin(), path.end(), next_atom);
        if (found == path.end()) {
            path.push_back(next_atom);
            extend_atom_paths(graph, target_size, path, seen_bond_signatures, emit_path);
            path.pop_back();
            continue;
        }

        const auto ring_closure_step = target_size > 2u && path.size() == target_size - 1u;
        const auto immediate_backtrack = path.size() >= 2u && path[path.size() - 2u] == next_atom;
        if (ring_closure_step && !immediate_backtrack) {
            path.push_back(next_atom);
            extend_atom_paths(graph, target_size, path, seen_bond_signatures, emit_path);
            path.pop_back();
        }
    }
}

template <typename PathSink>
void enumerate_atom_paths(
    const MoleculeGraph& graph,
    std::uint32_t target_size,
    PathSink&& emit_path) {
    std::set<std::vector<std::uint32_t>> seen_bond_signatures;
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom == nullptr) {
            continue;
        }
        std::vector<std::uint32_t> path{atom_record.index};
        if (target_size == 1u) {
            emit_path(path);
            continue;
        }
        extend_atom_paths(graph, target_size, path, seen_bond_signatures, emit_path);
    }
}

std::vector<std::uint32_t> canonical_path_codes(
    const std::vector<std::uint32_t>& path_codes) {
    bool reverse = false;
    std::size_t i = 0;
    std::size_t j = path_codes.size() - 1u;
    while (i < j) {
        if (path_codes[i] > path_codes[j]) {
            reverse = true;
            break;
        }
        if (path_codes[i] < path_codes[j]) {
            break;
        }
        ++i;
        --j;
    }

    if (!reverse) {
        return path_codes;
    }
    return std::vector<std::uint32_t>(path_codes.rbegin(), path_codes.rend());
}

std::uint32_t topological_torsion_hash(
    const std::vector<std::uint32_t>& canonical_codes) {
    // The folded/hashed fingerprint reduces each path code modulo the base
    // 9-bit limit, which collapses the chirality high bits back into the base
    // value (e.g. base 34 with chiral bit R -> 35). RDKit hashes this reduced
    // value for chirality-aware folded output, whereas the raw sparse-count and
    // descriptor codes keep the full-width code. For achiral codes the value is
    // already below the limit, so the reduction is a no-op and the output is
    // unchanged.
    constexpr std::uint32_t code_size_limit = (1u << CODE_SIZE) - 1u;
    std::uint32_t seed = 0;
    for (const auto code : canonical_codes) {
        seed = hash_combine_value(seed, code % code_size_limit);
    }
    return seed;
}

std::uint64_t topological_torsion_code(
    const std::vector<std::uint32_t>& canonical_codes,
    std::uint32_t code_size) {
    std::uint64_t code = 0;
    for (std::size_t i = 0; i < canonical_codes.size(); ++i) {
        code |= static_cast<std::uint64_t>(canonical_codes[i]) << (code_size * i);
    }
    return code;
}

std::vector<std::uint32_t> path_codes_from_atom_path(
    const std::vector<std::uint32_t>& atom_path,
    const std::vector<std::uint32_t>& atom_invariants,
    const std::vector<std::uint32_t>& atom_chiral_bits,
    bool use_chirality) {
    const auto code_size_limit = (1u << CODE_SIZE) - 1u;
    std::vector<std::uint32_t> path_codes;
    path_codes.reserve(atom_path.size());
    for (std::size_t i = 0; i < atom_path.size(); ++i) {
        auto code = atom_invariants[atom_path[i]] % code_size_limit + 1u;
        if (i != 0u && i + 1u != atom_path.size()) {
            --code;
        }
        // The base code occupies the low CODE_SIZE bits unchanged; chirality
        // bits sit above it so the achiral encoding is preserved exactly.
        if (use_chirality) {
            code |= atom_chiral_bits[atom_path[i]] << CODE_SIZE;
        }
        path_codes.push_back(code);
    }
    return canonical_path_codes(path_codes);
}

template <typename CodeSink>
void enumerate_code_events_into(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options,
    CodeSink&& emit_code_event) {
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEAssignHybridization(working_mol);
    const auto graph = build_graph(working_mol);

    std::vector<std::uint32_t> atom_invariants(graph.atoms.size(), 0u);
    std::vector<std::uint32_t> atom_chiral_bits(graph.atoms.size(), 0u);
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom != nullptr) {
            atom_invariants[atom_record.index] = atom_code(*atom_record.atom) - 2u;
            if (options.use_chirality) {
                atom_chiral_bits[atom_record.index] = detail::AtomPairChiralityBits(
                    detail::PerceiveAtomStereo(working_mol, *atom_record.atom));
            }
        }
    }

    const bool use_chirality = options.use_chirality;
    enumerate_atom_paths(
        graph,
        options.torsion_atom_count,
        [&atom_invariants, &atom_chiral_bits, use_chirality, &emit_code_event](
            const std::vector<std::uint32_t>& atom_path) {
            emit_code_event(TorsionCodeEvent{path_codes_from_atom_path(
                atom_path, atom_invariants, atom_chiral_bits, use_chirality)});
        });
}

template <typename EventSink>
void enumerate_events_into(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options,
    std::uint32_t fold_size,
    EventSink&& emit_event) {
    enumerate_code_events_into(
        mol,
        options,
        [fold_size, &emit_event](const TorsionCodeEvent& code_event) {
            const auto raw_hash = topological_torsion_hash(code_event.path_codes);
            emit_event(TorsionEvent{raw_hash, event_bit_id(raw_hash, fold_size)});
        });
}

OEFP make_standard_fingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options,
    const FingerprintSpec& spec) {
    std::vector<std::uint64_t> words(DenseWordCount(spec.size_bits), 0ULL);
    enumerate_events_into(
        mol,
        options,
        options.num_bits,
        [&words](const TorsionEvent& event) {
            words[event.bit_id / BITS_PER_WORD] |= 1ULL << (event.bit_id % BITS_PER_WORD);
        });
    return OEFP(spec, std::move(words));
}

OEFP make_count_simulated_fingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options,
    const FingerprintSpec& spec) {
    const auto bound_count = options.count_bounds.size();
    const auto effective_size = options.num_bits / bound_count;
    std::vector<std::uint32_t> effective_counts(effective_size, 0u);
    enumerate_events_into(
        mol,
        options,
        static_cast<std::uint32_t>(effective_size),
        [&effective_counts](const TorsionEvent& event) {
            auto& count = effective_counts[event.bit_id];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "Topological Torsions count simulation count exceeds uint32 storage.");
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

OEFPCount make_count_fingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    std::map<std::uint32_t, std::uint32_t> folded_counts;
    enumerate_events_into(
        mol,
        options,
        options.num_bits,
        [&folded_counts](const TorsionEvent& event) {
            auto& count = folded_counts[event.bit_id];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "Topological Torsions count fingerprint count exceeds uint32 storage.");
            }
            ++count;
        });

    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> counts;
    indices.reserve(folded_counts.size());
    counts.reserve(folded_counts.size());
    for (const auto& [index, count] : folded_counts) {
        indices.push_back(index);
        counts.push_back(count);
    }

    return OEFPCount(topological_torsions_count_spec(options), std::move(indices), std::move(counts));
}

OEFPSparse make_sparse_fingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    const auto sparse_size = sparse_result_size(options);
    std::set<std::uint32_t> on_bits;
    enumerate_events_into(
        mol,
        options,
        sparse_size,
        [&on_bits](const TorsionEvent& event) {
            on_bits.insert(event.bit_id);
        });

    std::vector<std::uint32_t> indices;
    indices.reserve(on_bits.size());
    for (const auto bit_id : on_bits) {
        indices.push_back(bit_id);
    }

    return OEFPSparse(topological_torsions_sparse_binary_spec(options), std::move(indices));
}

OEFPCount64 make_sparse_count_fingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    std::map<std::uint64_t, std::uint32_t> raw_counts;
    const auto code_size = path_code_size(options);
    enumerate_code_events_into(
        mol,
        options,
        [&raw_counts, code_size](const TorsionCodeEvent& event) {
            auto& count = raw_counts[topological_torsion_code(event.path_codes, code_size)];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "Topological Torsions sparse count fingerprint count exceeds uint32 storage.");
            }
            ++count;
        });

    std::vector<std::uint64_t> indices;
    std::vector<std::uint32_t> counts;
    indices.reserve(raw_counts.size());
    counts.reserve(raw_counts.size());
    for (const auto& [index, count] : raw_counts) {
        indices.push_back(index);
        counts.push_back(count);
    }

    return OEFPCount64(
        topological_torsions_sparse_count_spec(options),
        std::move(indices),
        std::move(counts));
}

OEFPSparse make_count_simulated_sparse_fingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    const auto bound_count = options.count_bounds.size();
    const auto effective_size = sparse_result_size(options) / bound_count;
    std::map<std::uint32_t, std::uint32_t> effective_counts;
    enumerate_events_into(
        mol,
        options,
        static_cast<std::uint32_t>(effective_size),
        [&effective_counts](const TorsionEvent& event) {
            auto& count = effective_counts[event.bit_id];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "Topological Torsions sparse count simulation count exceeds uint32 storage.");
            }
            ++count;
        });

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

    return OEFPSparse(topological_torsions_sparse_binary_spec(options), std::move(indices));
}

std::string topological_torsions_descriptor_key(
    const std::vector<std::uint32_t>& canonical_codes) {
    std::ostringstream key;
    for (std::size_t i = 0; i < canonical_codes.size(); ++i) {
        if (i != 0u) {
            key << '_';
        }
        key << canonical_codes[i];
    }
    return key.str();
}

DescriptorSet make_descriptors(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    std::map<std::string, std::uint32_t> counts_by_key;
    enumerate_code_events_into(
        mol,
        options,
        [&counts_by_key](const TorsionCodeEvent& event) {
            auto& count = counts_by_key[topological_torsions_descriptor_key(event.path_codes)];
            if (count == std::numeric_limits<std::uint32_t>::max()) {
                throw std::overflow_error(
                    "Topological Torsions descriptor count exceeds uint32 storage.");
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

    const auto schema = topological_torsions_descriptor_schema(options);
    DescriptorSetBuilder builder(schema);
    builder.Set(
        "topological_torsions",
        DescriptorValue::CountedStringKeys(std::move(keys), std::move(counts)));
    return builder.Build();
}

} // namespace

TopologicalTorsionsGenerator::TopologicalTorsionsGenerator(
    TopologicalTorsionsOptions options)
    : options_(std::move(options)) {
    validate_options(options_);
    binary_spec_ = topological_torsions_spec(options_);
}

OEFP TopologicalTorsionsGenerator::Fingerprint(const OEChem::OEMolBase& mol) const {
    if (options_.count_simulation) {
        return make_count_simulated_fingerprint(mol, options_, binary_spec_);
    }
    return make_standard_fingerprint(mol, options_, binary_spec_);
}

const TopologicalTorsionsOptions& TopologicalTorsionsGenerator::Options() const {
    return options_;
}

OEFP MakeTopologicalTorsionsFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    return TopologicalTorsionsGenerator(options).Fingerprint(mol);
}

OEFPCount MakeTopologicalTorsionsCountFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    validate_count_options(options);
    return make_count_fingerprint(mol, count_options(options));
}

OEFPSparse MakeTopologicalTorsionsSparseFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    validate_sparse_options(options);
    if (options.count_simulation) {
        return make_count_simulated_sparse_fingerprint(mol, options);
    }
    return make_sparse_fingerprint(mol, options);
}

OEFPCount64 MakeTopologicalTorsionsSparseCountFingerprint(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    validate_sparse_count_options(options);
    return make_sparse_count_fingerprint(mol, options);
}

DescriptorSet MakeTopologicalTorsionsDescriptors(
    const OEChem::OEMolBase& mol,
    const TopologicalTorsionsOptions& options) {
    validate_descriptor_options(options);
    return make_descriptors(mol, options);
}

} // namespace OEFP
