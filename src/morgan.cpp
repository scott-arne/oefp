#include "oefp/morgan.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <oesystem.h>

namespace OEFP {
namespace {

constexpr const char* MORGAN_COMPAT_VERSION = "Morgan-2026.03.1";
constexpr std::int32_t RDKIT_AROMATIC_BOND_TYPE = 12;
constexpr std::uint32_t UNFOLDED_MORGAN_IDS = 0u;

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
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    std::int32_t invariant = 1;
};

struct AtomRecord {
    const OEChem::OEAtomBase* atom = nullptr;
    std::uint32_t index = 0;
    std::vector<std::uint32_t> bond_indices;
};

struct MoleculeGraph {
    std::vector<AtomRecord> atoms;
    std::vector<BondRecord> bonds;
};

using BondSet = std::set<std::uint32_t>;

struct BondSetLess {
    bool operator()(const BondSet& lhs, const BondSet& rhs) const {
        auto lhs_it = lhs.rbegin();
        auto rhs_it = rhs.rbegin();
        while (lhs_it != lhs.rend() || rhs_it != rhs.rend()) {
            if (lhs_it == lhs.rend()) {
                return true;
            }
            if (rhs_it == rhs.rend()) {
                return false;
            }
            if (*lhs_it != *rhs_it) {
                return *lhs_it < *rhs_it;
            }
            ++lhs_it;
            ++rhs_it;
        }
        return false;
    }
};

using RoundEnvironment = std::tuple<BondSet, std::uint32_t, std::uint32_t>;

bool round_environment_less(const RoundEnvironment& lhs, const RoundEnvironment& rhs) {
    const BondSetLess bond_set_less;
    if (bond_set_less(std::get<0>(lhs), std::get<0>(rhs))) {
        return true;
    }
    if (bond_set_less(std::get<0>(rhs), std::get<0>(lhs))) {
        return false;
    }
    return std::tie(std::get<1>(lhs), std::get<2>(lhs))
           < std::tie(std::get<1>(rhs), std::get<2>(rhs));
}

void validate_options(const MorganOptions& options) {
    if (options.num_bits == 0) {
        throw std::invalid_argument("Morgan num_bits must be greater than zero.");
    }
    if (options.use_chirality) {
        throw std::invalid_argument("Morgan chirality conformance is not implemented yet.");
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
           << ";use_bond_types=" << bool_parameter(options.use_bond_types)
           << ";only_nonzero_invariants=" << bool_parameter(options.only_nonzero_invariants)
           << ";include_ring_membership=" << bool_parameter(options.include_ring_membership)
           << ";include_redundant_environments="
           << bool_parameter(options.include_redundant_environments)
           << ";output=sparse_binary";
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

std::uint32_t atom_invariant(const OEChem::OEAtomBase& atom, const MorganOptions& options) {
    std::vector<std::uint32_t> components;
    components.reserve(6);
    components.push_back(atom.GetAtomicNum());
    components.push_back(atom.GetDegree());
    components.push_back(atom.GetTotalHCount());
    components.push_back(static_cast<std::uint32_t>(atom.GetFormalCharge()));
    components.push_back(static_cast<std::uint32_t>(isotope_delta(atom)));
    if (options.include_ring_membership && atom.IsInRing()) {
        components.push_back(1u);
    }
    return vector_hash(components);
}

std::int32_t bond_type_value(const OEChem::OEBondBase& bond) {
    if (bond.IsAromatic()) {
        return RDKIT_AROMATIC_BOND_TYPE;
    }
    return static_cast<std::int32_t>(bond.GetOrder());
}

std::int32_t bond_invariant(const OEChem::OEBondBase& bond, const MorganOptions& options) {
    if (!options.use_bond_types) {
        return 1;
    }
    return bond_type_value(bond);
}

void assign_rdkit_bond_order(MoleculeGraph& graph, const std::vector<std::uint32_t>& bond_ids) {
    const auto missing_bond = std::numeric_limits<std::uint32_t>::max();
    std::vector<std::uint32_t> parent_bonds(graph.atoms.size(), missing_bond);
    std::vector<std::uint32_t> parent_atoms(graph.atoms.size(), 0u);

    for (const auto bond_id : bond_ids) {
        const auto& bond = graph.bonds[bond_id];
        const auto child = std::max(bond.begin, bond.end);
        const auto parent = std::min(bond.begin, bond.end);
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
        const auto child = std::max(bond.begin, bond.end);
        const auto parent = std::min(bond.begin, bond.end);
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
    graph.atoms.resize(mol.GetMaxAtomIdx());
    graph.bonds.resize(mol.GetMaxBondIdx());

    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto idx = atom->GetIdx();
        if (idx >= graph.atoms.size()) {
            throw std::runtime_error("OpenEye atom index exceeds molecule atom storage.");
        }
        graph.atoms[idx].atom = atom;
        graph.atoms[idx].index = idx;
    }

    std::vector<std::uint32_t> bond_ids;
    bond_ids.reserve(mol.NumBonds());
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto idx = bond->GetIdx();
        if (idx >= graph.bonds.size()) {
            throw std::runtime_error("OpenEye bond index exceeds molecule bond storage.");
        }
        bond_ids.push_back(idx);
        auto& record = graph.bonds[idx];
        record.index = idx;
        record.begin = bond->GetBgnIdx();
        record.end = bond->GetEndIdx();
        record.invariant = bond_invariant(*bond, options);
        graph.atoms[record.begin].bond_indices.push_back(idx);
        graph.atoms[record.end].bond_indices.push_back(idx);
    }

    assign_rdkit_bond_order(graph, bond_ids);
    return graph;
}

std::uint32_t other_atom(const BondRecord& bond, std::uint32_t atom_id) {
    return bond.begin == atom_id ? bond.end : bond.begin;
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

std::vector<MorganEvent> enumerate_events(
    const OEChem::OEMolBase& mol,
    const MorganOptions& options,
    std::uint32_t fold_size) {
    const auto graph = build_graph(mol, options);

    std::vector<std::uint32_t> atom_invariants(graph.atoms.size(), 0u);
    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom != nullptr) {
            atom_invariants[atom_record.index] = atom_invariant(*atom_record.atom, options);
        }
    }

    std::vector<MorganEvent> events;
    events.reserve((static_cast<std::size_t>(options.radius) + 1u) * mol.NumAtoms());

    std::vector<std::uint32_t> current = atom_invariants;
    std::vector<std::uint32_t> next(graph.atoms.size(), 0u);
    std::vector<BondSet> neighborhoods(graph.atoms.size());
    std::vector<BondSet> round_neighborhoods = neighborhoods;
    std::vector<bool> dead_atoms(graph.atoms.size(), false);
    std::set<BondSet, BondSetLess> seen_neighborhoods;
    const auto atom_order = atom_iteration_order(current, options);

    for (const auto& atom_record : graph.atoms) {
        if (atom_record.atom == nullptr) {
            continue;
        }
        if (!options.only_nonzero_invariants || current[atom_record.index] != 0u) {
            events.push_back(
                MorganEvent{atom_record.index, 0u, current[atom_record.index],
                            event_bit_id(current[atom_record.index], fold_size)});
        }
    }

    for (std::uint32_t layer = 0; layer < options.radius; ++layer) {
        std::vector<RoundEnvironment> this_round;

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

            std::vector<NeighborInvariant> neighbors;
            neighbors.reserve(atom_record.bond_indices.size());

            for (const auto bond_id : atom_record.bond_indices) {
                const auto& bond = graph.bonds[bond_id];
                round_neighborhoods[atom_id].insert(bond.sort_index);
                const auto nbr = other_atom(bond, atom_id);
                round_neighborhoods[atom_id].insert(
                    neighborhoods[nbr].begin(),
                    neighborhoods[nbr].end());
                neighbors.push_back(NeighborInvariant{bond.invariant, current[nbr]});
            }

            std::sort(neighbors.begin(), neighbors.end());

            std::uint32_t invariant = layer;
            invariant = combine_hash(invariant, current[atom_id]);
            for (const auto& neighbor : neighbors) {
                invariant = combine_hash(invariant, neighbor);
            }

            next[atom_id] = invariant;
            this_round.emplace_back(round_neighborhoods[atom_id], invariant, atom_id);
        }

        // RDKit sorts boost::dynamic_bitset neighborhoods before duplicate
        // suppression, so compare the stored bond ids from the highest bit down.
        std::sort(this_round.begin(), this_round.end(), round_environment_less);
        for (const auto& item : this_round) {
            const auto& neighborhood = std::get<0>(item);
            const auto raw_id = std::get<1>(item);
            const auto atom_id = std::get<2>(item);

            if (options.include_redundant_environments
                || seen_neighborhoods.count(neighborhood) == 0u) {
                if (!options.only_nonzero_invariants || atom_invariants[atom_id] != 0u) {
                    events.push_back(MorganEvent{
                        atom_id,
                        layer + 1u,
                        raw_id,
                        event_bit_id(raw_id, fold_size),
                    });
                    seen_neighborhoods.insert(neighborhood);
                }
            } else {
                dead_atoms[atom_id] = true;
            }
        }

        current.swap(next);
        std::fill(next.begin(), next.end(), 0u);
        neighborhoods = round_neighborhoods;
    }

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

} // namespace

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

OEFP MakeMorganFingerprint(const OEChem::OEMolBase& mol, const MorganOptions& options) {
    validate_options(options);

    if (options.count_simulation) {
        return make_count_simulated_fingerprint(mol, options);
    }

    OEFP fingerprint(morgan_spec(options, FingerprintValueType::Binary));
    for (const auto& event : enumerate_events(mol, options, options.num_bits)) {
        fingerprint.SetBit(event.bit_id);
    }
    return fingerprint;
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

} // namespace OEFP
