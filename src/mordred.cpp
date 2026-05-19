#include "oefp/mordred.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <oesystem.h>
#include <oemolprop.h>

namespace OEFP {
namespace {

struct MordredFirstBatchValues {
    std::uint32_t acidic_groups = 0u;
    std::uint32_t basic_groups = 0u;
    std::uint32_t aromatic_atoms = 0u;
    std::uint32_t heavy_atoms = 0u;
    std::uint32_t hetero_atoms = 0u;
    std::uint32_t hydrogens = 0u;
    std::uint32_t spiro_atoms = 0u;
    std::uint32_t bridgehead_atoms = 0u;
    std::uint32_t boron = 0u;
    std::uint32_t carbon = 0u;
    std::uint32_t nitrogen = 0u;
    std::uint32_t oxygen = 0u;
    std::uint32_t sulfur = 0u;
    std::uint32_t phosphorus = 0u;
    std::uint32_t fluorine = 0u;
    std::uint32_t chlorine = 0u;
    std::uint32_t bromine = 0u;
    std::uint32_t iodine = 0u;
    std::uint32_t halogens = 0u;
    std::uint32_t heavy_bonds = 0u;
    std::uint32_t aromatic_bonds = 0u;
    std::uint32_t single_heavy_bonds = 0u;
    std::uint32_t double_heavy_bonds = 0u;
    std::uint32_t triple_heavy_bonds = 0u;
    std::uint32_t multiple_heavy_bonds = 0u;
    std::uint32_t rotatable_bonds = 0u;
    std::array<std::uint32_t, 5> sp1_carbons_by_carbon_degree{};
    std::array<std::uint32_t, 5> sp2_carbons_by_carbon_degree{};
    std::array<std::uint32_t, 5> sp3_carbons_by_carbon_degree{};
    std::uint32_t sp2_carbons = 0u;
    std::uint32_t sp3_carbons = 0u;
    std::uint32_t hbond_acceptors = 0u;
    std::uint32_t hbond_donors = 0u;
    double topo_psa_no = 0.0;
    double topo_psa = 0.0;
    double exact_weight = 0.0;
    double crippen_logp = 0.0;
    double crippen_mr = 0.0;
};

struct MordredAdditivePropertyValues {
    std::optional<double> sz;
    std::optional<double> sm;
    std::optional<double> sv;
    std::optional<double> sse;
    std::optional<double> spe;
    std::optional<double> sare;
    std::optional<double> sp;
    std::optional<double> si;
    std::optional<double> m_z;
    std::optional<double> mm;
    std::optional<double> mv;
    std::optional<double> mse;
    std::optional<double> mpe;
    std::optional<double> mare;
    std::optional<double> mp;
    std::optional<double> mi;
    std::optional<double> v_mcgowan;
    std::optional<double> apol;
    std::optional<double> bpol;
    std::optional<double> vabc;
};

struct AtomicPropertyValue {
    std::uint32_t atomic_number;
    double value;
};

constexpr double kPi = 3.14159265358979323846;

bool is_hydrogen(const OEChem::OEAtomBase& atom) {
    return atom.GetAtomicNum() == 1u;
}

std::optional<double> lookup_atomic_property(
    const std::vector<AtomicPropertyValue>& values,
    std::uint32_t atomic_number) {
    for (const auto& entry : values) {
        if (entry.atomic_number == atomic_number) {
            return entry.value;
        }
    }
    return std::nullopt;
}

double sphere_volume(double radius) {
    return 4.0 / 3.0 * kPi * radius * radius * radius;
}

std::optional<double> mordred_mass(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> values{
        {1u, 1.008},      {5u, 10.81},      {6u, 12.011}, {7u, 14.007},
        {8u, 15.999},     {9u, 18.9984032}, {14u, 28.085}, {15u, 30.973762},
        {16u, 32.06},     {17u, 35.45},     {33u, 74.9216},
        {34u, 78.96},     {35u, 79.904},
    };
    return lookup_atomic_property(values, atomic_number);
}

std::optional<double> mordred_vdw_volume(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> radii{
        {1u, 1.10}, {5u, 1.92}, {6u, 1.70}, {7u, 1.55}, {8u, 1.52},
        {9u, 1.47}, {14u, 2.10}, {15u, 1.80}, {16u, 1.80},
        {17u, 1.75}, {33u, 1.85}, {34u, 1.90}, {35u, 1.85},
    };
    const auto radius = lookup_atomic_property(radii, atomic_number);
    if (!radius.has_value()) {
        return std::nullopt;
    }
    return sphere_volume(*radius);
}

std::optional<double> mordred_sanderson(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> values{
        {1u, 2.592}, {5u, 2.275}, {6u, 2.746}, {7u, 3.194}, {8u, 3.654},
        {9u, 4.000}, {14u, 2.138}, {15u, 2.515}, {16u, 2.957},
        {17u, 3.475}, {33u, 2.816}, {34u, 3.014}, {35u, 3.219},
    };
    return lookup_atomic_property(values, atomic_number);
}

std::optional<double> mordred_pauling(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> values{
        {1u, 2.2}, {5u, 2.04}, {6u, 2.55}, {7u, 3.04}, {8u, 3.44},
        {9u, 3.98}, {14u, 1.9}, {15u, 2.19}, {16u, 2.58},
        {17u, 3.16}, {33u, 2.18}, {34u, 2.55}, {35u, 2.96},
    };
    return lookup_atomic_property(values, atomic_number);
}

std::optional<double> mordred_allred_rocow(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> values{
        {1u, 2.20}, {5u, 2.01}, {6u, 2.50}, {7u, 3.07}, {8u, 3.50},
        {9u, 4.10}, {14u, 1.74}, {15u, 2.06}, {16u, 2.44},
        {17u, 2.83}, {33u, 2.20}, {34u, 2.48}, {35u, 2.74},
    };
    return lookup_atomic_property(values, atomic_number);
}

std::optional<double> mordred_polarizability94(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> values{
        {1u, 0.666793}, {5u, 3.03}, {6u, 1.67}, {7u, 1.10},
        {8u, 0.802}, {9u, 0.557}, {14u, 5.53}, {15u, 3.63},
        {16u, 2.90}, {17u, 2.18}, {33u, 4.31}, {34u, 3.77},
        {35u, 3.05},
    };
    return lookup_atomic_property(values, atomic_number);
}

std::optional<double> mordred_ionization_potential(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> values{
        {1u, 13.598443}, {5u, 8.29802}, {6u, 11.26030}, {7u, 14.5341},
        {8u, 13.61805}, {9u, 17.4228}, {14u, 8.15168}, {15u, 10.48669},
        {16u, 10.36001}, {17u, 12.96763}, {33u, 9.7886},
        {34u, 9.75239}, {35u, 11.8138},
    };
    return lookup_atomic_property(values, atomic_number);
}

std::optional<double> mordred_mc_gowan_volume(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> values{
        {1u, 8.71}, {5u, 18.31}, {6u, 16.35}, {7u, 14.39}, {8u, 12.43},
        {9u, 10.47}, {14u, 26.83}, {15u, 24.87}, {16u, 22.91},
        {17u, 20.95}, {33u, 29.42}, {34u, 27.81}, {35u, 26.21},
    };
    return lookup_atomic_property(values, atomic_number);
}

std::optional<double> bondi_atom_volume(std::uint32_t atomic_number) {
    static const std::vector<AtomicPropertyValue> radii{
        {1u, 1.20}, {5u, 2.13}, {6u, 1.70}, {7u, 1.55}, {8u, 1.52},
        {9u, 1.47}, {14u, 2.10}, {15u, 1.80}, {16u, 1.80},
        {17u, 1.75}, {33u, 1.85}, {34u, 1.90}, {35u, 1.85},
    };
    const auto radius = lookup_atomic_property(radii, atomic_number);
    if (!radius.has_value()) {
        return std::nullopt;
    }
    return sphere_volume(*radius);
}

bool is_halogen(std::uint32_t atomic_number) {
    return atomic_number == 9u || atomic_number == 17u || atomic_number == 35u
           || atomic_number == 53u;
}

double default_isotopic_mass(std::uint32_t atomic_number) {
    const auto mass_number = OEChem::OEGetDefaultMass(atomic_number);
    return OEChem::OEGetIsotopicWeight(atomic_number, mass_number);
}

double atom_exact_mass(const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto isotope = static_cast<std::uint32_t>(atom.GetIsotope());
    if (isotope != 0u) {
        return OEChem::OEGetIsotopicWeight(atomic_number, isotope);
    }
    return default_isotopic_mass(atomic_number);
}

double round_tpsa(double value) {
    return std::round(value * 100.0) / 100.0;
}

std::uint32_t carbon_neighbor_count(const OEChem::OEAtomBase& atom) {
    std::uint32_t count = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> nbr = atom.GetAtoms(); nbr; ++nbr) {
        if (nbr->GetAtomicNum() == 6u) {
            ++count;
        }
    }
    return count;
}

std::uint32_t nonterminal_rotor_neighbor_count(const OEChem::OEAtomBase& atom) {
    std::uint32_t count = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> nbr = atom.GetAtoms(); nbr; ++nbr) {
        const auto atomic_number = static_cast<std::uint32_t>(nbr->GetAtomicNum());
        if (atomic_number != 1u && !is_halogen(atomic_number)) {
            ++count;
        }
    }
    return count;
}

bool is_spiro_atom(const OEChem::OEAtomBase& atom) {
    if (!atom.IsInRing()) {
        return false;
    }

    std::uint32_t ring_bonds = 0u;
    for (OESystem::OEIter<OEChem::OEBondBase> bond = atom.GetBonds(); bond; ++bond) {
        if (bond->IsInRing()) {
            ++ring_bonds;
        }
    }
    return ring_bonds >= 4u;
}

bool is_mordred_rotatable_bond(const OEChem::OEBondBase& bond) {
    const auto* begin = bond.GetBgn();
    const auto* end = bond.GetEnd();
    if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
        return false;
    }
    if (bond.GetOrder() != 1u || bond.IsAromatic() || bond.IsInRing()) {
        return false;
    }
    return nonterminal_rotor_neighbor_count(*begin) > 1u
           && nonterminal_rotor_neighbor_count(*end) > 1u;
}

std::uint32_t count_unique_smarts_root_atoms(
    const OEChem::OEMolBase& mol,
    const std::vector<const char*>& smarts_patterns) {
    std::unordered_set<unsigned int> matched_atom_indices;
    for (const char* smarts : smarts_patterns) {
        OEChem::OESubSearch search(smarts);
        if (!search) {
            continue;
        }
        for (OESystem::OEIter<OEChem::OEMatchBase> match = search.Match(mol, true); match;
             ++match) {
            for (OESystem::OEIter<OEChem::OEMatchPair<OEChem::OEAtomBase>> atom_match =
                     match->GetAtoms();
                 atom_match; ++atom_match) {
                if (atom_match->pattern != nullptr && atom_match->target != nullptr
                    && atom_match->pattern->GetIdx() == 0u) {
                    matched_atom_indices.insert(atom_match->target->GetIdx());
                }
            }
        }
    }
    return static_cast<std::uint32_t>(matched_atom_indices.size());
}

struct CrippenPattern {
    const char* smarts;
    double logp;
    double mr;
};

const std::vector<CrippenPattern>& crippen_patterns() {
    // RDKit applies the Wildman-Crippen table in file order. The first pattern
    // that claims an atom wins, so table order is part of the descriptor.
    static const std::vector<CrippenPattern> patterns{
        {"[CH4]", 0.1441, 2.503},
        {"[CH3]C", 0.1441, 2.503},
        {"[CH2](C)C", 0.1441, 2.503},
        {"[CH](C)(C)C", 0.0, 2.433},
        {"[C](C)(C)(C)C", 0.0, 2.433},
        {"[CH3][N,O,P,S,F,Cl,Br,I]", -0.2035, 2.753},
        {"[CH2X4]([N,O,P,S,F,Cl,Br,I])[A;!#1]", -0.2035, 2.753},
        {"[CH1X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])[A;!#1]", -0.2051, 2.731},
        {"[CH0X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])([A;!#1])[A;!#1]", -0.2051, 2.731},
        {"[C]=[!C;A;!#1]", -0.2783, 5.007},
        {"[CH2]=C", 0.1551, 3.513},
        {"[CH1](=C)[A;!#1]", 0.1551, 3.513},
        {"[CH0](=C)([A;!#1])[A;!#1]", 0.1551, 3.513},
        {"[C](=C)=C", 0.1551, 3.513},
        {"[CX2]#[A;!#1]", 0.0017, 3.888},
        {"[CH3]c", 0.08452, 2.464},
        {"[CH3]a", -0.1444, 2.412},
        {"[CH2X4]a", -0.0516, 2.488},
        {"[CHX4]a", 0.1193, 2.582},
        {"[CH0X4]a", -0.0967, 2.576},
        {"[cH0]-[A;!C;!N;!O;!S;!F;!Cl;!Br;!I;!#1]", -0.5443, 4.041},
        {"[c][#9]", 0.0, 3.257},
        {"[c][#17]", 0.245, 3.564},
        {"[c][#35]", 0.198, 3.18},
        {"[c][#53]", 0.0, 3.104},
        {"[cH]", 0.1581, 3.35},
        {"[c](:a)(:a):a", 0.2955, 4.346},
        {"[c](:a)(:a)-a", 0.2713, 3.904},
        {"[c](:a)(:a)-C", 0.136, 3.509},
        {"[c](:a)(:a)-N", 0.4619, 4.067},
        {"[c](:a)(:a)-O", 0.5437, 3.853},
        {"[c](:a)(:a)-S", 0.1893, 2.673},
        {"[c](:a)(:a)=[C,N,O]", -0.8186, 3.135},
        {"[C](=C)(a)[A;!#1]", 0.264, 4.305},
        {"[C](=C)(c)a", 0.264, 4.305},
        {"[CH1](=C)a", 0.264, 4.305},
        {"[C]=c", 0.264, 4.305},
        {"[CX4][A;!C;!N;!O;!P;!S;!F;!Cl;!Br;!I;!#1]", 0.2148, 2.693},
        {"[#6]", 0.08129, 3.243},
        {"[#1][#6,#1]", 0.123, 1.057},
        {"[#1]O[CX4,c]", -0.2677, 1.395},
        {"[#1]O[!#6;!#7;!#8;!#16]", -0.2677, 1.395},
        {"[#1][!#6;!#7;!#8]", -0.2677, 1.395},
        {"[#1][#7]", 0.2142, 0.9627},
        {"[#1]O[#7]", 0.2142, 0.9627},
        {"[#1]OC=[#6,#7,O,S]", 0.298, 1.805},
        {"[#1]O[O,S]", 0.298, 1.805},
        {"[#1]", 0.1125, 1.112},
        {"[NH2+0][A;!#1]", -1.019, 2.262},
        {"[NH+0]([A;!#1])[A;!#1]", -0.7096, 2.173},
        {"[NH2+0]a", -1.027, 2.827},
        {"[NH1+0]([!#1;A,a])a", -0.5188, 3.0},
        {"[NH+0]=[!#1;A,a]", 0.08387, 1.757},
        {"[N+0](=[!#1;A,a])[!#1;A,a]", 0.1836, 2.428},
        {"[N+0]([A;!#1])([A;!#1])[A;!#1]", -0.3187, 1.839},
        {"[N+0](a)([!#1;A,a])[A;!#1]", -0.4458, 2.819},
        {"[N+0](a)(a)a", -0.4458, 2.819},
        {"[N+0]#[A;!#1]", 0.01508, 1.725},
        {"[NH3,NH2,NH;+,+2,+3]", -1.95, 0.0},
        {"[n+0]", -0.3239, 2.202},
        {"[n;+,+2,+3]", -1.119, 0.0},
        {"[NH0;+,+2,+3]([A;!#1])([A;!#1])([A;!#1])[A;!#1]", -0.3396, 0.2604},
        {"[NH0;+,+2,+3](=[A;!#1])([A;!#1])[!#1;A,a]", -0.3396, 0.2604},
        {"[NH0;+,+2,+3](=[#6])=[#7]", -0.3396, 0.2604},
        {"[N;+,+2,+3]#[A;!#1]", 0.2887, 3.359},
        {"[N;-,-2,-3]", 0.2887, 3.359},
        {"[N;+,+2,+3](=[N;-,-2,-3])=N", 0.2887, 3.359},
        {"[#7]", -0.4806, 2.134},
        {"[o]", 0.1552, 1.08},
        {"[OH,OH2]", -0.2893, 0.8238},
        {"[O]([A;!#1])[A;!#1]", -0.0684, 1.085},
        {"[O](a)[!#1;A,a]", -0.4195, 1.182},
        {"[O]=[#7,#8]", 0.0335, 3.367},
        {"[OX1;-,-2,-3][#7]", 0.0335, 3.367},
        {"[OX1;-,-2,-2][#16]", -0.3339, 0.7774},
        {"[O;-0]=[#16;-0]", -0.3339, 0.7774},
        {"[O-]C(=O)", -1.326, 0.0},
        {"[OX1;-,-2,-3][!#1;!N;!S]", -1.189, 0.0},
        {"[O]=c", 0.1788, 3.135},
        {"[O]=[CH]C", -0.1526, 0.0},
        {"[O]=C(C)([A;!#1])", -0.1526, 0.0},
        {"[O]=[CH][N,O]", -0.1526, 0.0},
        {"[O]=[CH2]", -0.1526, 0.0},
        {"[O]=[CX2]=O", -0.1526, 0.0},
        {"[O]=[CH]c", 0.1129, 0.2215},
        {"[O]=C([C,c])[a;!#1]", 0.1129, 0.2215},
        {"[O]=C(c)[A;!#1]", 0.1129, 0.2215},
        {"[O]=C([!#1;!#6])[!#1;!#6]", 0.4833, 0.389},
        {"[#8]", -0.1188, 0.6865},
        {"[#9-0]", 0.4202, 1.108},
        {"[#17-0]", 0.6895, 5.853},
        {"[#35-0]", 0.8456, 8.927},
        {"[#53-0]", 0.8857, 14.02},
        {"[#9,#17,#35,#53;-]", -2.996, 0.0},
        {"[#53;+,+2,+3]", -2.996, 0.0},
        {"[+;#3,#11,#19,#37,#55]", -2.996, 0.0},
        {"[#15]", 0.8612, 6.92},
        {"[S;-,-2,-3,-4,+1,+2,+3,+5,+6]", -0.0024, 7.365},
        {"[S-0]=[N,O,P,S]", -0.0024, 7.365},
        {"[S;A]", 0.6482, 7.591},
        {"[s;a]", 0.6237, 6.691},
        {"[#3,#11,#19,#37,#55]", -0.3808, 5.754},
        {"[#4,#12,#20,#38,#56]", -0.3808, 5.754},
        {"[#5,#13,#31,#49,#81]", -0.3808, 5.754},
        {"[#14,#32,#50,#82]", -0.3808, 5.754},
        {"[#33,#51,#83]", -0.3808, 5.754},
        {"[#34,#52,#84]", -0.3808, 5.754},
        {"[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30]", -0.0025, 0.0},
        {"[#39,#40,#41,#42,#43,#44,#45,#46,#47,#48]", -0.0025, 0.0},
        {"[#72,#73,#74,#75,#76,#77,#78,#79,#80]", -0.0025, 0.0},
    };
    return patterns;
}

std::pair<double, double> compute_crippen_descriptors(const OEChem::OEMolBase& mol) {
    OEChem::OEGraphMol crippen_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(crippen_mol);
    OEChem::OEAssignAromaticFlags(crippen_mol);
    OEChem::OEAssignHybridization(crippen_mol);
    OEChem::OEAddExplicitHydrogens(crippen_mol, false, false);

    std::unordered_set<unsigned int> assigned_atom_indices;
    double logp = 0.0;
    double mr = 0.0;
    for (const auto& pattern : crippen_patterns()) {
        OEChem::OESubSearch search(pattern.smarts);
        if (!search) {
            continue;
        }
        for (OESystem::OEIter<OEChem::OEMatchBase> match = search.Match(crippen_mol, false);
             match; ++match) {
            for (OESystem::OEIter<OEChem::OEMatchPair<OEChem::OEAtomBase>> atom_match =
                     match->GetAtoms();
                 atom_match; ++atom_match) {
                if (atom_match->pattern == nullptr || atom_match->target == nullptr
                    || atom_match->pattern->GetIdx() != 0u) {
                    continue;
                }
                const auto atom_index = atom_match->target->GetIdx();
                if (assigned_atom_indices.insert(atom_index).second) {
                    logp += pattern.logp;
                    mr += pattern.mr;
                }
            }
        }
    }
    return {logp, mr};
}

std::uint32_t count_mordred_acids(const OEChem::OEMolBase& mol) {
    static const std::vector<const char*> smarts_patterns{
        "[O;H1]-[C,S,P]=O",
        "[*;-;!$(*~[*;+])]",
        "[NH](S(=O)=O)C(F)(F)F",
        "n1nnnc1",
    };
    return count_unique_smarts_root_atoms(mol, smarts_patterns);
}

std::uint32_t count_mordred_bases(const OEChem::OEMolBase& mol) {
    static const std::vector<const char*> smarts_patterns{
        "[NH2]-[CX4]",
        "[NH](-[CX4])-[CX4]",
        "N(-[CX4])(-[CX4])-[CX4]",
        "[*;+;!$(*~[*;-])]",
        "N=C-N",
        "N-C=N",
    };
    return count_unique_smarts_root_atoms(mol, smarts_patterns);
}

void count_carbon_type(MordredFirstBatchValues& values, const OEChem::OEAtomBase& atom) {
    if (atom.GetAtomicNum() != 6u) {
        return;
    }

    const auto carbon_degree = carbon_neighbor_count(atom);
    const auto bucket = carbon_degree < 5u ? static_cast<std::size_t>(carbon_degree) : 4u;
    switch (atom.GetHyb()) {
    case OEChem::OEHybridization::sp:
        ++values.sp1_carbons_by_carbon_degree[bucket];
        break;
    case OEChem::OEHybridization::sp2:
        ++values.sp2_carbons_by_carbon_degree[bucket];
        ++values.sp2_carbons;
        break;
    case OEChem::OEHybridization::sp3:
        ++values.sp3_carbons_by_carbon_degree[bucket];
        ++values.sp3_carbons;
        break;
    default:
        break;
    }
}

MordredFirstBatchValues compute_first_batch_values(const OEChem::OEMolBase& mol) {
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignHybridization(working_mol);

    MordredFirstBatchValues values;
    OEChem::OEIsBridgeHead is_bridgehead(working_mol);
    values.acidic_groups = count_mordred_acids(working_mol);
    values.basic_groups = count_mordred_bases(working_mol);
    values.hbond_acceptors = OEMolProp::OEGetHBondAcceptorCount(working_mol);
    values.hbond_donors = OEMolProp::OEGetLipinskiDonorCount(working_mol);
    float topo_psa_no = 0.0f;
    if (OEMolProp::OEGet2dPSA(working_mol, topo_psa_no, nullptr, false)) {
        values.topo_psa_no = round_tpsa(static_cast<double>(topo_psa_no));
    }
    float topo_psa = 0.0f;
    if (OEMolProp::OEGet2dPSA(working_mol, topo_psa, nullptr, true)) {
        values.topo_psa = round_tpsa(static_cast<double>(topo_psa));
    }
    const auto crippen = compute_crippen_descriptors(working_mol);
    values.crippen_logp = crippen.first;
    values.crippen_mr = crippen.second;

    for (OESystem::OEIter<OEChem::OEAtomBase> atom = working_mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        if (is_hydrogen(*atom)) {
            ++values.hydrogens;
            values.exact_weight += atom_exact_mass(*atom);
            continue;
        }

        const auto total_h_count = static_cast<std::uint32_t>(atom->GetTotalHCount());
        ++values.heavy_atoms;
        values.hydrogens += total_h_count;
        values.exact_weight += atom_exact_mass(*atom)
                               + static_cast<double>(total_h_count)
                                     * default_isotopic_mass(1u);
        if (atom->IsAromatic()) {
            ++values.aromatic_atoms;
        }
        if (atomic_number != 6u) {
            ++values.hetero_atoms;
        }
        if (is_halogen(atomic_number)) {
            ++values.halogens;
        }
        if (!atom->IsAromatic() && is_bridgehead(*atom)) {
            ++values.bridgehead_atoms;
        }
        if (is_spiro_atom(*atom)) {
            ++values.spiro_atoms;
        }
        count_carbon_type(values, *atom);

        switch (atomic_number) {
        case 5u:
            ++values.boron;
            break;
        case 6u:
            ++values.carbon;
            break;
        case 7u:
            ++values.nitrogen;
            break;
        case 8u:
            ++values.oxygen;
            break;
        case 9u:
            ++values.fluorine;
            break;
        case 15u:
            ++values.phosphorus;
            break;
        case 16u:
            ++values.sulfur;
            break;
        case 17u:
            ++values.chlorine;
            break;
        case 35u:
            ++values.bromine;
            break;
        case 53u:
            ++values.iodine;
            break;
        default:
            break;
        }
    }

    for (OESystem::OEIter<OEChem::OEBondBase> bond = working_mol.GetBonds(); bond; ++bond) {
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
            continue;
        }

        ++values.heavy_bonds;
        if (is_mordred_rotatable_bond(*bond)) {
            ++values.rotatable_bonds;
        }

        const bool aromatic = bond->IsAromatic();
        const auto order = bond->GetOrder();
        if (aromatic) {
            ++values.aromatic_bonds;
            ++values.multiple_heavy_bonds;
        } else if (order == 1u) {
            ++values.single_heavy_bonds;
        } else {
            ++values.multiple_heavy_bonds;
            if (order == 2u) {
                ++values.double_heavy_bonds;
            } else if (order == 3u) {
                ++values.triple_heavy_bonds;
            }
        }
    }

    return values;
}

using AtomicPropertyGetter = std::optional<double> (*)(std::uint32_t);

OEChem::OEGraphMol explicit_hydrogen_copy(const OEChem::OEMolBase& mol) {
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    OEChem::OEAddExplicitHydrogens(working_mol, false, false);
    return working_mol;
}

std::optional<double> normalized_property_sum(
    const OEChem::OEMolBase& mol,
    AtomicPropertyGetter getter) {
    const auto carbon = getter(6u);
    if (!carbon.has_value()) {
        return std::nullopt;
    }

    double sum = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto value = getter(static_cast<std::uint32_t>(atom->GetAtomicNum()));
        if (!value.has_value()) {
            return std::nullopt;
        }
        sum += *value / *carbon;
    }
    return sum;
}

std::optional<double> mean_from_sum(std::optional<double> sum, std::uint32_t atom_count) {
    if (!sum.has_value() || atom_count == 0u) {
        return std::nullopt;
    }
    return *sum / static_cast<double>(atom_count);
}

std::uint32_t count_atoms(const OEChem::OEMolBase& mol) {
    std::uint32_t count = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        ++count;
    }
    return count;
}

std::uint32_t count_bonds(const OEChem::OEMolBase& mol) {
    std::uint32_t count = 0u;
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        ++count;
    }
    return count;
}

std::optional<double> sum_atom_property(
    const OEChem::OEMolBase& mol,
    AtomicPropertyGetter getter) {
    double sum = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto value = getter(static_cast<std::uint32_t>(atom->GetAtomicNum()));
        if (!value.has_value()) {
            return std::nullopt;
        }
        sum += *value;
    }
    return sum;
}

std::optional<double> compute_bpol(const OEChem::OEMolBase& mol) {
    double sum = 0.0;
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr) {
            return std::nullopt;
        }

        const auto begin_value =
            mordred_polarizability94(static_cast<std::uint32_t>(begin->GetAtomicNum()));
        const auto end_value =
            mordred_polarizability94(static_cast<std::uint32_t>(end->GetAtomicNum()));
        if (!begin_value.has_value() || !end_value.has_value()) {
            return std::nullopt;
        }
        sum += std::abs(*begin_value - *end_value);
    }
    return sum;
}

std::uint32_t ring_basis_count(const OEChem::OEMolBase& mol, bool aromatic_only) {
    // Mordred's Vabc depends on RDKit SymmSSSR ring counts. OpenEye exposes
    // ring systems, so compute the cycle rank of the ring-bond subgraph to
    // preserve fused-ring penalties without depending on row-specific fixes.
    std::unordered_map<unsigned int, std::size_t> atom_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (!atom->IsInRing()) {
            continue;
        }
        if (aromatic_only && !atom->IsAromatic()) {
            continue;
        }
        atom_indices.emplace(atom->GetIdx(), atom_indices.size());
    }

    std::vector<std::vector<std::size_t>> adjacency(atom_indices.size());
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        if (!bond->IsInRing()) {
            continue;
        }
        if (aromatic_only && !bond->IsAromatic()) {
            continue;
        }

        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr) {
            continue;
        }
        const auto begin_index = atom_indices.find(begin->GetIdx());
        const auto end_index = atom_indices.find(end->GetIdx());
        if (begin_index == atom_indices.end() || end_index == atom_indices.end()) {
            continue;
        }
        adjacency[begin_index->second].push_back(end_index->second);
        adjacency[end_index->second].push_back(begin_index->second);
    }

    std::vector<bool> visited(adjacency.size(), false);
    std::uint32_t rings = 0u;
    for (std::size_t start = 0u; start < adjacency.size(); ++start) {
        if (visited[start] || adjacency[start].empty()) {
            continue;
        }

        std::vector<std::size_t> stack{start};
        visited[start] = true;
        std::uint32_t vertices = 0u;
        std::uint32_t doubled_edges = 0u;
        while (!stack.empty()) {
            const auto current = stack.back();
            stack.pop_back();
            ++vertices;
            doubled_edges += static_cast<std::uint32_t>(adjacency[current].size());
            for (const auto neighbor : adjacency[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    stack.push_back(neighbor);
                }
            }
        }

        const auto edges = doubled_edges / 2u;
        if (edges >= vertices) {
            rings += edges - vertices + 1u;
        }
    }
    return rings;
}

std::optional<double> compute_vabc(
    const OEChem::OEMolBase& explicit_mol,
    const OEChem::OEMolBase& ring_mol) {
    const auto atom_volume = sum_atom_property(explicit_mol, bondi_atom_volume);
    if (!atom_volume.has_value()) {
        return std::nullopt;
    }

    const auto bond_count = count_bonds(explicit_mol);
    const auto aromatic_rings = ring_basis_count(ring_mol, true);
    const auto total_rings = ring_basis_count(ring_mol, false);
    const auto aliphatic_rings =
        total_rings > aromatic_rings ? total_rings - aromatic_rings : 0u;
    return *atom_volume - 5.92 * static_cast<double>(bond_count)
           - 14.7 * static_cast<double>(aromatic_rings)
           - 3.8 * static_cast<double>(aliphatic_rings);
}

MordredAdditivePropertyValues compute_additive_property_values(const OEChem::OEMolBase& mol) {
    const auto explicit_mol = explicit_hydrogen_copy(mol);
    OEChem::OEGraphMol ring_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(ring_mol);
    OEChem::OEAssignAromaticFlags(ring_mol);

    MordredAdditivePropertyValues values;
    const auto atom_count = count_atoms(explicit_mol);
    values.sz = normalized_property_sum(explicit_mol, [](std::uint32_t atomic_number) {
        return std::optional<double>(static_cast<double>(atomic_number));
    });
    values.sm = normalized_property_sum(explicit_mol, mordred_mass);
    values.sv = normalized_property_sum(explicit_mol, mordred_vdw_volume);
    values.sse = normalized_property_sum(explicit_mol, mordred_sanderson);
    values.spe = normalized_property_sum(explicit_mol, mordred_pauling);
    values.sare = normalized_property_sum(explicit_mol, mordred_allred_rocow);
    values.sp = normalized_property_sum(explicit_mol, mordred_polarizability94);
    values.si = normalized_property_sum(explicit_mol, mordred_ionization_potential);
    values.m_z = mean_from_sum(values.sz, atom_count);
    values.mm = mean_from_sum(values.sm, atom_count);
    values.mv = mean_from_sum(values.sv, atom_count);
    values.mse = mean_from_sum(values.sse, atom_count);
    values.mpe = mean_from_sum(values.spe, atom_count);
    values.mare = mean_from_sum(values.sare, atom_count);
    values.mp = mean_from_sum(values.sp, atom_count);
    values.mi = mean_from_sum(values.si, atom_count);

    const auto mc_gowan_atom_volume = sum_atom_property(explicit_mol, mordred_mc_gowan_volume);
    if (mc_gowan_atom_volume.has_value()) {
        values.v_mcgowan =
            *mc_gowan_atom_volume - static_cast<double>(count_bonds(explicit_mol)) * 6.56;
    }
    values.apol = sum_atom_property(explicit_mol, mordred_polarizability94);
    values.bpol = compute_bpol(explicit_mol);
    values.vabc = compute_vabc(explicit_mol, ring_mol);
    return values;
}

void set_int(DescriptorSetBuilder& builder, const std::string& name, std::uint32_t value) {
    builder.Set(name, DescriptorValue::Int(static_cast<std::int64_t>(value)));
}

void set_float(DescriptorSetBuilder& builder, const std::string& name, double value) {
    builder.Set(name, DescriptorValue::Float(value));
}

void set_optional_float(
    DescriptorSetBuilder& builder,
    const std::string& name,
    std::optional<double> value) {
    if (value.has_value()) {
        set_float(builder, name, *value);
    }
}

void set_bool(DescriptorSetBuilder& builder, const std::string& name, bool value) {
    builder.Set(name, DescriptorValue::Bool(value));
}

} // namespace

DescriptorSet MakeMordredDescriptors(const OEChem::OEMolBase& mol) {
    const auto values = compute_first_batch_values(mol);
    const auto additive_values = compute_additive_property_values(mol);
    DescriptorSetBuilder builder(MordredDescriptorSchema());

    const auto all_atoms = values.heavy_atoms + values.hydrogens;
    const auto all_bonds = values.heavy_bonds + values.hydrogens;
    const auto all_single_bonds = values.single_heavy_bonds + values.hydrogens;

    // Mordred's kekulized bond counts use RDKit's alternating aromatic form.
    // For the supported count subset, this parity approximation matches the
    // copied Mordred references and keeps aromatic descriptors deterministic.
    const auto kekulized_aromatic_double_bonds = values.aromatic_bonds / 2u;
    const auto kekulized_aromatic_single_bonds =
        values.aromatic_bonds - kekulized_aromatic_double_bonds;
    const auto kekulized_single_bonds =
        values.single_heavy_bonds + kekulized_aromatic_single_bonds + values.hydrogens;
    const auto kekulized_double_bonds =
        values.double_heavy_bonds + kekulized_aromatic_double_bonds;

    set_int(builder, "nAcid", values.acidic_groups);
    set_int(builder, "nBase", values.basic_groups);
    set_int(builder, "nAromAtom", values.aromatic_atoms);
    set_int(builder, "nAromBond", values.aromatic_bonds);
    set_int(builder, "nAtom", all_atoms);
    set_int(builder, "nHeavyAtom", values.heavy_atoms);
    set_int(builder, "nSpiro", values.spiro_atoms);
    set_int(builder, "nBridgehead", values.bridgehead_atoms);
    set_int(builder, "nHetero", values.hetero_atoms);
    set_int(builder, "nH", values.hydrogens);
    set_int(builder, "nB", values.boron);
    set_int(builder, "nC", values.carbon);
    set_int(builder, "nN", values.nitrogen);
    set_int(builder, "nO", values.oxygen);
    set_int(builder, "nS", values.sulfur);
    set_int(builder, "nP", values.phosphorus);
    set_int(builder, "nF", values.fluorine);
    set_int(builder, "nCl", values.chlorine);
    set_int(builder, "nBr", values.bromine);
    set_int(builder, "nI", values.iodine);
    set_int(builder, "nX", values.halogens);
    set_int(builder, "nBonds", all_bonds);
    set_int(builder, "nBondsO", values.heavy_bonds);
    set_int(builder, "nBondsS", all_single_bonds);
    set_int(builder, "nBondsD", values.double_heavy_bonds);
    set_int(builder, "nBondsT", values.triple_heavy_bonds);
    set_int(builder, "nBondsA", values.aromatic_bonds);
    set_int(builder, "nBondsM", values.multiple_heavy_bonds);
    set_int(builder, "nBondsKS", kekulized_single_bonds);
    set_int(builder, "nBondsKD", kekulized_double_bonds);
    set_int(builder, "C1SP1", values.sp1_carbons_by_carbon_degree[1]);
    set_int(builder, "C2SP1", values.sp1_carbons_by_carbon_degree[2]);
    set_int(builder, "C1SP2", values.sp2_carbons_by_carbon_degree[1]);
    set_int(builder, "C2SP2", values.sp2_carbons_by_carbon_degree[2]);
    set_int(builder, "C3SP2", values.sp2_carbons_by_carbon_degree[3]);
    set_int(builder, "C1SP3", values.sp3_carbons_by_carbon_degree[1]);
    set_int(builder, "C2SP3", values.sp3_carbons_by_carbon_degree[2]);
    set_int(builder, "C3SP3", values.sp3_carbons_by_carbon_degree[3]);
    set_int(builder, "C4SP3", values.sp3_carbons_by_carbon_degree[4]);

    const auto sp2_sp3_carbons = values.sp2_carbons + values.sp3_carbons;
    if (sp2_sp3_carbons != 0u) {
        set_float(
            builder,
            "HybRatio",
            static_cast<double>(values.sp3_carbons) / static_cast<double>(sp2_sp3_carbons));
    }
    set_float(
        builder,
        "FCSP3",
        values.carbon == 0u
            ? 0.0
            : static_cast<double>(values.sp3_carbons) / static_cast<double>(values.carbon));
    set_int(builder, "nHBAcc", values.hbond_acceptors);
    set_int(builder, "nHBDon", values.hbond_donors);

    set_int(builder, "nRot", values.rotatable_bonds);
    if (values.heavy_bonds != 0u) {
        set_float(
            builder,
            "RotRatio",
            static_cast<double>(values.rotatable_bonds) / static_cast<double>(values.heavy_bonds));
    }
    set_float(builder, "SLogP", values.crippen_logp);
    set_float(builder, "SMR", values.crippen_mr);
    set_optional_float(builder, "SZ", additive_values.sz);
    set_optional_float(builder, "Sm", additive_values.sm);
    set_optional_float(builder, "Sv", additive_values.sv);
    set_optional_float(builder, "Sse", additive_values.sse);
    set_optional_float(builder, "Spe", additive_values.spe);
    set_optional_float(builder, "Sare", additive_values.sare);
    set_optional_float(builder, "Sp", additive_values.sp);
    set_optional_float(builder, "Si", additive_values.si);
    set_optional_float(builder, "MZ", additive_values.m_z);
    set_optional_float(builder, "Mm", additive_values.mm);
    set_optional_float(builder, "Mv", additive_values.mv);
    set_optional_float(builder, "Mse", additive_values.mse);
    set_optional_float(builder, "Mpe", additive_values.mpe);
    set_optional_float(builder, "Mare", additive_values.mare);
    set_optional_float(builder, "Mp", additive_values.mp);
    set_optional_float(builder, "Mi", additive_values.mi);
    set_optional_float(builder, "VMcGowan", additive_values.v_mcgowan);
    set_optional_float(builder, "apol", additive_values.apol);
    set_optional_float(builder, "bpol", additive_values.bpol);
    set_optional_float(builder, "Vabc", additive_values.vabc);
    set_float(builder, "TopoPSA(NO)", values.topo_psa_no);
    set_float(builder, "TopoPSA", values.topo_psa);
    set_float(builder, "MW", values.exact_weight);
    set_float(
        builder,
        "AMW",
        all_atoms == 0u ? 0.0 : values.exact_weight / static_cast<double>(all_atoms));
    set_bool(
        builder,
        "Lipinski",
        values.hbond_donors <= 5u && values.hbond_acceptors <= 10u
            && values.exact_weight <= 500.0 && values.crippen_logp <= 5.0);
    set_bool(
        builder,
        "GhoseFilter",
        values.exact_weight >= 160.0 && values.exact_weight <= 480.0
            && all_atoms >= 20u && all_atoms <= 70u
            && values.crippen_logp >= -0.4 && values.crippen_logp <= 5.6
            && values.crippen_mr >= 40.0 && values.crippen_mr <= 130.0);

    return builder.Build();
}

} // namespace OEFP
