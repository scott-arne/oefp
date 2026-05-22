#include "oefp/mordred.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <oesystem.h>
#include <oemolprop.h>
#include <oeomega2.h>

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

struct MordredLabuteAsaValues {
    double total = 0.0;
    std::vector<double> atom_contributions;
    std::vector<unsigned int> atom_ids;
};

struct MordredCrippenAtomContributions {
    std::vector<double> logp;
    std::vector<double> mr;
    std::vector<unsigned int> atom_ids;
};

struct MordredGasteigerAtomCharges {
    std::vector<double> charges;
    std::vector<double> hydrogen_charges;
    std::vector<unsigned int> atom_ids;
};

struct MordredGasteigerParameters {
    const char* element;
    const char* mode;
    double a;
    double b;
    double c;
};

struct MordredLogSPattern {
    const char* smarts;
    double coefficient;
};

std::optional<int> mordred_outer_electrons(std::uint32_t atomic_number);
std::uint32_t mordred_principal_quantum_number(std::uint32_t atomic_number);

struct MordredWalkCountValues {
    std::array<double, 11> mwc{};
    std::array<double, 11> srw{};
    double total_mwc10 = 0.0;
    double total_srw10 = 0.0;
};

struct MordredPathCountValues {
    std::array<std::uint32_t, 11> mpc{};
    std::array<double, 11> pi_mpc{};
    std::uint32_t total_mpc10 = 0u;
    double total_pi_mpc10 = 0.0;
};

struct MordredChiPathValues {
    std::array<std::optional<double>, 8> xp_d{};
    std::array<std::optional<double>, 8> axp_d{};
    std::array<std::optional<double>, 8> xp_dv{};
    std::array<std::optional<double>, 8> axp_dv{};
};

struct MordredChiNonPathValues {
    std::array<std::optional<double>, 8> xch_d{};
    std::array<std::optional<double>, 8> xch_dv{};
    std::array<std::optional<double>, 7> xc_d{};
    std::array<std::optional<double>, 7> xc_dv{};
    std::array<std::optional<double>, 7> xpc_d{};
    std::array<std::optional<double>, 7> xpc_dv{};
};

struct MordredZagrebValues {
    double zagreb1 = 0.0;
    double zagreb2 = 0.0;
    std::optional<double> modified_zagreb1;
    double modified_zagreb2 = 0.0;
};

struct MordredABCIndexValues {
    double abc = 0.0;
    double abcgg = 0.0;
};

struct MordredCPSAChargeValues {
    double rncg = 0.0;
    double rpcg = 0.0;
};

struct MordredMolecularDistanceEdgeValues {
    std::array<std::array<std::optional<double>, 5>, 5> carbon{};
    std::array<std::array<std::optional<double>, 5>, 5> oxygen{};
    std::array<std::array<std::optional<double>, 5>, 5> nitrogen{};
};

struct MordredAutocorrelationAtomSnapshot {
    std::uint32_t atomic_number = 0u;
    int formal_charge = 0;
    std::uint32_t total_hydrogens = 0u;
    std::uint32_t non_hydrogen_neighbors = 0u;
};

struct MordredExplicitAtomDistanceValues {
    std::vector<MordredAutocorrelationAtomSnapshot> atoms;
    std::vector<std::uint32_t> atomic_numbers;
    std::vector<unsigned int> atom_ids;
    std::vector<std::vector<std::int64_t>> distances;
    std::optional<std::vector<double>> gasteiger_charges;
};

struct MordredAutocorrelationValues {
    const char* suffix = "";
    std::array<std::optional<double>, 9> ats{};
    std::array<std::optional<double>, 9> aats{};
    std::array<std::optional<double>, 9> atsc{};
    std::array<std::optional<double>, 9> aatsc{};
    std::array<std::optional<double>, 9> mats{};
    std::array<std::optional<double>, 9> gats{};
};

struct MordredEtaNeighbor {
    std::size_t atom_index;
    std::uint32_t bond_order;
    bool aromatic;
};

struct MordredEtaGraph {
    std::vector<const OEChem::OEAtomBase*> atoms;
    std::vector<std::vector<MordredEtaNeighbor>> adjacency;
};

struct MordredEtaValue {
    const char* name;
    std::optional<double> value;
};

using MordredEtaValues = std::vector<MordredEtaValue>;

struct MordredWienerValues {
    std::int64_t wpath = 0;
    std::int64_t wpol = 0;
};

struct MordredTopologicalIndexValues {
    std::optional<std::int64_t> diameter;
    std::optional<std::int64_t> radius;
    std::optional<double> topo_shape_index;
    std::optional<double> petitjean_index;
};

struct MordredTopologicalChargeValues {
    std::array<double, 11> raw{};
    std::array<double, 11> mean{};
    double global10 = 0.0;
};

struct MordredMolecularIdValues {
    std::size_t atom_count = 0u;
    double any = 0.0;
    double hetero = 0.0;
    double carbon = 0.0;
    double nitrogen = 0.0;
    double oxygen = 0.0;
    double halogen = 0.0;
};

struct MordredMatrixEigenvalueValues {
    double spectral_absolute = 0.0;
    double spectral_max = 0.0;
    double spectral_diameter = 0.0;
    double spectral_absolute_deviation = 0.0;
    double spectral_mean_absolute_deviation = 0.0;
    double log_estrada_like = 0.0;
    double spectral_moment = 0.0;
    double eigenvector_coefficient_sum = 0.0;
    double eigenvector_coefficient_mean = 0.0;
    double eigenvector_coefficient_log = 0.0;
    double randic_eigenvector_sum = 0.0;
    double randic_eigenvector_mean = 0.0;
    std::optional<double> randic_eigenvector_log;
};

struct MordredDetourMatrixValues {
    MordredMatrixEigenvalueValues matrix;
    std::int64_t detour_index = 0;
};

struct MordredDetourBlock {
    std::vector<std::size_t> nodes;
    std::vector<std::int64_t> longest_paths;
};

struct MordredDetourSearchContext {
    std::uint64_t max_operations = 0u;
    std::uint64_t operations = 0u;
};

using MordredAtomicPropertyLookup = std::optional<double> (*)(std::uint32_t);
using MordredAutocorrelationAtomPropertyLookup =
    std::optional<double> (*)(const MordredAutocorrelationAtomSnapshot&);

struct MordredBaryszMatrixProperty {
    const char* suffix;
    double carbon_reference;
    MordredAtomicPropertyLookup lookup;
};

struct MordredMoRSEProperty {
    const char* suffix;
    double carbon_reference;
    MordredAtomicPropertyLookup lookup;
};

const std::array<MordredMoRSEProperty, 5>& mordred_morse_properties();

struct MordredAutocorrelationProperty {
    const char* suffix;
    MordredAtomicPropertyLookup atomic_number_lookup;
    MordredAutocorrelationAtomPropertyLookup atom_lookup;
    bool gasteiger_charge = false;
};

enum class MordredBCUTPropertyKind {
    GasteigerCharge,
    ValenceElectrons,
    SigmaElectrons,
    IntrinsicState,
    AtomicNumber,
    Mass,
    VdwVolume,
    Sanderson,
    Pauling,
    AllredRocow,
    Polarizability,
    IonizationPotential,
};

struct MordredBCUTProperty {
    const char* suffix;
    MordredBCUTPropertyKind kind;
};

struct MordredBCUTValues {
    const char* suffix = "";
    std::optional<double> high;
    std::optional<double> low;
};

struct MordredSymmetricEigensystem {
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
};

struct MordredRingCountSummary {
    std::uint32_t total = 0u;
    std::array<std::uint32_t, 13> by_size{};
    std::uint32_t greater_or_equal_12 = 0u;
};

struct MordredRingCountValues {
    MordredRingCountSummary all;
    MordredRingCountSummary hetero;
    MordredRingCountSummary aromatic;
    MordredRingCountSummary aromatic_hetero;
    MordredRingCountSummary aliphatic;
    MordredRingCountSummary aliphatic_hetero;
};

struct MordredRingCountValueSets {
    MordredRingCountValues base;
    MordredRingCountValues fused;
};

struct MordredRingAtomProperties {
    std::uint32_t atomic_number;
    bool aromatic;
};

struct MordredEStatePattern {
    const char* descriptor_name;
    const char* smarts;
};

struct MordredEStateValues {
    std::vector<std::uint32_t> counts;
    std::vector<double> sums;
    std::vector<std::optional<double>> maxima;
    std::vector<std::optional<double>> minima;
};

struct Mordred3DAtom {
    unsigned int atom_index = 0u;
    std::uint32_t atomic_number = 0u;
    double mass = 0.0;
    std::array<double, 3> coord{};
};

struct Mordred3DAtomSet {
    std::vector<Mordred3DAtom> atoms;
    std::vector<double> distances;
    std::vector<unsigned char> adjacency;
};

struct Mordred3DContext {
    Mordred3DAtomSet all_atom_set;
    Mordred3DAtomSet heavy_atom_set;
};

struct MordredGeometricalIndexValues {
    double diameter = 0.0;
    double radius = 0.0;
    std::optional<double> shape_index;
    std::optional<double> petitjean_index;
};

struct MordredGravitationalIndexValues {
    std::optional<double> heavy;
    std::optional<double> all;
    std::optional<double> heavy_pair;
    std::optional<double> all_pair;
};

struct MordredMomentOfInertiaValues {
    std::array<double, 3> axes{};
};

struct MordredLowCount3DValues {
    std::optional<MordredGeometricalIndexValues> geometrical;
    MordredGravitationalIndexValues gravitational;
    std::optional<MordredMomentOfInertiaValues> moment_of_inertia;
    std::optional<double> pbf;
};

struct MordredMoRSEValues {
    const char* suffix = "";
    std::array<std::optional<double>, 33> values{};
};

struct AtomicPropertyValue {
    std::uint32_t atomic_number;
    double value;
};

constexpr double kPi = 3.14159265358979323846;
constexpr double kMissingAtomicProperty = -1.0;
// RDKit's GetDistanceMatrix uses this sentinel for disconnected atom pairs.
constexpr std::int64_t kMordredDisconnectedDistance = 100000000;

constexpr std::array<double, 111> kMordredMassValues{{
    kMissingAtomicProperty, 1.008, 4.002602, 6.94, 9.012182, 10.81,
    12.011, 14.007, 15.999, 18.9984032, 20.1797, 22.98976928,
    24.305, 26.9815386, 28.085, 30.973762, 32.06, 35.45,
    39.948, 39.0983, 40.078, 44.955912, 47.867, 50.9415,
    51.9961, 54.938045, 55.845, 58.933195, 58.6934, 63.546,
    65.38, 69.723, 72.63, 74.9216, 78.96, 79.904,
    83.798, 85.4678, 87.62, 88.90585, 91.224, 92.90638,
    95.96, 98.0, 101.07, 102.9055, 106.42, 107.8682,
    112.411, 114.818, 118.71, 121.76, 127.6, 126.90447,
    131.293, 132.9054519, 137.327, 138.90547, 140.116, 140.90765,
    144.242, 145.0, 150.36, 151.964, 157.25, 158.92535,
    162.5, 164.93032, 167.259, 168.93421, 173.054, 174.9668,
    178.49, 180.94788, 183.84, 186.207, 190.23, 192.217,
    195.084, 196.966569, 200.59, 204.38, 207.2, 208.9804,
    210.0, 210.0, 222.0, 223.0, 226.0, 227.0,
    232.03806, 231.03588, 238.02891, 237.0, 244.0, 243.0,
    247.0, 247.0, 251.0, 252.0, 257.0, 258.0,
    259.0, 262.0, 261.0, 262.0, 266.0, 264.0,
    269.0, 268.0, 271.0,
}};

// RDKit MolWt uses PeriodicTable::getAtomicWeight; these rounded weights differ
// from OpenEye averages enough to affect FilterItLogS parity.
constexpr std::array<double, 119> kRDKitAtomicWeights{{
    0.0,     1.008,   4.003,   6.941,   9.012,   10.812,  12.011,
    14.007,  15.999,  18.998,  20.18,   22.99,   24.305,  26.982,
    28.086,  30.974,  32.067,  35.453,  39.948,  39.098,  40.078,
    44.956,  47.867,  50.944,  51.996,  54.938,  55.845,  58.933,
    58.693,  63.546,  65.39,   69.723,  72.61,   74.922,  78.96,
    79.904,  83.8,    85.468,  87.62,   88.906,  91.224,  92.906,
    95.94,   98.0,    101.07,  102.906, 106.42,  107.868, 112.412,
    114.818, 118.711, 121.76,  127.6,   126.904, 131.29,  132.905,
    137.328, 138.906, 140.116, 140.908, 144.24,  145.0,   150.36,
    151.964, 157.25,  158.925, 162.5,   164.93,  167.26,  168.934,
    173.04,  174.967, 178.49,  180.948, 183.84,  186.207, 190.23,
    192.217, 195.078, 196.967, 200.59,  204.383, 207.2,   208.98,
    209.0,   210.0,   222.0,   223.0,   226.0,   227.0,   232.038,
    231.036, 238.029, 237.0,   244.0,   243.0,   247.0,   247.0,
    251.0,   252.0,   257.0,   258.0,   259.0,   262.0,   267.0,
    268.0,   269.0,   270.0,   269.0,   278.0,   281.0,   281.0,
    285.0,   284.0,   289.0,   288.0,   293.0,   292.0,   294.0,
}};

constexpr std::array<double, 104> kMordredVdwRadii{{
    kMissingAtomicProperty, 1.1, 1.4, 1.82, 1.53, 1.92,
    1.7, 1.55, 1.52, 1.47, 1.54, 2.27,
    1.73, 1.84, 2.1, 1.8, 1.8, 1.75,
    1.88, 2.75, 2.31, 2.15, 2.11, 2.07,
    2.06, 2.05, 2.04, 2.0, 1.97, 1.96,
    2.01, 1.87, 2.11, 1.85, 1.9, 1.85,
    2.02, 3.03, 2.49, 2.32, 2.23, 2.18,
    2.17, 2.16, 2.13, 2.1, 2.1, 2.11,
    2.18, 1.93, 2.17, 2.06, 2.06, 1.98,
    2.16, 3.43, 2.68, 2.43, 2.42, 2.4,
    2.39, 2.38, 2.36, 2.35, 2.34, 2.33,
    2.31, 2.3, 2.29, 2.27, 2.26, 2.24,
    2.23, 2.22, 2.18, 2.16, 2.16, 2.13,
    2.13, 2.14, 2.23, 1.96, 2.02, 2.07,
    1.97, 2.02, 2.2, 3.48, 2.83, 2.47,
    2.45, 2.43, 2.41, 2.39, 2.43, 2.44,
    2.45, 2.44, 2.45, 2.45, 2.45, 2.46,
    2.46, 2.46,
}};

constexpr std::array<double, 84> kMordredSandersonValues{{
    kMissingAtomicProperty, 2.592, kMissingAtomicProperty, 0.67, 1.81, 2.275,
    2.746, 3.194, 3.654, 4.0, 4.5, 0.56,
    1.318, 1.714, 2.138, 2.515, 2.957, 3.475,
    3.31, 0.445, 0.946, 1.02, 1.09, 1.39,
    1.66, 2.2, 2.2, 2.56, 1.94, 2.033,
    2.223, 2.419, 2.618, 2.816, 3.014, 3.219,
    2.91, 0.312, 0.721, 0.65, 0.9, 1.42,
    1.15, kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, 1.826, 1.978, 2.138, 2.298, 2.458,
    2.618, 2.778, 2.34, 0.22, 0.651, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, 0.98, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty, 2.195,
    2.246, 2.291, 2.342,
}};

constexpr std::array<double, 103> kMordredPaulingValues{{
    kMissingAtomicProperty, 2.2, kMissingAtomicProperty, 0.98, 1.57, 2.04,
    2.55, 3.04, 3.44, 3.98, kMissingAtomicProperty, 0.93,
    1.31, 1.61, 1.9, 2.19, 2.58, 3.16,
    kMissingAtomicProperty, 0.82, 1.0, 1.36, 1.54, 1.63,
    1.66, 1.55, 1.83, 1.88, 1.91, 1.9,
    1.65, 1.81, 2.01, 2.18, 2.55, 2.96,
    3.0, 0.82, 0.95, 1.22, 1.33, 1.6,
    2.16, 1.9, 2.2, 2.28, 2.2, 1.93,
    1.69, 1.78, 1.96, 2.05, 2.1, 2.66,
    2.6, 0.79, 0.89, 1.1, 1.12, 1.13,
    1.14, kMissingAtomicProperty, 1.17, kMissingAtomicProperty, 1.2,
    kMissingAtomicProperty, 1.22, 1.23, 1.24, 1.25,
    kMissingAtomicProperty, 1.27, 1.3, 1.5, 2.36, 1.9,
    2.2, 2.2, 2.28, 2.54, 2.0, 1.62,
    2.33, 2.02, 2.0, 2.2, kMissingAtomicProperty, 0.7,
    0.9, 1.1, 1.3, 1.5, 1.38, 1.36,
    1.28, 1.3, 1.3, 1.3, 1.3, 1.3,
    1.3, 1.3, 1.3,
}};

constexpr std::array<double, 86> kMordredAllredRocowValues{{
    kMissingAtomicProperty, 2.2, kMissingAtomicProperty, 0.97, 1.47, 2.01,
    2.5, 3.07, 3.5, 4.1, kMissingAtomicProperty, 1.01,
    1.23, 1.47, 1.74, 2.06, 2.44, 2.83,
    kMissingAtomicProperty, 0.91, 1.04, 1.2, 1.32, 1.45,
    1.56, 1.6, 1.64, 1.7, 1.75, 1.75,
    1.66, 1.82, 2.02, 2.2, 2.48, 2.74,
    kMissingAtomicProperty, 0.89, 0.99, 1.11, 1.22, 1.23,
    1.3, 1.36, 1.42, 1.45, 1.35, 1.42,
    1.46, 1.49, 1.72, 1.82, 2.01, 2.21,
    kMissingAtomicProperty, 0.86, 0.97, 1.08, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, 1.23, 1.33, 1.4, 1.46, 1.52,
    1.55, 1.44, 1.42, 1.44, 1.44, 1.55,
    1.67, 1.76, 1.9,
}};

constexpr std::array<double, 120> kMordredPolarizability94Values{{
    kMissingAtomicProperty, 0.666793, 0.2050522, 24.33, 5.6, 3.03,
    1.67, 1.1, 0.802, 0.557, 0.39432, 24.11,
    10.6, 6.8, 5.53, 3.63, 2.9, 2.18,
    1.6411, 43.06, 22.8, 17.8, 14.6, 12.4,
    11.6, 9.4, 8.4, 7.5, 6.8, 6.2,
    5.75, 8.12, 5.84, 4.31, 3.77, 3.05,
    2.4844, 47.24, 23.5, 22.7, 17.9, 15.7,
    12.8, 11.4, 9.6, 8.6, 4.8, 6.78,
    7.36, 10.2, 7.84, 6.6, 5.5, 5.35,
    4.044, 59.42, 39.7, 31.1, 29.6, 28.2,
    31.4, 30.1, 28.8, 27.7, 23.5, 25.5,
    24.5, 23.6, 22.7, 21.8, 20.9, 21.9,
    16.2, 13.1, 11.1, 9.7, 8.5, 7.6,
    6.5, 5.8, 5.02, 7.6, 7.01, 7.4,
    6.8, 6.0, 5.3, 48.6, 38.3, 32.1,
    32.1, 25.4, 24.9, 24.8, 24.5, 23.3,
    23.0, 22.7, 20.5, 19.7, 23.8, 18.2,
    16.4, kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, kMissingAtomicProperty, 4.06,
    kMissingAtomicProperty, 4.59, kMissingAtomicProperty, kMissingAtomicProperty,
    kMissingAtomicProperty, kMissingAtomicProperty, 24.26,
}};

constexpr std::array<double, 105> kMordredIonizationPotentialValues{{
    kMissingAtomicProperty, 13.598443, 24.587387, 5.391719, 9.3227, 8.29802,
    11.2603, 14.5341, 13.61805, 17.4228, 21.56454, 5.139076,
    7.646235, 5.985768, 8.15168, 10.48669, 10.36001, 12.96763,
    15.75961, 4.3406633, 6.11316, 6.56149, 6.82812, 6.74619,
    6.76651, 7.43402, 7.9024, 7.88101, 7.6398, 7.72638,
    9.394199, 5.999301, 7.89943, 9.7886, 9.75239, 11.8138,
    13.99961, 4.177128, 5.69485, 6.2173, 6.6339, 6.75885,
    7.09243, 7.28, 7.3605, 7.4589, 8.3369, 7.57623,
    8.99382, 5.78636, 7.34392, 8.60839, 9.0096, 10.45126,
    12.12984, 3.893905, 5.211664, 5.5769, 5.5387, 5.473,
    5.525, 5.582, 5.6437, 5.67038, 6.1498, 5.8638,
    5.9389, 6.0215, 6.1077, 6.18431, 6.25416, 5.42586,
    6.82507, 7.54957, 7.86403, 7.83352, 8.43823, 8.96702,
    8.9588, 9.22553, 10.4375, 6.108194, 7.41663, 7.2855,
    8.414, kMissingAtomicProperty, 10.7485, 4.072741, 5.278423, 5.17,
    6.3067, 5.89, 6.1941, 6.2657, 6.026, 5.9738,
    5.9914, 6.1979, 6.2817, 6.42, 6.5, 6.58,
    6.65, 4.9, 6.0,
}};

constexpr std::array<double, 104> kMordredMcGowanVolumeValues{{
    kMissingAtomicProperty, 8.71, 6.75, 22.23, 20.27, 18.31,
    16.35, 14.39, 12.43, 10.47, 8.51, 32.71,
    30.75, 28.79, 26.83, 24.87, 22.91, 20.95,
    18.99, 51.89, 50.28, 48.68, 47.07, 45.47,
    43.86, 42.26, 40.65, 39.05, 37.44, 35.84,
    34.23, 32.63, 31.02, 29.42, 27.81, 26.21,
    24.6, 60.22, 58.61, 57.01, 55.4, 53.8,
    52.19, 50.59, 48.98, 47.38, 45.77, 44.17,
    42.56, 40.96, 39.35, 37.75, 36.14, 34.54,
    32.93, 77.25, 76.0, 74.75, 73.49, 72.24,
    70.99, 69.74, 68.49, 67.23, 65.98, 64.73,
    63.48, 62.23, 60.97, 59.72, 58.47, 57.22,
    55.97, 54.71, 53.46, 52.21, 50.96, 49.71,
    48.45, 47.2, 45.95, 44.7, 43.45, 42.19,
    40.94, 39.69, 38.44, 75.59, 74.34, 73.09,
    71.83, 70.58, 69.33, 68.08, 66.83, 65.57,
    64.32, 63.07, 61.82, 60.57, 59.31, 58.06,
    56.81, 55.56,
}};

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

template <std::size_t N>
std::optional<double> lookup_atomic_property(
    const std::array<double, N>& values,
    std::uint32_t atomic_number) {
    if (atomic_number >= values.size()) {
        return std::nullopt;
    }

    const auto value = values[atomic_number];
    if (value < 0.0) {
        return std::nullopt;
    }
    return value;
}

double sphere_volume(double radius) {
    return 4.0 / 3.0 * kPi * radius * radius * radius;
}

std::optional<double> mordred_mass(std::uint32_t atomic_number) {
    return lookup_atomic_property(kMordredMassValues, atomic_number);
}

std::optional<double> rdkit_atomic_weight(std::uint32_t atomic_number) {
    return lookup_atomic_property(kRDKitAtomicWeights, atomic_number);
}

std::optional<double> mordred_vdw_volume(std::uint32_t atomic_number) {
    const auto radius = lookup_atomic_property(kMordredVdwRadii, atomic_number);
    if (!radius.has_value()) {
        return std::nullopt;
    }
    return sphere_volume(*radius);
}

std::optional<double> mordred_sanderson(std::uint32_t atomic_number) {
    return lookup_atomic_property(kMordredSandersonValues, atomic_number);
}

std::optional<double> mordred_pauling(std::uint32_t atomic_number) {
    return lookup_atomic_property(kMordredPaulingValues, atomic_number);
}

std::optional<double> mordred_allred_rocow(std::uint32_t atomic_number) {
    return lookup_atomic_property(kMordredAllredRocowValues, atomic_number);
}

std::optional<double> mordred_polarizability94(std::uint32_t atomic_number) {
    return lookup_atomic_property(kMordredPolarizability94Values, atomic_number);
}

std::optional<double> mordred_ionization_potential(std::uint32_t atomic_number) {
    return lookup_atomic_property(kMordredIonizationPotentialValues, atomic_number);
}

std::optional<double> mordred_mc_gowan_volume(std::uint32_t atomic_number) {
    return lookup_atomic_property(kMordredMcGowanVolumeValues, atomic_number);
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

double rdkit_bondi_radius(std::uint32_t atomic_number) {
    static constexpr std::array<double, 119> radii{{
        0.0,  0.33, 0.7,  1.23, 0.9,  0.82, 0.77, 0.7,  0.66, 0.611,
        0.7,  1.54, 1.36, 1.18, 0.937, 0.89, 1.04, 0.997, 1.74, 2.03,
        1.74, 1.44, 1.32, 1.22, 1.18, 1.17, 1.17, 1.16, 1.15, 1.17,
        1.25, 1.26, 1.188, 1.2,  1.17, 1.167, 1.91, 2.16, 1.91, 1.62,
        1.45, 1.34, 1.3,  1.27, 1.25, 1.25, 1.28, 1.34, 1.48, 1.44,
        1.385, 1.4,  1.378, 1.387, 1.98, 2.35, 1.98, 1.69, 1.83, 1.82,
        1.81, 1.8,  1.8,  1.99, 1.79, 1.76, 1.75, 1.74, 1.73, 1.72,
        1.94, 1.72, 1.44, 1.34, 1.3,  1.28, 1.26, 1.27, 1.3,  1.34,
        1.49, 1.48, 1.48, 1.45, 1.46, 1.45, 2.4,  2.0,  1.9,  1.88,
        1.79, 1.61, 1.58, 1.55, 1.53, 1.07, 0.0,  0.0,  0.0,  0.0,
        0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
        0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
    }};
    if (atomic_number >= radii.size()) {
        return 0.0;
    }
    return radii[atomic_number];
}

std::optional<MordredLabuteAsaValues> compute_labute_asa_values(const OEChem::OEMolBase& mol) {
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    OEChem::OESuppressHydrogens(working_mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);

    std::unordered_map<unsigned int, std::size_t> atom_indices;
    std::vector<unsigned int> atom_ids;
    std::vector<double> radii;
    radii.reserve(working_mol.NumAtoms());
    atom_ids.reserve(working_mol.NumAtoms());
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = working_mol.GetAtoms(); atom; ++atom) {
        atom_indices.emplace(atom->GetIdx(), radii.size());
        atom_ids.push_back(atom->GetIdx());
        radii.push_back(rdkit_bondi_radius(static_cast<std::uint32_t>(atom->GetAtomicNum())));
    }

    std::vector<double> atom_contribs(radii.size(), 0.0);
    for (OESystem::OEIter<OEChem::OEBondBase> bond = working_mol.GetBonds(); bond; ++bond) {
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

        const auto first_index = begin_index->second;
        const auto second_index = end_index->second;
        const auto first_radius = radii[first_index];
        const auto second_radius = radii[second_index];
        double bond_distance = first_radius + second_radius;
        if (bond->IsAromatic()) {
            bond_distance -= 0.1;
        } else {
            static constexpr std::array<double, 4> bond_scale_factors{{0.1, 0.0, 0.2, 0.3}};
            const auto order = bond->GetOrder();
            if (order < bond_scale_factors.size()) {
                bond_distance -= bond_scale_factors[order];
            }
        }

        const auto distance = std::min(
            std::max(std::fabs(first_radius - second_radius), bond_distance),
            first_radius + second_radius);
        if (distance <= 0.0 || !std::isfinite(distance)) {
            return std::nullopt;
        }
        atom_contribs[first_index] += second_radius * second_radius
                                      - (first_radius - distance)
                                            * (first_radius - distance) / distance;
        atom_contribs[second_index] += first_radius * first_radius
                                       - (second_radius - distance)
                                             * (second_radius - distance) / distance;
    }

    double hydrogen_contrib = 0.0;
    const auto hydrogen_radius = rdkit_bondi_radius(1u);
    for (std::size_t index = 0u; index < radii.size(); ++index) {
        const auto radius = radii[index];
        const auto distance = radius + hydrogen_radius;
        if (distance <= 0.0 || !std::isfinite(distance)) {
            return std::nullopt;
        }
        atom_contribs[index] += hydrogen_radius * hydrogen_radius
                                - (radius - distance) * (radius - distance) / distance;
        hydrogen_contrib += radius * radius
                            - (hydrogen_radius - distance)
                                  * (hydrogen_radius - distance) / distance;
    }

    double total = 0.0;
    for (std::size_t index = 0u; index < radii.size(); ++index) {
        const auto radius = radii[index];
        atom_contribs[index] = kPi * radius * (4.0 * radius - atom_contribs[index]);
        total += atom_contribs[index];
    }
    // RDKit adds one hydrogen shielding term per atom after Mordred has removed
    // explicit hydrogen atoms, which is why explicit-H inputs must be suppressed first.
    if (std::fabs(hydrogen_contrib) > 1.0e-4) {
        hydrogen_contrib =
            kPi * hydrogen_radius * (4.0 * hydrogen_radius - hydrogen_contrib);
        total += hydrogen_contrib;
    }
    if (!std::isfinite(total)) {
        return std::nullopt;
    }
    for (const auto atom_contribution : atom_contribs) {
        if (!std::isfinite(atom_contribution)) {
            return std::nullopt;
        }
    }
    return MordredLabuteAsaValues{total, std::move(atom_contribs), std::move(atom_ids)};
}

MordredLabuteAsaValues compute_peoe_labute_asa_values(const OEChem::OEMolBase& mol) {
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    OEChem::OESuppressHydrogens(working_mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);

    std::unordered_map<unsigned int, std::size_t> atom_indices;
    std::vector<unsigned int> atom_ids;
    std::vector<double> radii;
    radii.reserve(working_mol.NumAtoms());
    atom_ids.reserve(working_mol.NumAtoms());
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = working_mol.GetAtoms(); atom; ++atom) {
        atom_indices.emplace(atom->GetIdx(), radii.size());
        atom_ids.push_back(atom->GetIdx());
        radii.push_back(rdkit_bondi_radius(static_cast<std::uint32_t>(atom->GetAtomicNum())));
    }

    constexpr auto quiet_nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> atom_contribs(radii.size(), 0.0);
    for (OESystem::OEIter<OEChem::OEBondBase> bond = working_mol.GetBonds(); bond; ++bond) {
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

        const auto first_index = begin_index->second;
        const auto second_index = end_index->second;
        const auto first_radius = radii[first_index];
        const auto second_radius = radii[second_index];
        double bond_distance = first_radius + second_radius;
        if (bond->IsAromatic()) {
            bond_distance -= 0.1;
        } else {
            static constexpr std::array<double, 4> bond_scale_factors{{0.1, 0.0, 0.2, 0.3}};
            const auto order = bond->GetOrder();
            if (order < bond_scale_factors.size()) {
                bond_distance -= bond_scale_factors[order];
            }
        }

        const auto distance = std::min(
            std::max(std::fabs(first_radius - second_radius), bond_distance),
            first_radius + second_radius);
        if (distance <= 0.0 || !std::isfinite(distance)) {
            atom_contribs[first_index] = quiet_nan;
            atom_contribs[second_index] = quiet_nan;
            continue;
        }
        atom_contribs[first_index] += second_radius * second_radius
                                      - (first_radius - distance)
                                            * (first_radius - distance) / distance;
        atom_contribs[second_index] += first_radius * first_radius
                                       - (second_radius - distance)
                                             * (second_radius - distance) / distance;
    }

    double hydrogen_contrib = 0.0;
    const auto hydrogen_radius = rdkit_bondi_radius(1u);
    for (std::size_t index = 0u; index < radii.size(); ++index) {
        const auto radius = radii[index];
        const auto distance = radius + hydrogen_radius;
        if (distance <= 0.0 || !std::isfinite(distance)) {
            continue;
        }
        atom_contribs[index] += hydrogen_radius * hydrogen_radius
                                - (radius - distance) * (radius - distance) / distance;
        hydrogen_contrib += radius * radius
                            - (hydrogen_radius - distance)
                                  * (hydrogen_radius - distance) / distance;
    }

    double total = 0.0;
    for (std::size_t index = 0u; index < radii.size(); ++index) {
        const auto radius = radii[index];
        atom_contribs[index] = kPi * radius * (4.0 * radius - atom_contribs[index]);
        total += atom_contribs[index];
    }
    if (std::fabs(hydrogen_contrib) > 1.0e-4) {
        hydrogen_contrib =
            kPi * hydrogen_radius * (4.0 * hydrogen_radius - hydrogen_contrib);
        total += hydrogen_contrib;
    }
    return MordredLabuteAsaValues{total, std::move(atom_contribs), std::move(atom_ids)};
}

bool is_halogen(std::uint32_t atomic_number) {
    return atomic_number == 9u || atomic_number == 17u || atomic_number == 35u
           || atomic_number == 53u;
}

bool is_mordred_molecular_id_halogen(std::uint32_t atomic_number) {
    return atomic_number == 9u || atomic_number == 17u || atomic_number == 35u
           || atomic_number == 53u || atomic_number == 85u || atomic_number == 117u;
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

std::optional<double> atom_mol_wt_mass(const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto isotope = static_cast<std::uint32_t>(atom.GetIsotope());
    if (isotope != 0u) {
        return OEChem::OEGetIsotopicWeight(atomic_number, isotope);
    }
    return rdkit_atomic_weight(atomic_number);
}

std::uint32_t explicit_hydrogen_neighbor_count(const OEChem::OEAtomBase& atom) {
    std::uint32_t count = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> nbr = atom.GetAtoms(); nbr; ++nbr) {
        if (is_hydrogen(*nbr)) {
            ++count;
        }
    }
    return count;
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

const std::array<MordredEStatePattern, 79>& mordred_estate_patterns() {
    static const std::array<MordredEStatePattern, 79> patterns{{
        {"NsLi", "[LiD1]-*"},
        {"NssBe", "[BeD2](-*)-*"},
        {"NssssBe", "[BeD4](-*)(-*)(-*)-*"},
        {"NssBH", "[BD2H](-*)-*"},
        {"NsssB", "[BD3](-*)(-*)-*"},
        {"NssssB", "[BD4](-*)(-*)(-*)-*"},
        {"NsCH3", "[CD1H3]-*"},
        {"NdCH2", "[CD1H2]=*"},
        {"NssCH2", "[CD2H2](-*)-*"},
        {"NtCH", "[CD1H]#*"},
        {"NdsCH", "[CD2H](=*)-*"},
        {"NaaCH", "[C,c;D2H](:*):*"},
        {"NsssCH", "[CD3H](-*)(-*)-*"},
        {"NddC", "[CD2H0](=*)=*"},
        {"NtsC", "[CD2H0](#*)-*"},
        {"NdssC", "[CD3H0](=*)(-*)-*"},
        {"NaasC", "[C,c;D3H0](:*)(:*)-*"},
        {"NaaaC", "[C,c;D3H0](:*)(:*):*"},
        {"NssssC", "[CD4H0](-*)(-*)(-*)-*"},
        {"NsNH3", "[ND1H3]-*"},
        {"NsNH2", "[ND1H2]-*"},
        {"NssNH2", "[ND2H2](-*)-*"},
        {"NdNH", "[ND1H]=*"},
        {"NssNH", "[ND2H](-*)-*"},
        {"NaaNH", "[N,nD2H](:*):*"},
        {"NtN", "[ND1H0]#*"},
        {"NsssNH", "[ND3H](-*)(-*)-*"},
        {"NdsN", "[ND2H0](=*)-*"},
        {"NaaN", "[N,nD2H0](:*):*"},
        {"NsssN", "[ND3H0](-*)(-*)-*"},
        {"NddsN", "[ND3H0](~[OD1H0])(~[OD1H0])-,:*"},
        {"NaasN", "[N,nD3H0](:*)(:*)-,:*"},
        {"NssssN", "[ND4H0](-*)(-*)(-*)-*"},
        {"NsOH", "[OD1H]-*"},
        {"NdO", "[OD1H0]=*"},
        {"NssO", "[OD2H0](-*)-*"},
        {"NaaO", "[O,oD2H0](:*):*"},
        {"NsF", "[FD1]-*"},
        {"NsSiH3", "[SiD1H3]-*"},
        {"NssSiH2", "[SiD2H2](-*)-*"},
        {"NsssSiH", "[SiD3H1](-*)(-*)-*"},
        {"NssssSi", "[SiD4H0](-*)(-*)(-*)-*"},
        {"NsPH2", "[PD1H2]-*"},
        {"NssPH", "[PD2H1](-*)-*"},
        {"NsssP", "[PD3H0](-*)(-*)-*"},
        {"NdsssP", "[PD4H0](=*)(-*)(-*)-*"},
        {"NsssssP", "[PD5H0](-*)(-*)(-*)(-*)-*"},
        {"NsSH", "[SD1H1]-*"},
        {"NdS", "[SD1H0]=*"},
        {"NssS", "[SD2H0](-*)-*"},
        {"NaaS", "[S,sD2H0](:*):*"},
        {"NdssS", "[SD3H0](=*)(-*)-*"},
        {"NddssS", "[SD4H0](~[OD1H0])(~[OD1H0])(-*)-*"},
        {"NsCl", "[ClD1]-*"},
        {"NsGeH3", "[GeD1H3](-*)"},
        {"NssGeH2", "[GeD2H2](-*)-*"},
        {"NsssGeH", "[GeD3H1](-*)(-*)-*"},
        {"NssssGe", "[GeD4H0](-*)(-*)(-*)-*"},
        {"NsAsH2", "[AsD1H2]-*"},
        {"NssAsH", "[AsD2H1](-*)-*"},
        {"NsssAs", "[AsD3H0](-*)(-*)-*"},
        {"NsssdAs", "[AsD4H0](=*)(-*)(-*)-*"},
        {"NsssssAs", "[AsD5H0](-*)(-*)(-*)(-*)-*"},
        {"NsSeH", "[SeD1H1]-*"},
        {"NdSe", "[SeD1H0]=*"},
        {"NssSe", "[SeD2H0](-*)-*"},
        {"NaaSe", "[SeD2H0](:*):*"},
        {"NdssSe", "[SeD3H0](=*)(-*)-*"},
        {"NddssSe", "[SeD4H0](=*)(=*)(-*)-*"},
        {"NsBr", "[BrD1]-*"},
        {"NsSnH3", "[SnD1H3]-*"},
        {"NssSnH2", "[SnD2H2](-*)-*"},
        {"NsssSnH", "[SnD3H1](-*)(-*)-*"},
        {"NssssSn", "[SnD4H0](-*)(-*)(-*)-*"},
        {"NsI", "[ID1]-*"},
        {"NsPbH3", "[PbD1H3]-*"},
        {"NssPbH2", "[PbD2H2](-*)-*"},
        {"NsssPbH", "[PbD3H1](-*)(-*)-*"},
        {"NssssPb", "[PbD4H0](-*)(-*)(-*)-*"},
    }};
    return patterns;
}

OEChem::OEGraphMol implicit_hydrogen_estate_mol(const OEChem::OEMolBase& mol) {
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    // Mordred EState descriptors run with explicit_hydrogens=False; suppressing
    // H atoms before SMARTS matching preserves H-count queries on heavy atoms.
    OEChem::OESuppressHydrogens(working_mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    return working_mol;
}

std::unordered_set<unsigned int> mordred_estate_pattern_root_atom_indices(
    const OEChem::OEMolBase& mol,
    const char* smarts) {
    std::unordered_set<unsigned int> matched_atom_indices;
    OEChem::OESubSearch search(smarts);
    if (!search) {
        return matched_atom_indices;
    }
    for (OESystem::OEIter<OEChem::OEMatchBase> match = search.Match(mol, false); match;
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
    return matched_atom_indices;
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

std::optional<MordredCrippenAtomContributions> compute_crippen_atom_contributions(
    const OEChem::OEMolBase& mol) {
    OEChem::OEGraphMol crippen_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(crippen_mol);
    OEChem::OEAssignAromaticFlags(crippen_mol);
    OEChem::OEAssignHybridization(crippen_mol);
    OEChem::OESuppressHydrogens(crippen_mol);
    OEChem::OEFindRingAtomsAndBonds(crippen_mol);
    OEChem::OEAssignAromaticFlags(crippen_mol);
    OEChem::OEAssignHybridization(crippen_mol);

    std::unordered_map<unsigned int, std::size_t> atom_indices;
    MordredCrippenAtomContributions contributions;
    contributions.logp.reserve(crippen_mol.NumAtoms());
    contributions.mr.reserve(crippen_mol.NumAtoms());
    contributions.atom_ids.reserve(crippen_mol.NumAtoms());
    bool has_hydrogen_atom = false;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = crippen_mol.GetAtoms(); atom; ++atom) {
        atom_indices.emplace(atom->GetIdx(), contributions.atom_ids.size());
        contributions.logp.push_back(0.0);
        contributions.mr.push_back(0.0);
        contributions.atom_ids.push_back(atom->GetIdx());
        has_hydrogen_atom = has_hydrogen_atom || atom->GetAtomicNum() == 1;
    }

    std::vector<bool> assigned(contributions.atom_ids.size(), false);
    for (const auto& pattern : crippen_patterns()) {
        if (!has_hydrogen_atom && std::string(pattern.smarts).rfind("[#1]", 0u) == 0u) {
            continue;
        }
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
                const auto atom_index = atom_indices.find(atom_match->target->GetIdx());
                if (atom_index == atom_indices.end() || assigned[atom_index->second]) {
                    continue;
                }

                const auto contribution_index = atom_index->second;
                contributions.logp[contribution_index] = pattern.logp;
                contributions.mr[contribution_index] = pattern.mr;
                assigned[contribution_index] = true;
            }
        }
    }

    for (std::size_t atom_index = 0u; atom_index < assigned.size(); ++atom_index) {
        // RDKit initializes per-atom Crippen contributions to zero and leaves
        // unmatched atoms there; SMR_VSA should remain defined for finite surfaces.
        if (!std::isfinite(contributions.logp[atom_index])
            || !std::isfinite(contributions.mr[atom_index])) {
            return std::nullopt;
        }
    }
    return contributions;
}

const std::vector<MordredGasteigerParameters>& gasteiger_parameters() {
    static const std::vector<MordredGasteigerParameters> parameters{
        {"H", "*", 7.17, 6.24, -0.56},
        {"C", "sp3", 7.98, 9.18, 1.88},
        {"C", "sp2", 8.79, 9.32, 1.51},
        {"C", "sp", 10.39, 9.45, 0.73},
        {"N", "sp3", 11.54, 10.82, 1.36},
        {"N", "sp2", 12.87, 11.15, 0.85},
        {"N", "sp", 15.68, 11.7, -0.27},
        {"O", "sp3", 14.18, 12.92, 1.39},
        {"O", "sp2", 17.07, 13.79, 0.47},
        {"F", "sp3", 14.66, 13.85, 2.31},
        {"Cl", "sp3", 11.00, 9.69, 1.35},
        {"Br", "sp3", 10.08, 8.47, 1.16},
        {"I", "sp3", 9.9, 7.96, 0.96},
        {"S", "sp3", 10.14, 9.13, 1.38},
        {"S", "so", 10.14, 9.13, 1.38},
        {"S", "so2", 12.00, 10.81, 1.20},
        {"S", "sp2", 10.88, 9.49, 1.33},
        {"P", "sp3", 8.90, 8.24, 0.96},
        {"X", "*", 0.00, 0.00, 0.00},
        {"P", "sp2", 9.665, 8.530, 0.735},
        {"Si", "sp3", 7.300, 6.567, 0.657},
        {"Si", "sp2", 7.905, 6.748, 0.443},
        {"Si", "sp", 9.065, 7.027, -0.002},
        {"B", "sp3", 5.980, 6.820, 1.605},
        {"B", "sp2", 6.420, 6.807, 1.322},
        {"Be", "sp3", 3.845, 6.755, 3.165},
        {"Be", "sp2", 4.005, 6.725, 3.035},
        {"Mg", "sp2", 3.565, 5.572, 2.197},
        {"Mg", "sp3", 3.300, 5.587, 2.447},
        {"Mg", "sp", 4.040, 5.472, 1.823},
        {"Al", "sp3", 5.375, 4.953, 0.867},
        {"Al", "sp2", 5.795, 5.020, 0.695},
    };
    return parameters;
}

const char* gasteiger_element_symbol(std::uint32_t atomic_number) {
    switch (atomic_number) {
    case 1:
        return "H";
    case 4:
        return "Be";
    case 5:
        return "B";
    case 6:
        return "C";
    case 7:
        return "N";
    case 8:
        return "O";
    case 9:
        return "F";
    case 12:
        return "Mg";
    case 13:
        return "Al";
    case 14:
        return "Si";
    case 15:
        return "P";
    case 16:
        return "S";
    case 17:
        return "Cl";
    case 35:
        return "Br";
    case 53:
        return "I";
    default:
        return "X";
    }
}

std::uint32_t oxygen_neighbor_count(const OEChem::OEAtomBase& atom) {
    std::uint32_t count = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> nbr = atom.GetAtoms(); nbr; ++nbr) {
        if (nbr->GetAtomicNum() == 8u) {
            ++count;
        }
    }
    return count;
}

bool has_bond_order_at_least(const OEChem::OEAtomBase& atom, unsigned int order) {
    for (OESystem::OEIter<OEChem::OEBondBase> bond = atom.GetBonds(); bond; ++bond) {
        if (bond->GetOrder() >= order) {
            return true;
        }
    }
    return false;
}

const char* gasteiger_mode(const OEChem::OEAtomBase& atom, bool has_conjugated_bond) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    if (atomic_number == 1u) {
        return "*";
    }
    if (atomic_number != 16u && has_bond_order_at_least(atom, 3u)) {
        return "sp";
    }
    // RDKit hybridization uses its conjugation flags before Gasteiger lookup;
    // OpenEye can leave charged first-row conjugated atoms unspecified.
    if (atomic_number <= 10u
        && (atom.IsAromatic() || has_conjugated_bond || has_bond_order_at_least(atom, 2u))) {
        return "sp2";
    }

    switch (atom.GetHyb()) {
    case OEChem::OEHybridization::sp3:
        return "sp3";
    case OEChem::OEHybridization::sp2:
        return "sp2";
    case OEChem::OEHybridization::sp:
        return "sp";
    default:
        if (atomic_number == 16u) {
            const auto oxygens = oxygen_neighbor_count(atom);
            if (oxygens == 2u) {
                return "so2";
            }
            if (oxygens == 1u) {
                return "so";
            }
            return "sp3";
        }
        switch (atomic_number) {
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 12:
        case 13:
        case 14:
        case 15:
        case 17:
        case 35:
        case 53:
            return "sp3";
        default:
            break;
        }
        return "*";
    }
}

const MordredGasteigerParameters& lookup_gasteiger_parameters(
    const char* element,
    const char* mode) {
    const auto& parameters = gasteiger_parameters();
    for (const auto& parameter : parameters) {
        if (std::string(parameter.element) == element && std::string(parameter.mode) == mode) {
            return parameter;
        }
    }
    for (const auto& parameter : parameters) {
        if (std::string(parameter.element) == "X" && std::string(parameter.mode) == "*") {
            return parameter;
        }
    }
    return parameters.front();
}

std::optional<int> gasteiger_default_valence(std::uint32_t atomic_number) {
    constexpr std::array<int, 119> default_valences{{
        -1, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3,
        2, 1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 3, 4, 3, 2, 1, 0, 1, 2, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, 3, 2, 3, 2, 1, 0, 1, 2, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2,
        3, 2, 1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    }};
    if (atomic_number >= default_valences.size()) {
        return std::nullopt;
    }
    return default_valences[atomic_number];
}

std::uint32_t explicit_neighbor_count(const OEChem::OEAtomBase& atom) {
    std::uint32_t count = 0u;
    for (OESystem::OEIter<OEChem::OEAtomBase> neighbor = atom.GetAtoms(); neighbor; ++neighbor) {
        ++count;
    }
    return count;
}

int gasteiger_count_atom_electrons(const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto default_valence = gasteiger_default_valence(atomic_number);
    if (!default_valence.has_value() || *default_valence <= 1) {
        return -1;
    }

    const auto degree = static_cast<int>(atom.GetDegree());
    if (degree > 3) {
        return -1;
    }

    const auto outer_electrons = mordred_outer_electrons(atomic_number);
    if (!outer_electrons.has_value()) {
        return -1;
    }

    auto lone_pair_electrons = *outer_electrons - *default_valence - atom.GetFormalCharge();
    lone_pair_electrons = std::max(lone_pair_electrons, 0);
    auto electrons = (*default_valence - degree) + lone_pair_electrons;
    if (electrons > 1) {
        const auto unsaturations =
            static_cast<int>(atom.GetExplicitValence())
            - static_cast<int>(explicit_neighbor_count(atom));
        if (unsaturations > 1) {
            electrons = 1;
        }
    }
    return electrons;
}

bool is_gasteiger_conjugation_candidate(const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto default_valence = gasteiger_default_valence(atomic_number);
    if (atom.GetFormalCharge() == 0 && default_valence.has_value() && *default_valence >= 0
        && static_cast<int>(atom.GetValence()) > *default_valence) {
        return false;
    }

    const auto outer_electrons = mordred_outer_electrons(atomic_number);
    if (!outer_electrons.has_value()) {
        return false;
    }

    return (atomic_number <= 10u || (*outer_electrons != 5 && *outer_electrons != 6)
            || (*outer_electrons == 6 && atom.GetDegree() < 2u))
           && gasteiger_count_atom_electrons(atom) > 0;
}

double gasteiger_bond_valence_contribution(const OEChem::OEBondBase& bond) {
    return bond.IsAromatic() ? 1.5 : static_cast<double>(bond.GetOrder());
}

std::pair<std::size_t, std::size_t> gasteiger_bond_key(
    std::size_t first,
    std::size_t second) {
    return first < second ? std::make_pair(first, second) : std::make_pair(second, first);
}

std::set<std::pair<std::size_t, std::size_t>> compute_gasteiger_conjugated_bond_keys(
    const std::vector<OEChem::OEAtomBase*>& atoms,
    const std::vector<std::vector<std::pair<std::size_t, const OEChem::OEBondBase*>>>&
        atom_bonds) {
    // RDKit's Gasteiger formal-charge split consumes MolOps::setConjugation().
    // OpenEye does not expose that flag here, so mirror the RDKit marking pass.
    std::set<std::pair<std::size_t, std::size_t>> conjugated_bonds;
    for (std::size_t atom_index = 0u; atom_index < atoms.size(); ++atom_index) {
        for (const auto& neighbor : atom_bonds[atom_index]) {
            if (neighbor.second != nullptr && neighbor.second->IsAromatic()) {
                conjugated_bonds.insert(gasteiger_bond_key(atom_index, neighbor.first));
            }
        }
    }

    for (std::size_t atom_index = 0u; atom_index < atoms.size(); ++atom_index) {
        const auto* atom = atoms[atom_index];
        if (atom == nullptr || !is_gasteiger_conjugation_candidate(*atom)) {
            continue;
        }

        const auto substitution_count = atom->GetDegree();
        if (substitution_count < 2u || substitution_count > 3u) {
            continue;
        }

        for (const auto& first_bond : atom_bonds[atom_index]) {
            if (first_bond.second == nullptr
                || gasteiger_bond_valence_contribution(*first_bond.second) < 1.5
                || atoms[first_bond.first] == nullptr
                || !is_gasteiger_conjugation_candidate(*atoms[first_bond.first])) {
                continue;
            }

            for (const auto& second_bond : atom_bonds[atom_index]) {
                if (second_bond.second == nullptr || second_bond.second == first_bond.second
                    || atoms[second_bond.first] == nullptr
                    || atoms[second_bond.first]->GetDegree() > 3u
                    || !is_gasteiger_conjugation_candidate(*atoms[second_bond.first])) {
                    continue;
                }

                conjugated_bonds.insert(gasteiger_bond_key(atom_index, first_bond.first));
                conjugated_bonds.insert(gasteiger_bond_key(atom_index, second_bond.first));
            }
        }
    }
    return conjugated_bonds;
}

bool atom_has_gasteiger_conjugated_bond(
    std::size_t atom_index,
    const std::vector<std::vector<std::pair<std::size_t, const OEChem::OEBondBase*>>>&
        atom_bonds,
    const std::set<std::pair<std::size_t, std::size_t>>& conjugated_bonds) {
    for (const auto& neighbor : atom_bonds[atom_index]) {
        if (conjugated_bonds.count(gasteiger_bond_key(atom_index, neighbor.first)) != 0u) {
            return true;
        }
    }
    return false;
}

void split_gasteiger_formal_charges(
    const std::vector<OEChem::OEAtomBase*>& atoms,
    const std::vector<std::vector<std::pair<std::size_t, const OEChem::OEBondBase*>>>&
        atom_bonds,
    const std::set<std::pair<std::size_t, std::size_t>>& conjugated_bonds,
    std::vector<double>& charges) {
    static constexpr double eps = 1.0e-12;
    for (std::size_t atom_index = 0u; atom_index < atoms.size(); ++atom_index) {
        const auto* atom = atoms[atom_index];
        if (atom == nullptr) {
            continue;
        }

        double formal_charge = static_cast<double>(atom->GetFormalCharge());
        std::vector<std::size_t> markers;
        if (std::fabs(formal_charge) <= eps || std::fabs(charges[atom_index]) >= eps) {
            continue;
        }

        markers.push_back(atom_index);
        for (const auto& first_bond : atom_bonds[atom_index]) {
            if (first_bond.second == nullptr
                || conjugated_bonds.count(gasteiger_bond_key(atom_index, first_bond.first)) == 0u) {
                continue;
            }

            const auto center_index = first_bond.first;
            for (const auto& second_bond : atom_bonds[center_index]) {
                if (second_bond.second == nullptr || second_bond.second == first_bond.second
                    || conjugated_bonds.count(gasteiger_bond_key(center_index, second_bond.first))
                           == 0u) {
                    continue;
                }

                const auto terminal_index = second_bond.first;
                if (atoms[terminal_index] != nullptr
                    && atoms[terminal_index]->GetAtomicNum() == atom->GetAtomicNum()) {
                    formal_charge += static_cast<double>(atoms[terminal_index]->GetFormalCharge());
                    markers.push_back(terminal_index);
                }
            }
        }

        const auto split_charge = formal_charge / static_cast<double>(markers.size());
        for (const auto marker : markers) {
            charges[marker] = split_charge;
        }
    }
}

MordredGasteigerAtomCharges compute_gasteiger_atom_charges(
    const OEChem::OEMolBase& mol,
    bool suppress_hydrogens) {
    static constexpr double ionxh = 20.02;
    static constexpr double damp_scale = 0.5;
    static constexpr double initial_damp = 0.5;
    static constexpr unsigned int iterations = 12u;

    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    OEChem::OEAssignHybridization(working_mol);
    if (suppress_hydrogens) {
        OEChem::OESuppressHydrogens(working_mol);
        OEChem::OEFindRingAtomsAndBonds(working_mol);
        OEChem::OEAssignAromaticFlags(working_mol);
        OEChem::OEAssignHybridization(working_mol);
    }

    std::unordered_map<unsigned int, std::size_t> atom_indices;
    std::vector<OEChem::OEAtomBase*> atoms;
    std::vector<unsigned int> atom_ids;
    atoms.reserve(working_mol.NumAtoms());
    atom_ids.reserve(working_mol.NumAtoms());
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = working_mol.GetAtoms(); atom; ++atom) {
        atom_indices.emplace(atom->GetIdx(), atoms.size());
        atoms.push_back(&*atom);
        atom_ids.push_back(atom->GetIdx());
    }

    std::vector<std::vector<std::pair<std::size_t, const OEChem::OEBondBase*>>> atom_bonds(
        atoms.size());
    for (OESystem::OEIter<OEChem::OEBondBase> bond = working_mol.GetBonds(); bond; ++bond) {
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
        atom_bonds[begin_index->second].emplace_back(end_index->second, &*bond);
        atom_bonds[end_index->second].emplace_back(begin_index->second, &*bond);
    }

    std::vector<MordredGasteigerParameters> atom_parameters;
    atom_parameters.reserve(atoms.size());
    std::vector<double> charges(atoms.size(), 0.0);
    std::vector<double> hydrogen_charges(atoms.size(), 0.0);
    std::vector<double> ion_x(atoms.size(), 0.0);
    std::vector<double> energies(atoms.size(), 0.0);

    const auto conjugated_bonds = compute_gasteiger_conjugated_bond_keys(atoms, atom_bonds);
    split_gasteiger_formal_charges(atoms, atom_bonds, conjugated_bonds, charges);

    for (std::size_t atom_index = 0u; atom_index < atoms.size(); ++atom_index) {
        const auto has_conjugated_bond =
            atom_has_gasteiger_conjugated_bond(atom_index, atom_bonds, conjugated_bonds);
        const auto& parameters = lookup_gasteiger_parameters(
            gasteiger_element_symbol(static_cast<std::uint32_t>(atoms[atom_index]->GetAtomicNum())),
            gasteiger_mode(*atoms[atom_index], has_conjugated_bond));
        atom_parameters.push_back(parameters);
        ion_x[atom_index] =
            atoms[atom_index]->GetAtomicNum() == 1u
                ? ionxh
                : parameters.a + parameters.b + parameters.c;
    }

    const auto& hydrogen_parameters = lookup_gasteiger_parameters("H", "*");
    double damp = initial_damp;
    for (unsigned int iteration = 0u; iteration < iterations; ++iteration) {
        for (std::size_t atom_index = 0u; atom_index < atoms.size(); ++atom_index) {
            const auto& parameters = atom_parameters[atom_index];
            energies[atom_index] = parameters.a
                                   + charges[atom_index]
                                         * (parameters.b
                                            + parameters.c * charges[atom_index]);
        }

        for (std::size_t atom_index = 0u; atom_index < atoms.size(); ++atom_index) {
            double delta_charge = 0.0;
            for (const auto& neighbor : atom_bonds[atom_index]) {
                const auto neighbor_index = neighbor.first;
                const auto dx = energies[neighbor_index] - energies[atom_index];
                const auto sign = dx < 0.0 ? 0 : 1;
                delta_charge +=
                    dx / (static_cast<double>(sign) * (ion_x[atom_index] - ion_x[neighbor_index])
                          + ion_x[neighbor_index]);
            }

            const auto implicit_hydrogens =
                suppress_hydrogens ? atoms[atom_index]->GetTotalHCount() : 0u;
            if (implicit_hydrogens > 0u) {
                const auto hydrogen_charge =
                    hydrogen_charges[atom_index] / static_cast<double>(implicit_hydrogens);
                const auto hydrogen_energy =
                    hydrogen_parameters.a
                    + hydrogen_charge
                          * (hydrogen_parameters.b
                             + hydrogen_parameters.c * hydrogen_charge);
                const auto dx = hydrogen_energy - energies[atom_index];
                const auto sign = dx < 0.0 ? 0 : 1;
                const auto hydrogen_delta =
                    dx / (static_cast<double>(sign) * (ion_x[atom_index] - ionxh) + ionxh);
                delta_charge += static_cast<double>(implicit_hydrogens) * hydrogen_delta;
                hydrogen_charges[atom_index] -=
                    static_cast<double>(implicit_hydrogens) * hydrogen_delta * damp;
            }

            charges[atom_index] += damp * delta_charge;
        }
        damp *= damp_scale;
    }

    return MordredGasteigerAtomCharges{
        std::move(charges),
        std::move(hydrogen_charges),
        std::move(atom_ids)};
}

MordredGasteigerAtomCharges compute_gasteiger_atom_charges(const OEChem::OEMolBase& mol) {
    return compute_gasteiger_atom_charges(mol, true);
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

std::optional<double> compute_mordred_mol_wt(const OEChem::OEMolBase& mol) {
    const auto hydrogen_mass = rdkit_atomic_weight(1u);
    if (!hydrogen_mass.has_value()) {
        return std::nullopt;
    }

    double weight = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atom_mass = atom_mol_wt_mass(*atom);
        if (!atom_mass.has_value()) {
            return std::nullopt;
        }
        if (is_hydrogen(*atom)) {
            weight += *atom_mass;
            continue;
        }
        const auto total_hydrogens = static_cast<std::uint32_t>(atom->GetTotalHCount());
        const auto explicit_hydrogens = explicit_hydrogen_neighbor_count(*atom);
        const auto implicit_hydrogens =
            explicit_hydrogens > total_hydrogens ? 0u : total_hydrogens - explicit_hydrogens;
        weight += *atom_mass + static_cast<double>(implicit_hydrogens) * *hydrogen_mass;
    }
    return weight;
}

const std::array<MordredLogSPattern, 16>& mordred_log_s_patterns() {
    static constexpr std::array<MordredLogSPattern, 16> patterns{{
        {"[NH0;X3;v3]", 0.71535},
        {"[NH2;X3;v3]", 0.41056},
        {"[nH0;X3]", 0.82535},
        {"[OH0;X2;v2]", 0.31464},
        {"[OH0;X1;v2]", 0.14787},
        {"[OH1;X2;v2]", 0.62998},
        {"[CH2;!R]", -0.35634},
        {"[CH3;!R]", -0.33888},
        {"[CH0;R]", -0.21912},
        {"[CH2;R]", -0.23057},
        {"[ch0]", -0.37570},
        {"[ch1]", -0.22435},
        {"F", -0.21728},
        {"Cl", -0.49721},
        {"Br", -0.57982},
        {"I", -0.51547},
    }};
    return patterns;
}

std::uint32_t count_smarts_matches(const OEChem::OEMolBase& mol, const char* smarts) {
    OEChem::OESubSearch search(smarts);
    if (!search) {
        return 0u;
    }

    std::uint32_t count = 0u;
    for (OESystem::OEIter<OEChem::OEMatchBase> match = search.Match(mol, false); match;
         ++match) {
        ++count;
    }
    return count;
}

std::optional<double> compute_filter_it_log_s(const OEChem::OEMolBase& mol) {
    const auto molecular_weight = compute_mordred_mol_wt(mol);
    if (!molecular_weight.has_value()) {
        return std::nullopt;
    }

    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    OEChem::OEAssignHybridization(working_mol);
    // LogS runs with explicit_hydrogens=False while its Weight dependency does not.
    OEChem::OESuppressHydrogens(working_mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    OEChem::OEAssignHybridization(working_mol);

    double log_s = 0.89823 - 0.10369 * std::sqrt(*molecular_weight);
    for (const auto& pattern : mordred_log_s_patterns()) {
        log_s += static_cast<double>(count_smarts_matches(working_mol, pattern.smarts))
                 * pattern.coefficient;
    }
    return log_s;
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

double compute_fragment_complexity(const MordredFirstBatchValues& values) {
    const auto atom_count = static_cast<std::int64_t>(values.heavy_atoms);
    const auto bond_count = static_cast<std::int64_t>(values.heavy_bonds);
    const auto topology_term = bond_count * bond_count - atom_count * atom_count + atom_count;
    const auto absolute_topology_term =
        topology_term < 0 ? -topology_term : topology_term;
    return static_cast<double>(absolute_topology_term)
           + static_cast<double>(values.hetero_atoms) / 100.0;
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

bool has_graph_edge(
    const std::vector<std::unordered_set<std::size_t>>& adjacency_sets,
    std::size_t left,
    std::size_t right) {
    return adjacency_sets[left].find(right) != adjacency_sets[left].end();
}

bool is_chordless_cycle(
    const std::vector<std::size_t>& cycle,
    const std::vector<std::unordered_set<std::size_t>>& adjacency_sets) {
    for (std::size_t i = 0u; i < cycle.size(); ++i) {
        for (std::size_t j = i + 2u; j < cycle.size(); ++j) {
            if (i == 0u && j + 1u == cycle.size()) {
                continue;
            }
            if (has_graph_edge(adjacency_sets, cycle[i], cycle[j])) {
                return false;
            }
        }
    }
    return true;
}

std::vector<std::size_t> canonical_cycle(std::vector<std::size_t> cycle) {
    const auto smallest = std::min_element(cycle.begin(), cycle.end());
    std::rotate(cycle.begin(), smallest, cycle.end());

    std::vector<std::size_t> reversed{cycle.front()};
    reversed.insert(reversed.end(), cycle.rbegin(), cycle.rend() - 1);
    return reversed < cycle ? reversed : cycle;
}

std::uint64_t edge_key(std::size_t left, std::size_t right) {
    const auto low = std::min(left, right);
    const auto high = std::max(left, right);
    return (static_cast<std::uint64_t>(low) << 32u) | static_cast<std::uint64_t>(high);
}

std::vector<std::uint64_t> cycle_edge_vector(
    const std::vector<std::size_t>& cycle,
    const std::unordered_map<std::uint64_t, std::size_t>& edge_indices,
    std::size_t word_count) {
    std::vector<std::uint64_t> bits(word_count, 0u);
    for (std::size_t i = 0u; i < cycle.size(); ++i) {
        const auto next = (i + 1u) % cycle.size();
        const auto edge = edge_indices.at(edge_key(cycle[i], cycle[next]));
        bits[edge / 64u] |= std::uint64_t{1u} << (edge % 64u);
    }
    return bits;
}

std::vector<std::size_t> cycle_edge_indices(
    const std::vector<std::size_t>& cycle,
    const std::unordered_map<std::uint64_t, std::size_t>& edge_indices) {
    std::vector<std::size_t> edges;
    edges.reserve(cycle.size());
    for (std::size_t i = 0u; i < cycle.size(); ++i) {
        const auto next = (i + 1u) % cycle.size();
        edges.push_back(edge_indices.at(edge_key(cycle[i], cycle[next])));
    }
    return edges;
}

bool bit_vector_is_subset(
    const std::vector<std::uint64_t>& left,
    const std::vector<std::uint64_t>& right) {
    for (std::size_t index = 0u; index < left.size(); ++index) {
        if ((left[index] & ~right[index]) != 0u) {
            return false;
        }
    }
    return true;
}

void bit_vector_or_into(std::vector<std::uint64_t>& target, const std::vector<std::uint64_t>& source) {
    for (std::size_t index = 0u; index < target.size(); ++index) {
        target[index] |= source[index];
    }
}

std::uint32_t bit_vector_overlap_count(
    const std::vector<std::uint64_t>& left,
    const std::vector<std::uint64_t>& right) {
    std::uint32_t count = 0u;
    for (std::size_t index = 0u; index < left.size(); ++index) {
        auto value = left[index] & right[index];
        while (value != 0u) {
            value &= value - 1u;
            ++count;
        }
    }
    return count;
}

bool contains_edge(const std::vector<std::size_t>& edges, std::size_t edge) {
    return std::find(edges.begin(), edges.end(), edge) != edges.end();
}

void enumerate_chordless_cycles(
    std::size_t start,
    std::size_t current,
    const std::vector<std::vector<std::size_t>>& adjacency,
    const std::vector<std::unordered_set<std::size_t>>& adjacency_sets,
    std::vector<std::size_t>& path,
    std::vector<bool>& in_path,
    std::set<std::vector<std::size_t>>& cycles) {
    for (const auto neighbor : adjacency[current]) {
        if (neighbor == start) {
            if (path.size() >= 3u && is_chordless_cycle(path, adjacency_sets)) {
                cycles.insert(canonical_cycle(path));
            }
            continue;
        }
        if (neighbor < start || in_path[neighbor]) {
            continue;
        }

        in_path[neighbor] = true;
        path.push_back(neighbor);
        enumerate_chordless_cycles(start, neighbor, adjacency, adjacency_sets, path, in_path, cycles);
        path.pop_back();
        in_path[neighbor] = false;
    }
}

void fast_find_ring_cycles_dfs(
    std::size_t atom,
    std::optional<std::size_t> from_atom,
    const std::vector<std::vector<std::size_t>>& adjacency,
    std::vector<std::uint8_t>& atom_colors,
    std::vector<std::size_t>& traversal_order,
    std::vector<std::vector<std::size_t>>& cycles) {
    atom_colors[atom] = 1u;
    traversal_order.push_back(atom);

    for (const auto neighbor : adjacency[atom]) {
        if (atom_colors[neighbor] == 0u) {
            if (adjacency[neighbor].size() < 2u) {
                atom_colors[neighbor] = 2u;
            } else {
                fast_find_ring_cycles_dfs(
                    neighbor, atom, adjacency, atom_colors, traversal_order, cycles);
            }
        } else if (
            atom_colors[neighbor] == 1u && from_atom.has_value() && neighbor != *from_atom) {
            std::vector<std::size_t> cycle;
            for (auto reverse = traversal_order.rbegin();
                 reverse != traversal_order.rend() && *reverse != neighbor;
                 ++reverse) {
                cycle.push_back(*reverse);
            }
            cycle.push_back(neighbor);
            cycles.push_back(std::move(cycle));
        }
    }

    atom_colors[atom] = 2u;
    traversal_order.pop_back();
}

std::vector<std::vector<std::size_t>> fast_find_ring_cycles(
    const std::vector<std::vector<std::size_t>>& adjacency) {
    std::vector<std::vector<std::size_t>> cycles;
    std::vector<std::uint8_t> atom_colors(adjacency.size(), 0u);

    for (std::size_t atom = 0u; atom < adjacency.size(); ++atom) {
        if (atom_colors[atom] != 0u) {
            continue;
        }
        if (adjacency[atom].size() < 2u) {
            atom_colors[atom] = 2u;
            continue;
        }

        std::vector<std::size_t> traversal_order;
        fast_find_ring_cycles_dfs(
            atom, std::nullopt, adjacency, atom_colors, traversal_order, cycles);
    }
    return cycles;
}

bool needs_complete_graph_fast_find_fallback(
    const std::vector<std::vector<std::size_t>>& adjacency) {
    if (adjacency.size() < 4u) {
        return false;
    }

    const auto expected_degree = adjacency.size() - 1u;
    const auto is_complete = std::all_of(
        adjacency.begin(),
        adjacency.end(),
        [expected_degree](const auto& neighbors) {
            return neighbors.size() == expected_degree;
        });
    if (!is_complete) {
        return false;
    }

    const auto edge_count = adjacency.size() * expected_degree / 2u;
    const auto cycle_rank = edge_count - adjacency.size() + 1u;
    return cycle_rank > adjacency.size();
}

std::vector<std::size_t> rdkit_like_symmetrized_ring_indices(
    const std::vector<std::vector<std::size_t>>& sorted_cycles,
    const std::unordered_map<std::uint64_t, std::size_t>& edge_indices,
    std::size_t word_count) {
    std::vector<std::vector<std::uint64_t>> bit_cycles;
    std::vector<std::vector<std::size_t>> edge_cycles;
    bit_cycles.reserve(sorted_cycles.size());
    edge_cycles.reserve(sorted_cycles.size());
    for (const auto& cycle : sorted_cycles) {
        bit_cycles.push_back(cycle_edge_vector(cycle, edge_indices, word_count));
        edge_cycles.push_back(cycle_edge_indices(cycle, edge_indices));
    }

    std::vector<bool> available(sorted_cycles.size(), true);
    std::vector<bool> keep(sorted_cycles.size(), false);
    std::vector<std::uint64_t> bond_union(word_count, 0u);

    for (std::size_t i = 0u; i < sorted_cycles.size(); ++i) {
        if (bit_vector_is_subset(bit_cycles[i], bond_union)) {
            available[i] = false;
        }
        if (!available[i]) {
            continue;
        }

        bit_vector_or_into(bond_union, bit_cycles[i]);
        keep[i] = true;

        std::vector<bool> consider(sorted_cycles.size(), false);
        for (std::size_t j = i + 1u; j < sorted_cycles.size(); ++j) {
            if (available[j] && sorted_cycles[j].size() == sorted_cycles[i].size()) {
                consider[j] = true;
            }
        }

        while (std::any_of(consider.begin(), consider.end(), [](bool value) { return value; })) {
            std::size_t best_index = i + 1u;
            std::uint32_t best_overlap = 0u;
            bool found = false;
            for (std::size_t j = i + 1u; j < sorted_cycles.size(); ++j) {
                if (!consider[j] || !available[j]) {
                    continue;
                }
                const auto overlap = bit_vector_overlap_count(bit_cycles[j], bond_union);
                if (!found || overlap > best_overlap) {
                    found = true;
                    best_overlap = overlap;
                    best_index = j;
                }
            }
            if (!found) {
                break;
            }

            consider[best_index] = false;
            if (bit_vector_is_subset(bit_cycles[best_index], bond_union)) {
                available[best_index] = false;
            } else {
                keep[best_index] = true;
                available[best_index] = false;
                bit_vector_or_into(bond_union, bit_cycles[best_index]);
            }
        }
    }

    std::vector<int> bond_counts(edge_indices.size(), 0);
    for (std::size_t ring_index = 0u; ring_index < sorted_cycles.size(); ++ring_index) {
        if (!keep[ring_index]) {
            continue;
        }
        for (const auto edge : edge_cycles[ring_index]) {
            ++bond_counts[edge];
        }
    }

    std::vector<std::size_t> selected_indices;
    for (std::size_t ring_index = 0u; ring_index < sorted_cycles.size(); ++ring_index) {
        if (keep[ring_index]) {
            selected_indices.push_back(ring_index);
        }
    }

    for (std::size_t extra_index = 0u; extra_index < sorted_cycles.size(); ++extra_index) {
        if (keep[extra_index]) {
            continue;
        }
        for (std::size_t ring_index = 0u; ring_index < sorted_cycles.size(); ++ring_index) {
            if (!keep[ring_index]
                || sorted_cycles[ring_index].size() != sorted_cycles[extra_index].size()) {
                continue;
            }

            bool shares_bond = false;
            bool replaces_unique_bonds = true;
            for (const auto edge : edge_cycles[ring_index]) {
                if (bond_counts[edge] == 1 || !shares_bond) {
                    if (contains_edge(edge_cycles[extra_index], edge)) {
                        shares_bond = true;
                    } else if (bond_counts[edge] == 1) {
                        replaces_unique_bonds = false;
                    }
                }
            }

            if (shares_bond && replaces_unique_bonds) {
                selected_indices.push_back(extra_index);
                break;
            }
        }
    }

    return selected_indices;
}

std::unordered_map<std::uint64_t, std::size_t> edge_indices_for_adjacency(
    const std::vector<std::vector<std::size_t>>& adjacency) {
    std::unordered_map<std::uint64_t, std::size_t> edge_indices;
    for (std::size_t left = 0u; left < adjacency.size(); ++left) {
        for (const auto right : adjacency[left]) {
            if (left < right) {
                edge_indices.emplace(edge_key(left, right), edge_indices.size());
            }
        }
    }
    return edge_indices;
}

void add_ring_count(MordredRingCountSummary& summary, std::size_t size) {
    ++summary.total;
    if (size < summary.by_size.size()) {
        ++summary.by_size[size];
    }
    if (size >= 12u) {
        ++summary.greater_or_equal_12;
    }
}

void add_ring_count_values(
    MordredRingCountValues& values,
    const std::vector<std::size_t>& ring,
    const std::vector<MordredRingAtomProperties>& atom_properties) {
    const auto has_hetero = std::any_of(
        ring.begin(),
        ring.end(),
        [&atom_properties](std::size_t atom) {
            return atom_properties[atom].atomic_number != 6u;
        });
    const auto all_aromatic = std::all_of(
        ring.begin(),
        ring.end(),
        [&atom_properties](std::size_t atom) {
            return atom_properties[atom].aromatic;
        });

    add_ring_count(values.all, ring.size());
    if (has_hetero) {
        add_ring_count(values.hetero, ring.size());
    }
    if (all_aromatic) {
        add_ring_count(values.aromatic, ring.size());
        if (has_hetero) {
            add_ring_count(values.aromatic_hetero, ring.size());
        }
    } else {
        add_ring_count(values.aliphatic, ring.size());
        if (has_hetero) {
            add_ring_count(values.aliphatic_hetero, ring.size());
        }
    }
}

MordredRingCountValues count_ring_values(
    const std::vector<std::vector<std::size_t>>& cycles,
    const std::vector<MordredRingAtomProperties>& atom_properties) {
    MordredRingCountValues values;
    for (const auto& cycle : cycles) {
        add_ring_count_values(values, cycle, atom_properties);
    }
    return values;
}

std::vector<std::vector<std::size_t>> compute_component_base_ring_cycles(
    const std::vector<std::vector<std::size_t>>& adjacency) {
    if (needs_complete_graph_fast_find_fallback(adjacency)) {
        // RDKit falls back to FastFindRings for some highly connected graph
        // components where the SSSR search cannot find the expected cycle rank.
        return fast_find_ring_cycles(adjacency);
    }

    std::vector<std::unordered_set<std::size_t>> adjacency_sets(adjacency.size());
    for (std::size_t atom = 0u; atom < adjacency.size(); ++atom) {
        for (const auto neighbor : adjacency[atom]) {
            adjacency_sets[atom].insert(neighbor);
        }
    }

    std::set<std::vector<std::size_t>> cycles;
    for (std::size_t start = 0u; start < adjacency.size(); ++start) {
        std::vector<std::size_t> path{start};
        std::vector<bool> in_path(adjacency.size(), false);
        in_path[start] = true;
        enumerate_chordless_cycles(start, start, adjacency, adjacency_sets, path, in_path, cycles);
    }
    std::vector<std::vector<std::size_t>> sorted_cycles(cycles.begin(), cycles.end());
    std::sort(
        sorted_cycles.begin(),
        sorted_cycles.end(),
        [](const auto& left, const auto& right) {
            if (left.size() != right.size()) {
                return left.size() < right.size();
            }
            return left < right;
        });

    const auto edge_indices = edge_indices_for_adjacency(adjacency);
    const auto word_count = (edge_indices.size() + 63u) / 64u;
    const auto selected_indices =
        rdkit_like_symmetrized_ring_indices(sorted_cycles, edge_indices, word_count);

    std::vector<std::vector<std::size_t>> selected_cycles;
    selected_cycles.reserve(selected_indices.size());
    for (const auto ring_index : selected_indices) {
        selected_cycles.push_back(sorted_cycles[ring_index]);
    }
    return selected_cycles;
}

std::size_t shared_atom_count(
    const std::vector<std::size_t>& left,
    const std::vector<std::size_t>& right) {
    std::size_t count = 0u;
    for (const auto atom : left) {
        if (std::find(right.begin(), right.end(), atom) != right.end()) {
            ++count;
        }
    }
    return count;
}

std::vector<std::vector<std::size_t>> fused_ring_systems(
    const std::vector<std::vector<std::size_t>>& rings) {
    if (rings.size() < 2u) {
        return {};
    }

    std::vector<std::vector<std::size_t>> ring_adjacency(rings.size());
    for (std::size_t left = 0u; left < rings.size(); ++left) {
        for (std::size_t right = left + 1u; right < rings.size(); ++right) {
            if (shared_atom_count(rings[left], rings[right]) >= 2u) {
                ring_adjacency[left].push_back(right);
                ring_adjacency[right].push_back(left);
            }
        }
    }

    std::vector<std::vector<std::size_t>> systems;
    std::vector<bool> visited(rings.size(), false);
    for (std::size_t start = 0u; start < rings.size(); ++start) {
        if (visited[start] || ring_adjacency[start].empty()) {
            continue;
        }

        std::set<std::size_t> fused_atoms;
        std::vector<std::size_t> stack{start};
        visited[start] = true;
        while (!stack.empty()) {
            const auto ring_index = stack.back();
            stack.pop_back();
            fused_atoms.insert(rings[ring_index].begin(), rings[ring_index].end());

            for (const auto neighbor : ring_adjacency[ring_index]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    stack.push_back(neighbor);
                }
            }
        }

        systems.emplace_back(fused_atoms.begin(), fused_atoms.end());
    }
    return systems;
}

std::vector<std::vector<std::size_t>> connected_components(
    const std::vector<std::vector<std::size_t>>& adjacency) {
    std::vector<std::vector<std::size_t>> components;
    std::vector<bool> visited(adjacency.size(), false);

    for (std::size_t start = 0u; start < adjacency.size(); ++start) {
        if (visited[start]) {
            continue;
        }

        std::vector<std::size_t> component;
        std::vector<std::size_t> stack{start};
        visited[start] = true;
        while (!stack.empty()) {
            const auto current = stack.back();
            stack.pop_back();
            component.push_back(current);
            for (const auto neighbor : adjacency[current]) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    stack.push_back(neighbor);
                }
            }
        }
        std::sort(component.begin(), component.end());
        components.push_back(std::move(component));
    }
    return components;
}

std::vector<std::vector<std::size_t>> component_adjacency(
    const std::vector<std::vector<std::size_t>>& adjacency,
    const std::vector<std::size_t>& component_atoms) {
    std::vector<std::size_t> local_indices(adjacency.size(), adjacency.size());
    for (std::size_t local = 0u; local < component_atoms.size(); ++local) {
        local_indices[component_atoms[local]] = local;
    }

    std::vector<std::vector<std::size_t>> component(component_atoms.size());
    for (std::size_t local = 0u; local < component_atoms.size(); ++local) {
        const auto global = component_atoms[local];
        for (const auto neighbor : adjacency[global]) {
            const auto local_neighbor = local_indices[neighbor];
            if (local_neighbor != adjacency.size()) {
                component[local].push_back(local_neighbor);
            }
        }
    }
    return component;
}

std::vector<std::vector<std::size_t>> compute_global_base_ring_cycles(
    const std::vector<std::vector<std::size_t>>& adjacency) {
    std::vector<std::vector<std::size_t>> cycles;
    for (const auto& atoms : connected_components(adjacency)) {
        const auto component_adjacency_values = component_adjacency(adjacency, atoms);
        const auto base_cycles = compute_component_base_ring_cycles(component_adjacency_values);
        for (const auto& cycle : base_cycles) {
            std::vector<std::size_t> global_cycle;
            global_cycle.reserve(cycle.size());
            for (const auto atom : cycle) {
                global_cycle.push_back(atoms[atom]);
            }
            cycles.push_back(std::move(global_cycle));
        }
    }
    return cycles;
}

std::vector<MordredRingAtomProperties> component_atom_properties(
    const std::vector<MordredRingAtomProperties>& atom_properties,
    const std::vector<std::size_t>& component_atoms) {
    std::vector<MordredRingAtomProperties> component;
    component.reserve(component_atoms.size());
    for (const auto atom : component_atoms) {
        component.push_back(atom_properties[atom]);
    }
    return component;
}

void add_ring_count_summary(
    MordredRingCountSummary& total,
    const MordredRingCountSummary& component) {
    total.total += component.total;
    for (std::size_t size = 0u; size < total.by_size.size(); ++size) {
        total.by_size[size] += component.by_size[size];
    }
    total.greater_or_equal_12 += component.greater_or_equal_12;
}

void add_ring_count_values(MordredRingCountValues& total, const MordredRingCountValues& component) {
    add_ring_count_summary(total.all, component.all);
    add_ring_count_summary(total.hetero, component.hetero);
    add_ring_count_summary(total.aromatic, component.aromatic);
    add_ring_count_summary(total.aromatic_hetero, component.aromatic_hetero);
    add_ring_count_summary(total.aliphatic, component.aliphatic);
    add_ring_count_summary(total.aliphatic_hetero, component.aliphatic_hetero);
}

MordredRingCountValueSets compute_ring_count_value_sets(const OEChem::OEMolBase& mol) {
    // Mordred/RDKit makes ring-selection decisions per connected component.
    // Fused descriptors then group only the selected Rings() basis, matching
    // Mordred's FusedRings dependency chain.
    std::unordered_map<unsigned int, std::size_t> atom_indices;
    std::vector<MordredRingAtomProperties> atom_properties;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (atom->GetAtomicNum() == 1) {
            continue;
        }
        atom_indices.emplace(atom->GetIdx(), atom_indices.size());
        atom_properties.push_back(
            {static_cast<std::uint32_t>(atom->GetAtomicNum()), atom->IsAromatic()});
    }

    std::vector<std::vector<std::size_t>> adjacency(atom_indices.size());
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
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

        const auto left = begin_index->second;
        const auto right = end_index->second;
        adjacency[left].push_back(right);
        adjacency[right].push_back(left);
    }

    for (auto& neighbors : adjacency) {
        std::sort(neighbors.begin(), neighbors.end());
    }

    MordredRingCountValueSets values;
    for (const auto& atoms : connected_components(adjacency)) {
        const auto component_adjacency_values = component_adjacency(adjacency, atoms);
        const auto component_properties = component_atom_properties(atom_properties, atoms);
        const auto base_cycles = compute_component_base_ring_cycles(component_adjacency_values);
        add_ring_count_values(values.base, count_ring_values(base_cycles, component_properties));
        add_ring_count_values(
            values.fused,
            count_ring_values(fused_ring_systems(base_cycles), component_properties));
    }
    return values;
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

double matrix_sum(const std::vector<double>& matrix) {
    double sum = 0.0;
    for (const auto value : matrix) {
        sum += value;
    }
    return sum;
}

double matrix_trace(const std::vector<double>& matrix, std::size_t size) {
    double trace = 0.0;
    for (std::size_t i = 0u; i < size; ++i) {
        trace += matrix[i * size + i];
    }
    return trace;
}

std::vector<double> multiply_square_matrices(
    const std::vector<double>& left,
    const std::vector<double>& right,
    std::size_t size) {
    std::vector<double> product(size * size, 0.0);
    for (std::size_t row = 0u; row < size; ++row) {
        for (std::size_t shared = 0u; shared < size; ++shared) {
            const auto left_value = left[row * size + shared];
            if (left_value == 0.0) {
                continue;
            }
            for (std::size_t column = 0u; column < size; ++column) {
                product[row * size + column] += left_value * right[shared * size + column];
            }
        }
    }
    return product;
}

MordredWalkCountValues compute_walk_count_values(const OEChem::OEMolBase& mol) {
    std::unordered_map<unsigned int, std::size_t> heavy_atom_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (!is_hydrogen(*atom)) {
            heavy_atom_indices.emplace(atom->GetIdx(), heavy_atom_indices.size());
        }
    }

    const auto heavy_atom_count = heavy_atom_indices.size();
    std::vector<double> adjacency(heavy_atom_count * heavy_atom_count, 0.0);
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
            continue;
        }

        const auto begin_index = heavy_atom_indices.find(begin->GetIdx());
        const auto end_index = heavy_atom_indices.find(end->GetIdx());
        if (begin_index == heavy_atom_indices.end() || end_index == heavy_atom_indices.end()) {
            continue;
        }
        adjacency[begin_index->second * heavy_atom_count + end_index->second] = 1.0;
        adjacency[end_index->second * heavy_atom_count + begin_index->second] = 1.0;
    }

    MordredWalkCountValues values;
    values.total_mwc10 = static_cast<double>(heavy_atom_count);
    values.total_srw10 = static_cast<double>(heavy_atom_count);
    auto power = adjacency;
    for (std::size_t order = 1u; order <= 10u; ++order) {
        const auto sum = matrix_sum(power);
        const auto trace = matrix_trace(power, heavy_atom_count);
        if (order == 1u) {
            values.mwc[order] = 0.5 * sum;
        } else {
            values.mwc[order] = std::log(sum + 1.0);
            values.srw[order] = std::log(trace + 1.0);
        }
        values.total_mwc10 += values.mwc[order];
        values.total_srw10 += values.srw[order];

        if (order != 10u) {
            power = multiply_square_matrices(power, adjacency, heavy_atom_count);
        }
    }
    return values;
}

struct PathCountNeighbor {
    std::size_t atom_index;
    double bond_order;
};

struct MordredHeavyAtomGraph {
    std::vector<const OEChem::OEAtomBase*> atoms;
    std::vector<std::vector<PathCountNeighbor>> adjacency;
    std::vector<std::pair<std::size_t, std::size_t>> bonds;
    std::vector<std::vector<std::size_t>> bond_neighbors;
};

struct MordredInformationContentValues {
    std::array<std::optional<double>, 6> ic;
    std::array<std::optional<double>, 6> tic;
    std::array<std::optional<double>, 6> sic;
    std::array<std::optional<double>, 6> bic;
    std::array<std::optional<double>, 6> cic;
    std::array<std::optional<double>, 6> mic;
    std::array<std::optional<double>, 6> zmic;
};

struct InformationContentNeighbor {
    std::size_t atom_index;
    std::uint32_t bond_order;
};

struct InformationContentAtom {
    const OEChem::OEAtomBase* atom = nullptr;
    std::uint32_t atomic_number = 0u;
    std::uint32_t degree = 0u;
    std::vector<InformationContentNeighbor> neighbors;
};

struct InformationContentGraph {
    OEChem::OEGraphMol mol;
    std::vector<InformationContentAtom> atoms;
    double bond_order_sum = 0.0;
};

struct InformationContentTreeNode {
    bool unexpanded = true;
    std::vector<std::pair<std::size_t, InformationContentTreeNode>> children;
};

using InformationContentTrail = std::vector<std::uint32_t>;
using InformationContentAtomCode = std::vector<InformationContentTrail>;

struct SimplePathWalkTotals {
    std::uint32_t count = 0u;
    double pi_sum = 0.0;
};

struct ChiPathWalkTotals {
    std::uint32_t count = 0u;
    double sum = 0.0;
    bool valid = true;
};

double mordred_bond_order(const OEChem::OEBondBase& bond) {
    if (bond.IsAromatic()) {
        return 1.5;
    }
    return static_cast<double>(bond.GetOrder());
}

MordredHeavyAtomGraph build_mordred_heavy_atom_graph(const OEChem::OEMolBase& mol) {
    MordredHeavyAtomGraph graph;
    std::unordered_map<unsigned int, std::size_t> heavy_atom_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (is_hydrogen(*atom)) {
            continue;
        }
        heavy_atom_indices.emplace(atom->GetIdx(), graph.atoms.size());
        graph.atoms.push_back(&*atom);
    }

    graph.adjacency.resize(graph.atoms.size());
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
            continue;
        }

        const auto begin_index = heavy_atom_indices.find(begin->GetIdx());
        const auto end_index = heavy_atom_indices.find(end->GetIdx());
        if (begin_index == heavy_atom_indices.end() || end_index == heavy_atom_indices.end()) {
            continue;
        }
        const auto bond_order = mordred_bond_order(*bond);
        graph.adjacency[begin_index->second].push_back({end_index->second, bond_order});
        graph.adjacency[end_index->second].push_back({begin_index->second, bond_order});
        graph.bonds.emplace_back(begin_index->second, end_index->second);
    }

    graph.bond_neighbors.resize(graph.bonds.size());
    for (std::size_t left = 0u; left < graph.bonds.size(); ++left) {
        const auto [left_begin, left_end] = graph.bonds[left];
        for (std::size_t right = left + 1u; right < graph.bonds.size(); ++right) {
            const auto [right_begin, right_end] = graph.bonds[right];
            if (left_begin == right_begin || left_begin == right_end || left_end == right_begin
                || left_end == right_end) {
                graph.bond_neighbors[left].push_back(right);
                graph.bond_neighbors[right].push_back(left);
            }
        }
    }
    return graph;
}

InformationContentGraph build_information_content_graph(const OEChem::OEMolBase& mol) {
    InformationContentGraph graph;
    graph.mol = explicit_hydrogen_copy(mol);

    std::unordered_map<unsigned int, std::size_t> atom_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = graph.mol.GetAtoms(); atom; ++atom) {
        atom_indices.emplace(atom->GetIdx(), graph.atoms.size());
        graph.atoms.push_back({
            &*atom,
            static_cast<std::uint32_t>(atom->GetAtomicNum()),
            static_cast<std::uint32_t>(atom->GetDegree()),
            {},
        });
    }

    for (OESystem::OEIter<OEChem::OEBondBase> bond = graph.mol.GetBonds(); bond; ++bond) {
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

        const auto bond_order = static_cast<std::uint32_t>(bond->GetOrder());
        graph.bond_order_sum += static_cast<double>(bond_order);
        graph.atoms[begin_index->second].neighbors.push_back({end_index->second, bond_order});
        graph.atoms[end_index->second].neighbors.push_back({begin_index->second, bond_order});
    }

    for (std::size_t atom_index = 0u; atom_index < graph.atoms.size(); ++atom_index) {
        auto& neighbors = graph.atoms[atom_index].neighbors;
        std::stable_sort(
            neighbors.begin(),
            neighbors.end(),
            [&graph, atom_index](const auto& left, const auto& right) {
                const auto category = [&graph, atom_index](const auto& neighbor) {
                    const auto& atom = graph.atoms[neighbor.atom_index];
                    if (atom.atomic_number == 1u) {
                        return 2;
                    }
                    const auto lhs = atom_index < neighbor.atom_index
                        ? neighbor.atom_index - atom_index
                        : atom_index - neighbor.atom_index;
                    return lhs == 1u ? 0 : 1;
                };
                return category(left) < category(right);
            });
    }

    return graph;
}

void expand_information_content_tree(
    const InformationContentGraph& graph,
    std::vector<std::pair<std::size_t, InformationContentTreeNode>>& tree,
    std::unordered_set<std::size_t>& visited) {
    for (auto& [src, node] : tree) {
        visited.insert(src);

        if (node.unexpanded) {
            node.unexpanded = false;
            for (const auto& neighbor : graph.atoms[src].neighbors) {
                if (visited.find(neighbor.atom_index) == visited.end()) {
                    node.children.push_back({neighbor.atom_index, {}});
                }
            }
        } else {
            expand_information_content_tree(graph, node.children, visited);
        }
    }
}

void collect_information_content_tree_code(
    const InformationContentGraph& graph,
    const std::vector<std::pair<std::size_t, InformationContentTreeNode>>& tree,
    std::optional<std::size_t> before,
    InformationContentTrail trail,
    InformationContentAtomCode& code) {
    if (tree.empty()) {
        code.push_back(std::move(trail));
        return;
    }

    for (const auto& [src, node] : tree) {
        auto next = trail;
        if (before.has_value()) {
            const auto& before_neighbors = graph.atoms[*before].neighbors;
            const auto found = std::find_if(
                before_neighbors.begin(),
                before_neighbors.end(),
                [src](const InformationContentNeighbor& neighbor) {
                    return neighbor.atom_index == src;
                });
            if (found != before_neighbors.end()) {
                next.push_back(found->bond_order);
            }
        }
        next.push_back(graph.atoms[src].atomic_number);
        next.push_back(graph.atoms[src].degree);
        collect_information_content_tree_code(graph, node.children, src, std::move(next), code);
    }
}

InformationContentAtomCode information_content_atom_code(
    const InformationContentGraph& graph,
    std::size_t atom_index,
    std::size_t order) {
    if (order == 0u) {
        return {{graph.atoms[atom_index].atomic_number}};
    }

    std::vector<std::pair<std::size_t, InformationContentTreeNode>> tree{{atom_index, {}}};
    std::unordered_set<std::size_t> visited;
    for (std::size_t step = 0u; step < order; ++step) {
        expand_information_content_tree(graph, tree, visited);
    }

    InformationContentAtomCode code;
    collect_information_content_tree_code(graph, tree, std::nullopt, {}, code);
    std::sort(code.begin(), code.end());
    return code;
}

std::optional<double> mordred_information_content_atom_mass(
    const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto isotope = static_cast<std::uint32_t>(atom.GetIsotope());
    if (isotope != 0u) {
        const auto mass = OEChem::OEGetIsotopicWeight(atomic_number, isotope);
        return std::round(mass * 100000000.0) / 100000000.0;
    }

    switch (atomic_number) {
    case 1u:
        return 1.008;
    case 6u:
        return 12.011;
    case 7u:
        return 14.007;
    case 8u:
        return 15.999;
    case 9u:
        return 18.998;
    case 11u:
        return 22.99;
    case 15u:
        return 30.974;
    case 16u:
        return 32.067;
    case 17u:
        return 35.453;
    case 34u:
        return 78.96;
    case 35u:
        return 79.904;
    case 53u:
        return 126.904;
    default:
        break;
    }

    const auto mass = mordred_mass(atomic_number);
    if (!mass.has_value()) {
        return std::nullopt;
    }
    return *mass;
}

std::optional<double> shannon_entropy(
    const std::vector<double>& class_sizes,
    const std::vector<double>& weights) {
    const double total = std::accumulate(class_sizes.begin(), class_sizes.end(), 0.0);
    if (total <= 0.0) {
        return std::nullopt;
    }

    double entropy = 0.0;
    for (std::size_t index = 0u; index < class_sizes.size(); ++index) {
        const auto probability = class_sizes[index] / total;
        if (probability > 0.0) {
            entropy -= weights[index] * probability * std::log2(probability);
        }
    }
    return entropy;
}

std::optional<double> divide_by_valid_log2(double value, double denominator_argument) {
    if (denominator_argument <= 0.0) {
        return std::nullopt;
    }

    const auto denominator = std::log2(denominator_argument);
    if (!std::isfinite(denominator) || denominator == 0.0) {
        return std::nullopt;
    }
    return value / denominator;
}

MordredInformationContentValues compute_information_content_values(
    const OEChem::OEMolBase& mol) {
    const auto graph = build_information_content_graph(mol);
    MordredInformationContentValues values;
    const auto atom_count = graph.atoms.size();
    if (atom_count == 0u) {
        return values;
    }

    for (std::size_t order = 0u; order <= 5u; ++order) {
        std::vector<InformationContentAtomCode> atom_codes;
        atom_codes.reserve(atom_count);
        std::map<InformationContentAtomCode, std::size_t> representative_ids;
        for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
            atom_codes.push_back(information_content_atom_code(graph, atom_index, order));
            representative_ids[atom_codes.back()] = atom_index;
        }

        std::sort(atom_codes.begin(), atom_codes.end());

        std::vector<double> class_sizes;
        std::vector<std::size_t> representative_indices;
        for (std::size_t begin = 0u; begin < atom_codes.size();) {
            std::size_t end = begin + 1u;
            while (end < atom_codes.size() && atom_codes[end] == atom_codes[begin]) {
                ++end;
            }
            class_sizes.push_back(static_cast<double>(end - begin));
            representative_indices.push_back(representative_ids[atom_codes[begin]]);
            begin = end;
        }

        std::vector<double> unit_weights(class_sizes.size(), 1.0);
        const auto ic = shannon_entropy(class_sizes, unit_weights);
        if (!ic.has_value()) {
            continue;
        }

        values.ic[order] = *ic;
        values.tic[order] = static_cast<double>(atom_count) * *ic;
        values.sic[order] = divide_by_valid_log2(*ic, static_cast<double>(atom_count));
        values.bic[order] = divide_by_valid_log2(*ic, graph.bond_order_sum);
        values.cic[order] = std::log2(static_cast<double>(atom_count)) - *ic;

        std::vector<double> mass_weights;
        std::vector<double> z_weights;
        mass_weights.reserve(class_sizes.size());
        z_weights.reserve(class_sizes.size());
        bool weights_valid = true;
        for (std::size_t class_index = 0u; class_index < class_sizes.size(); ++class_index) {
            const auto representative_index = representative_indices[class_index];
            const auto* atom = graph.atoms[representative_index].atom;
            if (atom == nullptr) {
                weights_valid = false;
                break;
            }
            const auto mass = mordred_information_content_atom_mass(*atom);
            if (!mass.has_value()) {
                weights_valid = false;
                break;
            }
            mass_weights.push_back(*mass);
            z_weights.push_back(
                class_sizes[class_index]
                * static_cast<double>(graph.atoms[representative_index].atomic_number));
        }

        if (weights_valid) {
            values.mic[order] = shannon_entropy(class_sizes, mass_weights);
            values.zmic[order] = shannon_entropy(class_sizes, z_weights);
        }
    }

    return values;
}

bool is_connected_heavy_atom_graph(const MordredHeavyAtomGraph& graph) {
    const auto atom_count = graph.atoms.size();
    if (atom_count == 0u) {
        return false;
    }

    std::vector<bool> visited(atom_count, false);
    std::vector<std::size_t> queue;
    queue.reserve(atom_count);
    visited.front() = true;
    queue.push_back(0u);

    for (std::size_t head = 0u; head < queue.size(); ++head) {
        const auto current = queue[head];
        for (const auto& neighbor : graph.adjacency[current]) {
            if (visited[neighbor.atom_index]) {
                continue;
            }
            visited[neighbor.atom_index] = true;
            queue.push_back(neighbor.atom_index);
        }
    }

    return std::all_of(visited.begin(), visited.end(), [](bool value) { return value; });
}

std::vector<std::vector<std::size_t>> simple_heavy_atom_adjacency(
    const MordredHeavyAtomGraph& graph) {
    std::vector<std::vector<std::size_t>> adjacency(graph.atoms.size());
    for (std::size_t atom = 0u; atom < graph.adjacency.size(); ++atom) {
        adjacency[atom].reserve(graph.adjacency[atom].size());
        for (const auto& neighbor : graph.adjacency[atom]) {
            adjacency[atom].push_back(neighbor.atom_index);
        }
        std::sort(adjacency[atom].begin(), adjacency[atom].end());
    }
    return adjacency;
}

void add_collapsed_edge(
    std::vector<std::vector<std::size_t>>& adjacency,
    std::size_t left,
    std::size_t right) {
    if (left == right) {
        return;
    }
    adjacency[left].push_back(right);
    adjacency[right].push_back(left);
}

void collect_shortest_path_linkers(
    const std::vector<std::vector<std::size_t>>& collapsed_adjacency,
    std::size_t atom_count,
    std::size_t source,
    std::size_t target,
    std::vector<bool>& linkers) {
    std::vector<std::size_t> parent(collapsed_adjacency.size(), collapsed_adjacency.size());
    std::vector<std::size_t> queue;
    queue.reserve(collapsed_adjacency.size());
    parent[source] = source;
    queue.push_back(source);

    for (std::size_t current_index = 0u;
         current_index < queue.size() && parent[target] == collapsed_adjacency.size();
         ++current_index) {
        const auto current = queue[current_index];
        for (const auto neighbor : collapsed_adjacency[current]) {
            if (parent[neighbor] != collapsed_adjacency.size()) {
                continue;
            }
            parent[neighbor] = current;
            queue.push_back(neighbor);
            if (neighbor == target) {
                break;
            }
        }
    }

    if (parent[target] == collapsed_adjacency.size()) {
        return;
    }

    for (auto node = target; node != source; node = parent[node]) {
        if (node < atom_count) {
            linkers[node] = true;
        }
    }
}

std::optional<double> compute_framework_ratio(
    const MordredHeavyAtomGraph& graph,
    std::uint32_t framework_atom_count) {
    const auto atom_count = graph.atoms.size();
    if (atom_count == 0u || framework_atom_count == 0u) {
        return std::nullopt;
    }

    const auto rings = compute_global_base_ring_cycles(simple_heavy_atom_adjacency(graph));
    if (rings.empty()) {
        return 0.0;
    }

    std::vector<std::size_t> collapsed_node_for_atom(atom_count);
    std::iota(collapsed_node_for_atom.begin(), collapsed_node_for_atom.end(), 0u);
    std::vector<bool> ring_atoms(atom_count, false);
    for (std::size_t ring_index = 0u; ring_index < rings.size(); ++ring_index) {
        const auto ring_node = atom_count + ring_index;
        for (const auto atom : rings[ring_index]) {
            collapsed_node_for_atom[atom] = ring_node;
            ring_atoms[atom] = true;
        }
    }

    std::vector<std::vector<std::size_t>> collapsed_adjacency(atom_count + rings.size());
    for (const auto& bond : graph.bonds) {
        add_collapsed_edge(
            collapsed_adjacency,
            collapsed_node_for_atom[bond.first],
            collapsed_node_for_atom[bond.second]);
    }

    std::vector<bool> linkers(atom_count, false);
    for (std::size_t left = 0u; left < rings.size(); ++left) {
        for (std::size_t right = left + 1u; right < rings.size(); ++right) {
            collect_shortest_path_linkers(
                collapsed_adjacency,
                atom_count,
                atom_count + left,
                atom_count + right,
                linkers);
        }
    }

    const auto linker_count =
        static_cast<std::size_t>(std::count(linkers.begin(), linkers.end(), true));
    const auto ring_atom_count =
        static_cast<std::size_t>(std::count(ring_atoms.begin(), ring_atoms.end(), true));
    return static_cast<double>(linker_count + ring_atom_count)
           / static_cast<double>(framework_atom_count);
}

void accumulate_molecular_id_paths(
    const MordredHeavyAtomGraph& graph,
    std::size_t atom_index,
    double previous_weight,
    std::vector<bool>& visited,
    double& atomic_id) {
    constexpr double kWeightLimit = 1.0e20;
    visited[atom_index] = true;
    const auto degree = static_cast<double>(graph.adjacency[atom_index].size());
    for (const auto& neighbor : graph.adjacency[atom_index]) {
        if (visited[neighbor.atom_index]) {
            continue;
        }

        visited[neighbor.atom_index] = true;
        const auto neighbor_degree =
            static_cast<double>(graph.adjacency[neighbor.atom_index].size());
        const auto weight = degree * neighbor_degree * previous_weight;
        atomic_id += 1.0 / std::sqrt(weight);
        if (weight < kWeightLimit) {
            accumulate_molecular_id_paths(
                graph,
                neighbor.atom_index,
                weight,
                visited,
                atomic_id);
        }
        visited[neighbor.atom_index] = false;
    }
}

std::optional<std::vector<double>> compute_mordred_atomic_ids(
    const MordredHeavyAtomGraph& graph) {
    if (!is_connected_heavy_atom_graph(graph)) {
        return std::nullopt;
    }

    std::vector<double> atomic_ids;
    atomic_ids.reserve(graph.atoms.size());
    for (std::size_t atom_index = 0u; atom_index < graph.atoms.size(); ++atom_index) {
        std::vector<bool> visited(graph.atoms.size(), false);
        double atomic_id = 0.0;
        accumulate_molecular_id_paths(graph, atom_index, 1.0, visited, atomic_id);
        atomic_ids.push_back(1.0 + atomic_id / 2.0);
    }
    return atomic_ids;
}

std::optional<MordredMolecularIdValues> compute_molecular_id_values(
    const MordredHeavyAtomGraph& graph) {
    const auto atomic_ids = compute_mordred_atomic_ids(graph);
    if (!atomic_ids.has_value()) {
        return std::nullopt;
    }

    MordredMolecularIdValues values;
    values.atom_count = atomic_ids->size();
    for (std::size_t atom_index = 0u; atom_index < atomic_ids->size(); ++atom_index) {
        const auto atomic_id = (*atomic_ids)[atom_index];
        const auto atomic_number =
            static_cast<std::uint32_t>(graph.atoms[atom_index]->GetAtomicNum());
        values.any += atomic_id;
        if (atomic_number != 1u && atomic_number != 6u) {
            values.hetero += atomic_id;
        }
        if (atomic_number == 6u) {
            values.carbon += atomic_id;
        } else if (atomic_number == 7u) {
            values.nitrogen += atomic_id;
        } else if (atomic_number == 8u) {
            values.oxygen += atomic_id;
        }
        if (is_mordred_molecular_id_halogen(atomic_number)) {
            values.halogen += atomic_id;
        }
    }
    return values;
}

std::optional<MordredSymmetricEigensystem> symmetric_eigensystem_jacobi(
    std::vector<double> matrix,
    std::size_t dimension) {
    MordredSymmetricEigensystem eigensystem;
    if (dimension == 0u) {
        return eigensystem;
    }

    eigensystem.eigenvectors.assign(dimension * dimension, 0.0);
    for (std::size_t index = 0u; index < dimension; ++index) {
        eigensystem.eigenvectors[index * dimension + index] = 1.0;
    }

    if (dimension == 1u) {
        eigensystem.eigenvalues = {matrix.front()};
        return eigensystem;
    }

    constexpr double kTolerance = 1.0e-13;
    const auto at = [dimension](std::size_t row, std::size_t column) {
        return row * dimension + column;
    };
    const auto max_iterations = std::max<std::size_t>(100u, 100u * dimension * dimension);

    for (std::size_t iteration = 0u; iteration < max_iterations; ++iteration) {
        std::size_t pivot_row = 0u;
        std::size_t pivot_column = 1u;
        double max_off_diagonal = std::abs(matrix[at(pivot_row, pivot_column)]);

        for (std::size_t row = 0u; row < dimension; ++row) {
            for (std::size_t column = row + 1u; column < dimension; ++column) {
                const auto value = std::abs(matrix[at(row, column)]);
                if (value > max_off_diagonal) {
                    max_off_diagonal = value;
                    pivot_row = row;
                    pivot_column = column;
                }
            }
        }

        if (max_off_diagonal <= kTolerance) {
            eigensystem.eigenvalues.clear();
            eigensystem.eigenvalues.reserve(dimension);
            for (std::size_t index = 0u; index < dimension; ++index) {
                eigensystem.eigenvalues.push_back(matrix[at(index, index)]);
            }
            return eigensystem;
        }

        const auto app = matrix[at(pivot_row, pivot_row)];
        const auto aqq = matrix[at(pivot_column, pivot_column)];
        const auto apq = matrix[at(pivot_row, pivot_column)];
        const auto tau = (aqq - app) / (2.0 * apq);
        const auto t = std::copysign(
            1.0 / (std::abs(tau) + std::sqrt(1.0 + tau * tau)),
            tau);
        const auto c = 1.0 / std::sqrt(1.0 + t * t);
        const auto s = t * c;

        matrix[at(pivot_row, pivot_row)] = app - t * apq;
        matrix[at(pivot_column, pivot_column)] = aqq + t * apq;
        matrix[at(pivot_row, pivot_column)] = 0.0;
        matrix[at(pivot_column, pivot_row)] = 0.0;

        for (std::size_t index = 0u; index < dimension; ++index) {
            if (index == pivot_row || index == pivot_column) {
                continue;
            }

            const auto aip = matrix[at(index, pivot_row)];
            const auto aiq = matrix[at(index, pivot_column)];
            const auto rotated_ip = c * aip - s * aiq;
            const auto rotated_iq = s * aip + c * aiq;
            matrix[at(index, pivot_row)] = rotated_ip;
            matrix[at(pivot_row, index)] = rotated_ip;
            matrix[at(index, pivot_column)] = rotated_iq;
            matrix[at(pivot_column, index)] = rotated_iq;
        }

        // Mordred/NumPy returns eigenvectors as columns; keep the same layout.
        for (std::size_t row = 0u; row < dimension; ++row) {
            const auto vip = eigensystem.eigenvectors[at(row, pivot_row)];
            const auto viq = eigensystem.eigenvectors[at(row, pivot_column)];
            eigensystem.eigenvectors[at(row, pivot_row)] = c * vip - s * viq;
            eigensystem.eigenvectors[at(row, pivot_column)] = s * vip + c * viq;
        }
    }

    return std::nullopt;
}

bool has_mordred_3d_coordinates(const OEChem::OEMolBase& mol) {
    if (mol.NumAtoms() == 0u || mol.GetDimension() != 3u) {
        return false;
    }

    double coords[3] = {0.0, 0.0, 0.0};
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        if (!mol.GetCoords(atom, coords)) {
            return false;
        }
    }
    return true;
}

std::optional<OEChem::OEMol> mordred_omega_single_conformer_copy(
    const OEChem::OEMolBase& mol) {
    OEChem::OEMol working_mol(mol);
    if (has_mordred_3d_coordinates(working_mol)) {
        return working_mol;
    }

    OEConfGen::OEOmegaOptions options;
    options.SetMaxConfs(1u);
    options.SetStrictAtomTypes(false);
    OEConfGen::OEOmega omega(options);
    const auto return_code = omega.Build(working_mol);
    if (return_code == OEConfGen::OEOmegaReturnCode::Success
            && has_mordred_3d_coordinates(working_mol)) {
        return working_mol;
    }
    return std::nullopt;
}

double mordred_3d_distance(
    const std::array<double, 3>& left,
    const std::array<double, 3>& right) {
    const auto dx = left[0] - right[0];
    const auto dy = left[1] - right[1];
    const auto dz = left[2] - right[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::optional<Mordred3DAtomSet> build_mordred_3d_atom_set(
    const OEChem::OEMolBase& mol,
    bool include_hydrogens) {
    Mordred3DAtomSet atom_set;
    std::vector<std::optional<std::size_t>> selected_by_atom_index(mol.GetMaxAtomIdx());

    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const auto atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        if (!include_hydrogens && atomic_number <= OEChem::OEElemNo::H) {
            continue;
        }

        double coords[3] = {0.0, 0.0, 0.0};
        if (!mol.GetCoords(atom, coords)) {
            return std::nullopt;
        }

        const auto mass = mordred_mass(atomic_number);
        if (!mass.has_value()) {
            return std::nullopt;
        }

        const auto selected_index = atom_set.atoms.size();
        if (atom->GetIdx() >= selected_by_atom_index.size()) {
            selected_by_atom_index.resize(atom->GetIdx() + 1u);
        }
        selected_by_atom_index[atom->GetIdx()] = selected_index;
        atom_set.atoms.push_back(Mordred3DAtom{
            atom->GetIdx(),
            atomic_number,
            *mass,
            {coords[0], coords[1], coords[2]},
        });
    }

    const auto atom_count = atom_set.atoms.size();
    atom_set.distances.assign(atom_count * atom_count, 0.0);
    atom_set.adjacency.assign(atom_count * atom_count, 0u);
    for (std::size_t row = 0u; row < atom_count; ++row) {
        for (std::size_t column = row + 1u; column < atom_count; ++column) {
            const auto distance = mordred_3d_distance(
                atom_set.atoms[row].coord,
                atom_set.atoms[column].coord);
            atom_set.distances[row * atom_count + column] = distance;
            atom_set.distances[column * atom_count + row] = distance;
        }
    }

    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        const auto begin_index = bond->GetBgnIdx();
        const auto end_index = bond->GetEndIdx();
        if (begin_index >= selected_by_atom_index.size()
                || end_index >= selected_by_atom_index.size()) {
            continue;
        }
        const auto begin = selected_by_atom_index[begin_index];
        const auto end = selected_by_atom_index[end_index];
        if (!begin.has_value() || !end.has_value()) {
            continue;
        }
        atom_set.adjacency[*begin * atom_count + *end] = 1u;
        atom_set.adjacency[*end * atom_count + *begin] = 1u;
    }

    return atom_set;
}

std::optional<Mordred3DContext> build_mordred_3d_context(const OEChem::OEMolBase& mol) {
    auto explicit_mol = explicit_hydrogen_copy(mol);
    auto three_d_mol = mordred_omega_single_conformer_copy(explicit_mol);
    if (!three_d_mol.has_value()) {
        return std::nullopt;
    }

    auto all_atom_set = build_mordred_3d_atom_set(*three_d_mol, true);
    auto heavy_atom_set = build_mordred_3d_atom_set(*three_d_mol, false);
    if (!all_atom_set.has_value() || !heavy_atom_set.has_value()) {
        return std::nullopt;
    }

    Mordred3DContext context;
    context.all_atom_set = std::move(*all_atom_set);
    context.heavy_atom_set = std::move(*heavy_atom_set);
    return context;
}

std::optional<std::vector<double>> mordred_morse_atom_weights(
    const Mordred3DAtomSet& atom_set,
    const MordredMoRSEProperty& property) {
    std::vector<double> weights;
    weights.reserve(atom_set.atoms.size());

    if (property.lookup == nullptr) {
        weights.assign(atom_set.atoms.size(), 1.0);
        return weights;
    }

    for (const auto& atom : atom_set.atoms) {
        const auto value = property.lookup(atom.atomic_number);
        if (!value.has_value()) {
            return std::nullopt;
        }
        weights.push_back(*value / property.carbon_reference);
    }
    return weights;
}

std::optional<double> compute_mordred_morse_value(
    const Mordred3DAtomSet& atom_set,
    const std::vector<double>& weights,
    std::size_t distance_index) {
    const auto atom_count = atom_set.atoms.size();
    if (atom_count <= 1u) {
        return std::nullopt;
    }

    double total = 0.0;
    for (std::size_t row = 0u; row < atom_count; ++row) {
        for (std::size_t column = 0u; column < atom_count; ++column) {
            if (row == column) {
                continue;
            }

            double kernel = 1.0;
            if (distance_index > 1u) {
                const auto sr = static_cast<double>(distance_index - 1u)
                    * atom_set.distances[row * atom_count + column];
                if (sr == 0.0) {
                    return std::nullopt;
                }
                kernel = std::sin(sr) / sr;
            }
            total += weights[row] * kernel * weights[column];
        }
    }
    return 0.5 * total;
}

std::vector<MordredMoRSEValues> compute_mordred_morse_values(
    const std::optional<Mordred3DContext>& context) {
    if (!context.has_value() || context->all_atom_set.atoms.size() <= 1u) {
        return {};
    }

    std::vector<MordredMoRSEValues> values;
    values.reserve(mordred_morse_properties().size());
    for (const auto& property : mordred_morse_properties()) {
        const auto weights = mordred_morse_atom_weights(context->all_atom_set, property);
        if (!weights.has_value()) {
            continue;
        }

        MordredMoRSEValues property_values;
        property_values.suffix = property.suffix;
        for (std::size_t distance_index = 1u; distance_index <= 32u; ++distance_index) {
            property_values.values[distance_index] =
                compute_mordred_morse_value(
                    context->all_atom_set,
                    *weights,
                    distance_index);
        }
        values.push_back(std::move(property_values));
    }
    return values;
}

std::optional<MordredGeometricalIndexValues> compute_geometrical_index_values(
    const Mordred3DAtomSet& atom_set) {
    const auto atom_count = atom_set.atoms.size();
    if (atom_count == 0u) {
        return std::nullopt;
    }

    MordredGeometricalIndexValues values;
    values.radius = std::numeric_limits<double>::infinity();
    for (std::size_t column = 0u; column < atom_count; ++column) {
        double eccentricity = 0.0;
        for (std::size_t row = 0u; row < atom_count; ++row) {
            const auto distance = atom_set.distances[row * atom_count + column];
            eccentricity = std::max(eccentricity, distance);
            values.diameter = std::max(values.diameter, distance);
        }
        values.radius = std::min(values.radius, eccentricity);
    }

    if (values.radius != 0.0) {
        values.shape_index = (values.diameter - values.radius) / values.radius;
    }
    if (values.diameter != 0.0) {
        values.petitjean_index = (values.diameter - values.radius) / values.diameter;
    }
    return values;
}

std::optional<double> compute_gravitational_index_value(
    const Mordred3DAtomSet& atom_set,
    bool pair_only) {
    const auto atom_count = atom_set.atoms.size();
    double total = 0.0;

    for (std::size_t row = 0u; row < atom_count; ++row) {
        for (std::size_t column = 0u; column < atom_count; ++column) {
            if (row == column) {
                continue;
            }
            if (pair_only && atom_set.adjacency[row * atom_count + column] == 0u) {
                continue;
            }

            const auto distance = atom_set.distances[row * atom_count + column];
            if (distance == 0.0) {
                return std::nullopt;
            }
            total += atom_set.atoms[row].mass * atom_set.atoms[column].mass
                / (distance * distance);
        }
    }

    return 0.5 * total;
}

MordredGravitationalIndexValues compute_gravitational_index_values(
    const Mordred3DAtomSet& heavy_atom_set,
    const Mordred3DAtomSet& all_atom_set) {
    MordredGravitationalIndexValues values;
    values.heavy = compute_gravitational_index_value(heavy_atom_set, false);
    values.all = compute_gravitational_index_value(all_atom_set, false);
    values.heavy_pair = compute_gravitational_index_value(heavy_atom_set, true);
    values.all_pair = compute_gravitational_index_value(all_atom_set, true);
    return values;
}

std::array<double, 3> mordred_centroid(const Mordred3DAtomSet& atom_set) {
    std::array<double, 3> centroid{};
    if (atom_set.atoms.empty()) {
        return centroid;
    }
    for (const auto& atom : atom_set.atoms) {
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            centroid[axis] += atom.coord[axis];
        }
    }
    for (auto& value : centroid) {
        value /= static_cast<double>(atom_set.atoms.size());
    }
    return centroid;
}

std::optional<MordredMomentOfInertiaValues> compute_moment_of_inertia_values(
    const Mordred3DAtomSet& atom_set) {
    if (atom_set.atoms.empty()) {
        return std::nullopt;
    }

    double mass_sum = 0.0;
    std::array<double, 3> center_of_mass{};
    for (const auto& atom : atom_set.atoms) {
        mass_sum += atom.mass;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            center_of_mass[axis] += atom.mass * atom.coord[axis];
        }
    }
    if (mass_sum == 0.0) {
        return std::nullopt;
    }
    for (auto& value : center_of_mass) {
        value /= mass_sum;
    }

    std::vector<double> inertia(9u, 0.0);
    const auto at = [](std::size_t row, std::size_t column) {
        return row * 3u + column;
    };
    for (const auto& atom : atom_set.atoms) {
        std::array<double, 3> shifted{};
        double radius2 = 0.0;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            shifted[axis] = atom.coord[axis] - center_of_mass[axis];
            radius2 += shifted[axis] * shifted[axis];
        }
        for (std::size_t row = 0u; row < 3u; ++row) {
            for (std::size_t column = 0u; column < 3u; ++column) {
                if (row == column) {
                    inertia[at(row, column)] +=
                        atom.mass * (radius2 - shifted[row] * shifted[column]);
                } else {
                    inertia[at(row, column)] -= atom.mass * shifted[row] * shifted[column];
                }
            }
        }
    }

    auto eigensystem = symmetric_eigensystem_jacobi(std::move(inertia), 3u);
    if (!eigensystem.has_value() || eigensystem->eigenvalues.size() != 3u) {
        return std::nullopt;
    }

    MordredMomentOfInertiaValues values;
    std::copy_n(eigensystem->eigenvalues.begin(), 3u, values.axes.begin());
    std::sort(values.axes.begin(), values.axes.end(), std::greater<double>{});
    return values;
}

std::optional<double> compute_pbf_value(const Mordred3DAtomSet& atom_set) {
    const auto atom_count = atom_set.atoms.size();
    if (atom_count < 4u) {
        return 0.0;
    }

    const auto centroid = mordred_centroid(atom_set);
    std::vector<double> covariance(9u, 0.0);
    const auto at = [](std::size_t row, std::size_t column) {
        return row * 3u + column;
    };
    for (const auto& atom : atom_set.atoms) {
        std::array<double, 3> shifted{};
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            shifted[axis] = atom.coord[axis] - centroid[axis];
        }
        for (std::size_t row = 0u; row < 3u; ++row) {
            for (std::size_t column = 0u; column < 3u; ++column) {
                covariance[at(row, column)] += shifted[row] * shifted[column];
            }
        }
    }

    auto eigensystem = symmetric_eigensystem_jacobi(std::move(covariance), 3u);
    if (!eigensystem.has_value() || eigensystem->eigenvalues.size() != 3u) {
        return std::nullopt;
    }
    const auto normal_index = static_cast<std::size_t>(std::distance(
        eigensystem->eigenvalues.begin(),
        std::min_element(eigensystem->eigenvalues.begin(), eigensystem->eigenvalues.end())));
    std::array<double, 3> normal{
        eigensystem->eigenvectors[at(0u, normal_index)],
        eigensystem->eigenvectors[at(1u, normal_index)],
        eigensystem->eigenvectors[at(2u, normal_index)],
    };
    const auto normal_norm = std::sqrt(
        normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
    if (normal_norm == 0.0) {
        return 0.0;
    }

    double total_distance = 0.0;
    for (const auto& atom : atom_set.atoms) {
        double projection = 0.0;
        for (std::size_t axis = 0u; axis < 3u; ++axis) {
            projection += (atom.coord[axis] - centroid[axis]) * normal[axis];
        }
        total_distance += std::abs(projection) / normal_norm;
    }
    return total_distance / static_cast<double>(atom_count);
}

std::optional<MordredLowCount3DValues> compute_low_count_3d_values(
    const std::optional<Mordred3DContext>& context) {
    if (!context.has_value()) {
        return std::nullopt;
    }

    MordredLowCount3DValues values;
    values.geometrical = compute_geometrical_index_values(context->all_atom_set);
    values.gravitational =
        compute_gravitational_index_values(context->heavy_atom_set, context->all_atom_set);
    values.moment_of_inertia = compute_moment_of_inertia_values(context->all_atom_set);
    values.pbf = compute_pbf_value(context->all_atom_set);
    return values;
}

std::optional<MordredMatrixEigenvalueValues> compute_matrix_eigenvalue_values(
    std::vector<double> matrix,
    std::size_t atom_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& bonds) {
    const auto spectral_moment = matrix_trace(matrix, atom_count);
    auto eigensystem = symmetric_eigensystem_jacobi(std::move(matrix), atom_count);
    if (!eigensystem.has_value()) {
        return std::nullopt;
    }

    MordredMatrixEigenvalueValues values;
    const auto& eigenvalues = eigensystem->eigenvalues;
    const auto [min_eigenvalue, max_eigenvalue] =
        std::minmax_element(eigenvalues.begin(), eigenvalues.end());
    const auto mean = std::accumulate(eigenvalues.begin(), eigenvalues.end(), 0.0)
        / static_cast<double>(atom_count);

    for (const auto eigenvalue : eigenvalues) {
        values.spectral_absolute += std::abs(eigenvalue);
        values.spectral_absolute_deviation += std::abs(eigenvalue - mean);
    }

    values.spectral_max = *max_eigenvalue;
    values.spectral_diameter = *max_eigenvalue - *min_eigenvalue;
    values.spectral_mean_absolute_deviation =
        values.spectral_absolute_deviation / static_cast<double>(atom_count);
    values.spectral_moment = spectral_moment;

    const auto a = std::max(values.spectral_max, 0.0);
    double exp_sum = std::exp(-a);
    for (const auto eigenvalue : eigenvalues) {
        exp_sum += std::exp(eigenvalue - a);
    }
    values.log_estrada_like = a + std::log(exp_sum);

    const auto dominant_index =
        static_cast<std::size_t>(std::distance(eigenvalues.begin(), max_eigenvalue));
    for (std::size_t row = 0u; row < atom_count; ++row) {
        values.eigenvector_coefficient_sum +=
            std::abs(eigensystem->eigenvectors[row * atom_count + dominant_index]);
    }
    values.eigenvector_coefficient_mean =
        values.eigenvector_coefficient_sum / static_cast<double>(atom_count);

    const auto eigenvector_log_argument =
        0.1 * static_cast<double>(atom_count) * values.eigenvector_coefficient_sum;
    if (eigenvector_log_argument <= 0.0) {
        return std::nullopt;
    }
    values.eigenvector_coefficient_log = std::log(eigenvector_log_argument);

    for (const auto [begin, end] : bonds) {
        const auto begin_coefficient =
            eigensystem->eigenvectors[begin * atom_count + dominant_index];
        const auto end_coefficient = eigensystem->eigenvectors[end * atom_count + dominant_index];
        values.randic_eigenvector_sum += std::pow(begin_coefficient * end_coefficient, -0.5);
    }
    values.randic_eigenvector_mean =
        values.randic_eigenvector_sum / static_cast<double>(atom_count);

    const auto randic_log_argument =
        0.1 * static_cast<double>(atom_count) * values.randic_eigenvector_sum;
    if (randic_log_argument > 0.0) {
        values.randic_eigenvector_log = std::log(randic_log_argument);
    }

    return values;
}

std::optional<MordredMatrixEigenvalueValues> compute_adjacency_matrix_eigenvalue_values(
    const MordredHeavyAtomGraph& graph) {
    if (!is_connected_heavy_atom_graph(graph)) {
        return std::nullopt;
    }

    const auto atom_count = graph.atoms.size();
    std::vector<double> adjacency_matrix(atom_count * atom_count, 0.0);
    for (const auto [begin, end] : graph.bonds) {
        adjacency_matrix[begin * atom_count + end] = 1.0;
        adjacency_matrix[end * atom_count + begin] = 1.0;
    }

    return compute_matrix_eigenvalue_values(std::move(adjacency_matrix), atom_count, graph.bonds);
}

std::optional<MordredMatrixEigenvalueValues> compute_distance_matrix_eigenvalue_values(
    const MordredHeavyAtomGraph& graph,
    const std::vector<std::vector<std::int64_t>>& distances) {
    if (!is_connected_heavy_atom_graph(graph)) {
        return std::nullopt;
    }

    const auto atom_count = graph.atoms.size();
    std::vector<double> distance_matrix;
    distance_matrix.reserve(atom_count * atom_count);
    for (const auto& row : distances) {
        for (const auto distance : row) {
            distance_matrix.push_back(static_cast<double>(distance));
        }
    }

    return compute_matrix_eigenvalue_values(std::move(distance_matrix), atom_count, graph.bonds);
}

std::int64_t& matrix_entry(
    std::vector<std::int64_t>& matrix,
    std::size_t size,
    std::size_t row,
    std::size_t column) {
    return matrix[row * size + column];
}

std::int64_t matrix_entry(
    const std::vector<std::int64_t>& matrix,
    std::size_t size,
    std::size_t row,
    std::size_t column) {
    return matrix[row * size + column];
}

std::pair<std::size_t, std::size_t> normalized_edge(std::size_t begin, std::size_t end) {
    return begin < end ? std::make_pair(begin, end) : std::make_pair(end, begin);
}

// Mordred uses a wall-clock timeout here; OEFP uses a deterministic budget so
// pathological cyclic blocks return missing without making tests timing-sensitive.
constexpr std::uint64_t kDefaultMordredDetourSearchOperations = 5'000'000u;

bool consume_mordred_detour_search_operation(MordredDetourSearchContext& context) {
    if (context.operations >= context.max_operations) {
        return false;
    }
    ++context.operations;
    return true;
}

void collect_mordred_biconnected_components(
    const std::vector<std::vector<std::size_t>>& adjacency,
    std::size_t atom,
    std::size_t parent,
    std::size_t& visit_order,
    std::vector<std::size_t>& discovery,
    std::vector<std::size_t>& lowlink,
    std::vector<std::pair<std::size_t, std::size_t>>& edge_stack,
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>>& components) {
    discovery[atom] = ++visit_order;
    lowlink[atom] = discovery[atom];

    for (const auto neighbor : adjacency[atom]) {
        if (discovery[neighbor] == 0u) {
            edge_stack.push_back(normalized_edge(atom, neighbor));
            collect_mordred_biconnected_components(
                adjacency,
                neighbor,
                atom,
                visit_order,
                discovery,
                lowlink,
                edge_stack,
                components);
            lowlink[atom] = std::min(lowlink[atom], lowlink[neighbor]);

            if (lowlink[neighbor] >= discovery[atom]) {
                std::vector<std::pair<std::size_t, std::size_t>> component;
                const auto stop_edge = normalized_edge(atom, neighbor);
                while (!edge_stack.empty()) {
                    const auto edge = edge_stack.back();
                    edge_stack.pop_back();
                    component.push_back(edge);
                    if (edge == stop_edge) {
                        break;
                    }
                }
                components.push_back(std::move(component));
            }
        } else if (neighbor != parent && discovery[neighbor] < discovery[atom]) {
            edge_stack.push_back(normalized_edge(atom, neighbor));
            lowlink[atom] = std::min(lowlink[atom], discovery[neighbor]);
        }
    }
}

std::vector<std::vector<std::pair<std::size_t, std::size_t>>> mordred_biconnected_edge_components(
    const MordredHeavyAtomGraph& graph) {
    const auto adjacency = simple_heavy_atom_adjacency(graph);
    std::vector<std::size_t> discovery(graph.atoms.size(), 0u);
    std::vector<std::size_t> lowlink(graph.atoms.size(), 0u);
    std::vector<std::pair<std::size_t, std::size_t>> edge_stack;
    std::vector<std::vector<std::pair<std::size_t, std::size_t>>> components;
    std::size_t visit_order = 0u;

    for (std::size_t atom = 0u; atom < graph.atoms.size(); ++atom) {
        if (discovery[atom] != 0u) {
            continue;
        }
        collect_mordred_biconnected_components(
            adjacency,
            atom,
            graph.atoms.size(),
            visit_order,
            discovery,
            lowlink,
            edge_stack,
            components);
    }

    return components;
}

bool accumulate_mordred_longest_simple_paths(
    const std::vector<std::vector<std::size_t>>& adjacency,
    std::size_t size,
    std::size_t start,
    std::size_t atom,
    std::int64_t distance,
    std::vector<bool>& visited,
    std::vector<std::int64_t>& longest_paths,
    MordredDetourSearchContext& context) {
    for (const auto neighbor : adjacency[atom]) {
        if (visited[neighbor]) {
            continue;
        }
        if (!consume_mordred_detour_search_operation(context)) {
            return false;
        }

        visited[neighbor] = true;
        const auto next_distance = distance + 1;
        auto& longest_path = matrix_entry(longest_paths, size, start, neighbor);
        if (next_distance > longest_path) {
            longest_path = next_distance;
        }
        if (!accumulate_mordred_longest_simple_paths(
            adjacency,
            size,
            start,
            neighbor,
            next_distance,
            visited,
            longest_paths,
            context)) {
            return false;
        }
        visited[neighbor] = false;
    }
    return true;
}

std::optional<MordredDetourBlock> compute_mordred_detour_block(
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    std::size_t atom_count,
    MordredDetourSearchContext& context) {
    MordredDetourBlock block;
    std::vector<bool> in_block(atom_count, false);
    for (const auto [begin, end] : edges) {
        in_block[begin] = true;
        in_block[end] = true;
    }
    for (std::size_t atom = 0u; atom < atom_count; ++atom) {
        if (in_block[atom]) {
            block.nodes.push_back(atom);
        }
    }

    std::vector<std::vector<std::size_t>> adjacency(atom_count);
    for (const auto [begin, end] : edges) {
        adjacency[begin].push_back(end);
        adjacency[end].push_back(begin);
    }

    block.longest_paths.assign(atom_count * atom_count, -1);
    for (const auto atom : block.nodes) {
        matrix_entry(block.longest_paths, atom_count, atom, atom) = 0;
        std::vector<bool> visited(atom_count, false);
        visited[atom] = true;
        if (!accumulate_mordred_longest_simple_paths(
            adjacency,
            atom_count,
            atom,
            atom,
            0,
            visited,
            block.longest_paths,
            context)) {
            return std::nullopt;
        }
    }

    for (const auto begin : block.nodes) {
        for (const auto end : block.nodes) {
            const auto longest = std::max(
                matrix_entry(block.longest_paths, atom_count, begin, end),
                matrix_entry(block.longest_paths, atom_count, end, begin));
            matrix_entry(block.longest_paths, atom_count, begin, end) = longest;
            matrix_entry(block.longest_paths, atom_count, end, begin) = longest;
        }
    }

    return block;
}

std::optional<std::vector<double>> compute_mordred_detour_matrix(
    const MordredHeavyAtomGraph& graph,
    std::uint64_t max_search_operations = kDefaultMordredDetourSearchOperations) {
    if (!is_connected_heavy_atom_graph(graph)) {
        return std::nullopt;
    }

    const auto atom_count = graph.atoms.size();
    if (atom_count == 1u) {
        return std::vector<double>{0.0};
    }

    auto edge_components = mordred_biconnected_edge_components(graph);
    if (edge_components.empty()) {
        return std::nullopt;
    }

    std::vector<MordredDetourBlock> queue;
    queue.reserve(edge_components.size());
    MordredDetourSearchContext search_context{max_search_operations, 0u};
    for (const auto& component : edge_components) {
        auto block = compute_mordred_detour_block(component, atom_count, search_context);
        if (!block.has_value()) {
            return std::nullopt;
        }
        queue.push_back(std::move(*block));
    }

    auto current = queue.back();
    queue.pop_back();
    std::vector<bool> active(atom_count, false);
    for (const auto atom : current.nodes) {
        active[atom] = true;
    }

    while (!queue.empty()) {
        auto merge_block = queue.end();
        std::size_t common_atom = atom_count;
        for (auto iter = queue.end(); iter != queue.begin();) {
            --iter;
            std::size_t common_count = 0u;
            std::size_t candidate_common = atom_count;
            for (const auto atom : iter->nodes) {
                if (active[atom]) {
                    ++common_count;
                    candidate_common = atom;
                }
            }
            if (common_count == 1u) {
                merge_block = iter;
                common_atom = candidate_common;
                break;
            }
        }
        if (merge_block == queue.end()) {
            return std::nullopt;
        }

        auto block = std::move(*merge_block);
        queue.erase(merge_block);
        std::vector<bool> merged_active = active;
        for (const auto atom : block.nodes) {
            merged_active[atom] = true;
        }

        std::vector<std::int64_t> merged(atom_count * atom_count, -1);
        for (std::size_t begin = 0u; begin < atom_count; ++begin) {
            if (!merged_active[begin]) {
                continue;
            }
            for (std::size_t end = begin; end < atom_count; ++end) {
                if (!merged_active[end]) {
                    continue;
                }

                const auto current_path =
                    matrix_entry(current.longest_paths, atom_count, begin, end);
                const auto block_path =
                    matrix_entry(block.longest_paths, atom_count, begin, end);
                std::int64_t detour = -1;
                if (current_path >= 0) {
                    detour = current_path;
                } else if (block_path >= 0) {
                    detour = block_path;
                } else if (begin == common_atom && end == common_atom) {
                    detour = std::max(current_path, block_path);
                } else {
                    const auto begin_to_common =
                        matrix_entry(current.longest_paths, atom_count, begin, common_atom);
                    const auto end_to_common =
                        matrix_entry(block.longest_paths, atom_count, end, common_atom);
                    const auto reverse_begin_to_common =
                        matrix_entry(block.longest_paths, atom_count, begin, common_atom);
                    const auto reverse_end_to_common =
                        matrix_entry(current.longest_paths, atom_count, end, common_atom);

                    if (begin_to_common >= 0 && end_to_common >= 0) {
                        detour = begin_to_common + end_to_common;
                    } else if (reverse_end_to_common >= 0 && reverse_begin_to_common >= 0) {
                        detour = reverse_end_to_common + reverse_begin_to_common;
                    }
                }

                if (detour < 0) {
                    return std::nullopt;
                }
                matrix_entry(merged, atom_count, begin, end) = detour;
                matrix_entry(merged, atom_count, end, begin) = detour;
            }
        }

        current.nodes.clear();
        for (std::size_t atom = 0u; atom < atom_count; ++atom) {
            if (merged_active[atom]) {
                current.nodes.push_back(atom);
            }
        }
        current.longest_paths = std::move(merged);
        active = std::move(merged_active);
    }

    std::vector<double> detour_matrix;
    detour_matrix.reserve(atom_count * atom_count);
    for (const auto distance : current.longest_paths) {
        if (distance < 0) {
            return std::nullopt;
        }
        detour_matrix.push_back(static_cast<double>(distance));
    }
    return detour_matrix;
}

std::optional<MordredDetourMatrixValues> compute_detour_matrix_values(
    const MordredHeavyAtomGraph& graph,
    std::uint64_t max_search_operations = kDefaultMordredDetourSearchOperations) {
    auto detour_matrix = compute_mordred_detour_matrix(graph, max_search_operations);
    if (!detour_matrix.has_value()) {
        return std::nullopt;
    }

    MordredDetourMatrixValues values;
    for (const auto distance : *detour_matrix) {
        values.detour_index += static_cast<std::int64_t>(distance);
    }
    values.detour_index /= 2;

    auto matrix_values = compute_matrix_eigenvalue_values(
        std::move(*detour_matrix),
        graph.atoms.size(),
        graph.bonds);
    if (!matrix_values.has_value()) {
        return std::nullopt;
    }
    values.matrix = *matrix_values;
    return values;
}

std::optional<double> mordred_atomic_number_property(std::uint32_t atomic_number) {
    return static_cast<double>(atomic_number);
}

std::optional<double> mordred_autocorrelation_valence_electrons(
    const MordredAutocorrelationAtomSnapshot& atom) {
    if (atom.atomic_number == 1u) {
        return 0.0;
    }

    const auto outer_electrons = mordred_outer_electrons(atom.atomic_number);
    if (!outer_electrons.has_value()) {
        return std::nullopt;
    }

    const auto zv = static_cast<double>(*outer_electrons - atom.formal_charge);
    const auto z = static_cast<double>(static_cast<int>(atom.atomic_number) - atom.formal_charge);
    const auto denominator = z - zv - 1.0;
    if (denominator == 0.0) {
        return std::nullopt;
    }
    // OpenEye includes explicit H neighbors in GetTotalHCount() after hydrogen expansion.
    return (zv - static_cast<double>(atom.total_hydrogens)) / denominator;
}

std::optional<double> mordred_autocorrelation_sigma_electrons(
    const MordredAutocorrelationAtomSnapshot& atom) {
    return static_cast<double>(atom.non_hydrogen_neighbors);
}

std::optional<double> mordred_autocorrelation_intrinsic_state(
    const MordredAutocorrelationAtomSnapshot& atom) {
    const auto sigma_electrons = mordred_autocorrelation_sigma_electrons(atom);
    if (!sigma_electrons.has_value() || *sigma_electrons == 0.0) {
        return std::nullopt;
    }

    const auto valence_electrons = mordred_autocorrelation_valence_electrons(atom);
    if (!valence_electrons.has_value()) {
        return std::nullopt;
    }

    const auto period = static_cast<double>(mordred_principal_quantum_number(atom.atomic_number));
    return (std::pow(2.0 / period, 2.0) * *valence_electrons + 1.0) / *sigma_electrons;
}

const std::array<MordredAutocorrelationProperty, 12>& mordred_autocorrelation_properties() {
    static const std::array<MordredAutocorrelationProperty, 12> properties{{
        {"Z", mordred_atomic_number_property, nullptr},
        {"m", mordred_mass, nullptr},
        {"v", mordred_vdw_volume, nullptr},
        {"se", mordred_sanderson, nullptr},
        {"pe", mordred_pauling, nullptr},
        {"are", mordred_allred_rocow, nullptr},
        {"p", mordred_polarizability94, nullptr},
        {"i", mordred_ionization_potential, nullptr},
        {"dv", nullptr, mordred_autocorrelation_valence_electrons},
        {"d", nullptr, mordred_autocorrelation_sigma_electrons},
        {"s", nullptr, mordred_autocorrelation_intrinsic_state},
        {"c", nullptr, nullptr, true},
    }};
    return properties;
}

const std::array<MordredBaryszMatrixProperty, 8>& mordred_barysz_matrix_properties() {
    static const std::array<MordredBaryszMatrixProperty, 8> properties{{
        {"Z", 6.0, mordred_atomic_number_property},
        {"m", 12.011, mordred_mass},
        {"v", 20.579526276115534, mordred_vdw_volume},
        {"se", 2.746, mordred_sanderson},
        {"pe", 2.55, mordred_pauling},
        {"are", 2.5, mordred_allred_rocow},
        {"p", 1.67, mordred_polarizability94},
        {"i", 11.2603, mordred_ionization_potential},
    }};
    return properties;
}

const std::array<MordredMoRSEProperty, 5>& mordred_morse_properties() {
    static const std::array<MordredMoRSEProperty, 5> properties{{
        {"", 1.0, nullptr},
        {"m", 12.011, mordred_mass},
        {"v", 20.579526276115534, mordred_vdw_volume},
        {"se", 2.746, mordred_sanderson},
        {"p", 1.67, mordred_polarizability94},
    }};
    return properties;
}

const std::array<MordredBCUTProperty, 12>& mordred_bcut_properties() {
    static const std::array<MordredBCUTProperty, 12> properties{{
        {"c", MordredBCUTPropertyKind::GasteigerCharge},
        {"dv", MordredBCUTPropertyKind::ValenceElectrons},
        {"d", MordredBCUTPropertyKind::SigmaElectrons},
        {"s", MordredBCUTPropertyKind::IntrinsicState},
        {"Z", MordredBCUTPropertyKind::AtomicNumber},
        {"m", MordredBCUTPropertyKind::Mass},
        {"v", MordredBCUTPropertyKind::VdwVolume},
        {"se", MordredBCUTPropertyKind::Sanderson},
        {"pe", MordredBCUTPropertyKind::Pauling},
        {"are", MordredBCUTPropertyKind::AllredRocow},
        {"p", MordredBCUTPropertyKind::Polarizability},
        {"i", MordredBCUTPropertyKind::IonizationPotential},
    }};
    return properties;
}

std::optional<MordredMatrixEigenvalueValues> compute_barysz_matrix_values(
    const MordredHeavyAtomGraph& graph,
    MordredAtomicPropertyLookup property_lookup,
    double carbon_reference) {
    if (!is_connected_heavy_atom_graph(graph)) {
        return std::nullopt;
    }

    const auto atom_count = graph.atoms.size();
    std::vector<double> atom_properties;
    atom_properties.reserve(atom_count);
    for (const auto* atom : graph.atoms) {
        const auto atom_property =
            property_lookup(static_cast<std::uint32_t>(atom->GetAtomicNum()));
        if (!atom_property.has_value() || !std::isfinite(*atom_property)
            || *atom_property == 0.0) {
            return std::nullopt;
        }
        atom_properties.push_back(*atom_property);
    }

    std::vector<double> matrix(
        atom_count * atom_count,
        static_cast<double>(kMordredDisconnectedDistance));

    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        matrix[atom_index * atom_count + atom_index] = 0.0;
        const auto atom_property = atom_properties[atom_index];
        for (const auto& neighbor : graph.adjacency[atom_index]) {
            if (!std::isfinite(neighbor.bond_order) || neighbor.bond_order <= 0.0) {
                return std::nullopt;
            }
            const auto neighbor_property = atom_properties[neighbor.atom_index];
            const auto edge_weight = carbon_reference * carbon_reference
                / (atom_property * neighbor_property * neighbor.bond_order);
            if (!std::isfinite(edge_weight)) {
                return std::nullopt;
            }
            auto& distance = matrix[atom_index * atom_count + neighbor.atom_index];
            distance = std::min(distance, edge_weight);
        }
    }

    for (std::size_t via = 0u; via < atom_count; ++via) {
        for (std::size_t begin = 0u; begin < atom_count; ++begin) {
            for (std::size_t end = 0u; end < atom_count; ++end) {
                const auto through_via =
                    matrix[begin * atom_count + via] + matrix[via * atom_count + end];
                auto& distance = matrix[begin * atom_count + end];
                if (through_via < distance) {
                    distance = through_via;
                }
            }
        }
    }

    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        matrix[atom_index * atom_count + atom_index] =
            1.0 - carbon_reference / atom_properties[atom_index];
    }

    return compute_matrix_eigenvalue_values(std::move(matrix), atom_count, graph.bonds);
}

std::optional<double> mordred_bcut_valence_electrons(const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    if (atomic_number == 1u) {
        return 0.0;
    }

    const auto outer_electrons = mordred_outer_electrons(atomic_number);
    if (!outer_electrons.has_value()) {
        return std::nullopt;
    }

    const auto formal_charge = atom.GetFormalCharge();
    const auto zv = static_cast<double>(*outer_electrons - formal_charge);
    const auto z = static_cast<double>(static_cast<int>(atomic_number) - formal_charge);
    const auto hydrogens = static_cast<double>(atom.GetTotalHCount());
    const auto denominator = z - zv - 1.0;
    if (denominator == 0.0) {
        return std::nullopt;
    }
    return (zv - hydrogens) / denominator;
}

double mordred_bcut_sigma_electrons(const OEChem::OEAtomBase& atom) {
    double sigma_electrons = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> neighbor = atom.GetAtoms(); neighbor; ++neighbor) {
        if (!is_hydrogen(*neighbor)) {
            sigma_electrons += 1.0;
        }
    }
    return sigma_electrons;
}

std::optional<double> mordred_bcut_intrinsic_state(const OEChem::OEAtomBase& atom) {
    const auto sigma_electrons = mordred_bcut_sigma_electrons(atom);
    if (sigma_electrons == 0.0) {
        return std::nullopt;
    }

    const auto valence_electrons = mordred_bcut_valence_electrons(atom);
    if (!valence_electrons.has_value()) {
        return std::nullopt;
    }

    const auto period = static_cast<double>(
        mordred_principal_quantum_number(static_cast<std::uint32_t>(atom.GetAtomicNum())));
    return (std::pow(2.0 / period, 2.0) * *valence_electrons + 1.0) / sigma_electrons;
}

bool has_mordred_gasteiger_parameters(std::uint32_t atomic_number) {
    return std::string(gasteiger_element_symbol(atomic_number)) != "X";
}

std::optional<std::vector<double>> compute_bcut_gasteiger_charge_values(
    const OEChem::OEMolBase& mol,
    const MordredHeavyAtomGraph& graph) {
    for (const auto* atom : graph.atoms) {
        if (atom == nullptr
            || !has_mordred_gasteiger_parameters(
                static_cast<std::uint32_t>(atom->GetAtomicNum()))) {
            return std::nullopt;
        }
    }

    const auto charges = compute_gasteiger_atom_charges(mol);
    if (charges.charges.size() != charges.hydrogen_charges.size()
        || charges.charges.size() != charges.atom_ids.size()) {
        return std::nullopt;
    }

    std::unordered_map<unsigned int, std::size_t> charge_index_by_atom_id;
    charge_index_by_atom_id.reserve(charges.atom_ids.size());
    for (std::size_t index = 0u; index < charges.atom_ids.size(); ++index) {
        charge_index_by_atom_id.emplace(charges.atom_ids[index], index);
    }

    std::vector<double> values;
    values.reserve(graph.atoms.size());
    for (const auto* atom : graph.atoms) {
        const auto found = charge_index_by_atom_id.find(atom->GetIdx());
        if (found == charge_index_by_atom_id.end()) {
            return std::nullopt;
        }

        const auto charge_index = found->second;
        const auto value = charges.charges[charge_index] + charges.hydrogen_charges[charge_index];
        if (!std::isfinite(value)) {
            return std::nullopt;
        }
        values.push_back(value);
    }
    return values;
}

std::optional<double> mordred_bcut_atom_property(
    const OEChem::OEAtomBase& atom,
    MordredBCUTPropertyKind property_kind) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    switch (property_kind) {
    case MordredBCUTPropertyKind::ValenceElectrons:
        return mordred_bcut_valence_electrons(atom);
    case MordredBCUTPropertyKind::SigmaElectrons:
        return mordred_bcut_sigma_electrons(atom);
    case MordredBCUTPropertyKind::IntrinsicState:
        return mordred_bcut_intrinsic_state(atom);
    case MordredBCUTPropertyKind::AtomicNumber:
        return static_cast<double>(atomic_number);
    case MordredBCUTPropertyKind::Mass:
        return mordred_mass(atomic_number);
    case MordredBCUTPropertyKind::VdwVolume:
        return mordred_vdw_volume(atomic_number);
    case MordredBCUTPropertyKind::Sanderson:
        return mordred_sanderson(atomic_number);
    case MordredBCUTPropertyKind::Pauling:
        return mordred_pauling(atomic_number);
    case MordredBCUTPropertyKind::AllredRocow:
        return mordred_allred_rocow(atomic_number);
    case MordredBCUTPropertyKind::Polarizability:
        return mordred_polarizability94(atomic_number);
    case MordredBCUTPropertyKind::IonizationPotential:
        return mordred_ionization_potential(atomic_number);
    case MordredBCUTPropertyKind::GasteigerCharge:
        break;
    }
    return std::nullopt;
}

std::optional<std::vector<double>> compute_bcut_atom_property_values(
    const OEChem::OEMolBase& mol,
    const MordredHeavyAtomGraph& graph,
    MordredBCUTPropertyKind property_kind) {
    if (property_kind == MordredBCUTPropertyKind::GasteigerCharge) {
        return compute_bcut_gasteiger_charge_values(mol, graph);
    }

    std::vector<double> values;
    values.reserve(graph.atoms.size());
    for (const auto* atom : graph.atoms) {
        const auto value = mordred_bcut_atom_property(*atom, property_kind);
        if (!value.has_value() || !std::isfinite(*value)) {
            return std::nullopt;
        }
        values.push_back(*value);
    }
    return values;
}

MordredBCUTValues compute_bcut_property_values(
    const OEChem::OEMolBase& mol,
    const MordredHeavyAtomGraph& graph,
    const MordredBCUTProperty& property) {
    MordredBCUTValues values;
    values.suffix = property.suffix;
    if (!is_connected_heavy_atom_graph(graph)) {
        return values;
    }

    const auto atom_properties = compute_bcut_atom_property_values(mol, graph, property.kind);
    if (!atom_properties.has_value()) {
        return values;
    }

    const auto atom_count = graph.atoms.size();
    std::vector<double> matrix(atom_count * atom_count, 0.001);
    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        matrix[atom_index * atom_count + atom_index] = (*atom_properties)[atom_index];
    }

    for (std::size_t atom_index = 0u; atom_index < graph.adjacency.size(); ++atom_index) {
        for (const auto& neighbor : graph.adjacency[atom_index]) {
            if (atom_index >= neighbor.atom_index) {
                continue;
            }
            auto weight = neighbor.bond_order / 10.0;
            if (graph.adjacency[atom_index].size() == 1u
                || graph.adjacency[neighbor.atom_index].size() == 1u) {
                weight += 0.01;
            }
            matrix[atom_index * atom_count + neighbor.atom_index] = weight;
            matrix[neighbor.atom_index * atom_count + atom_index] = weight;
        }
    }

    const auto eigensystem = symmetric_eigensystem_jacobi(std::move(matrix), atom_count);
    if (!eigensystem.has_value() || eigensystem->eigenvalues.empty()) {
        return values;
    }

    const auto [low, high] =
        std::minmax_element(eigensystem->eigenvalues.begin(), eigensystem->eigenvalues.end());
    values.high = *high;
    values.low = *low;
    return values;
}

std::vector<MordredBCUTValues> compute_bcut_values(
    const OEChem::OEMolBase& mol,
    const MordredHeavyAtomGraph& graph) {
    std::vector<MordredBCUTValues> values;
    values.reserve(mordred_bcut_properties().size());
    for (const auto& property : mordred_bcut_properties()) {
        values.push_back(compute_bcut_property_values(mol, graph, property));
    }
    return values;
}

std::optional<double> compute_vertex_adjacency_information(const MordredHeavyAtomGraph& graph) {
    const auto heavy_heavy_bonds = graph.bonds.size();
    if (heavy_heavy_bonds == 0u) {
        return std::nullopt;
    }
    return 1.0 + std::log2(static_cast<double>(heavy_heavy_bonds));
}

std::optional<double> compute_balaban_j(
    const MordredHeavyAtomGraph& graph,
    const std::vector<std::vector<std::int64_t>>& distances) {
    const auto atom_count = graph.atoms.size();
    if (atom_count == 0u) {
        return std::nullopt;
    }

    std::vector<double> distance_row_sums;
    distance_row_sums.reserve(atom_count);
    for (const auto& row : distances) {
        double sum = 0.0;
        for (const auto distance : row) {
            sum += static_cast<double>(distance);
        }
        distance_row_sums.push_back(sum);
    }

    double edge_sum = 0.0;
    for (const auto [begin, end] : graph.bonds) {
        edge_sum +=
            1.0 / std::sqrt(distance_row_sums[begin] * distance_row_sums[end]);
    }

    const auto bond_count = graph.bonds.size();
    const auto mu = static_cast<std::int64_t>(bond_count)
                    - static_cast<std::int64_t>(atom_count) + 1;
    if (mu + 1 == 0) {
        return 0.0;
    }
    return static_cast<double>(bond_count) / static_cast<double>(mu + 1) * edge_sum;
}

std::string bertz_distance_key(double distance) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(4) << distance;
    return stream.str();
}

std::vector<std::uint32_t> assign_bertz_symmetry_classes(
    const std::vector<std::vector<double>>& distances) {
    constexpr std::size_t kCutoff = 100u;

    std::vector<std::vector<std::string>> keys_seen;
    std::vector<std::uint32_t> symmetry_classes;
    symmetry_classes.reserve(distances.size());

    for (const auto& row : distances) {
        auto sorted_row = row;
        std::sort(sorted_row.begin(), sorted_row.end());
        if (sorted_row.size() > kCutoff) {
            sorted_row.resize(kCutoff);
        }

        std::vector<std::string> key;
        key.reserve(sorted_row.size());
        for (const auto distance : sorted_row) {
            key.push_back(bertz_distance_key(distance));
        }

        auto found = std::find(keys_seen.begin(), keys_seen.end(), key);
        if (found == keys_seen.end()) {
            keys_seen.push_back(std::move(key));
            symmetry_classes.push_back(static_cast<std::uint32_t>(keys_seen.size()));
        } else {
            symmetry_classes.push_back(
                static_cast<std::uint32_t>(std::distance(keys_seen.begin(), found) + 1));
        }
    }

    return symmetry_classes;
}

std::vector<std::vector<double>> compute_mordred_bond_order_distances(
    const MordredHeavyAtomGraph& graph) {
    const auto atom_count = graph.adjacency.size();
    std::vector<std::vector<double>> distances(
        atom_count,
        std::vector<double>(atom_count, static_cast<double>(kMordredDisconnectedDistance)));

    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        distances[atom_index][atom_index] = 0.0;
        for (const auto& neighbor : graph.adjacency[atom_index]) {
            const auto edge_distance = 1.0 / neighbor.bond_order;
            distances[atom_index][neighbor.atom_index] =
                std::min(distances[atom_index][neighbor.atom_index], edge_distance);
        }
    }

    for (std::size_t via = 0u; via < atom_count; ++via) {
        for (std::size_t begin = 0u; begin < atom_count; ++begin) {
            for (std::size_t end = 0u; end < atom_count; ++end) {
                const auto through_via = distances[begin][via] + distances[via][end];
                if (through_via < distances[begin][end]) {
                    distances[begin][end] = through_via;
                }
            }
        }
    }

    return distances;
}

double info_entropy(const std::vector<double>& counts) {
    double total = 0.0;
    for (const auto count : counts) {
        total += count;
    }
    if (total == 0.0) {
        return 0.0;
    }

    double entropy = 0.0;
    for (const auto count : counts) {
        const auto probability = count / total;
        if (probability != 0.0) {
            entropy += -probability * std::log2(probability);
        }
    }
    return entropy;
}

double compute_bertz_ct(const MordredHeavyAtomGraph& graph) {
    const auto atom_count = graph.atoms.size();
    if (atom_count < 2u) {
        return 0.0;
    }

    auto sorted_adjacency = graph.adjacency;
    for (auto& neighbors : sorted_adjacency) {
        std::sort(
            neighbors.begin(),
            neighbors.end(),
            [](const PathCountNeighbor& left, const PathCountNeighbor& right) {
                return left.atom_index < right.atom_index;
            });
    }

    // RDKit BertzCT keeps forceDMat=1, so symmetry classes use the
    // bond-order "Balaban" matrix even when Mordred passes DistanceMatrix(False).
    const auto symmetry_distances = compute_mordred_bond_order_distances(graph);
    const auto symmetry_classes = assign_bertz_symmetry_classes(symmetry_distances);
    std::map<unsigned int, double> atom_type_counts;
    std::map<std::vector<std::uint32_t>, double> connection_counts;

    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        atom_type_counts[graph.atoms[atom_index]->GetAtomicNum()] += 1.0;
        const auto hinge_class = symmetry_classes[atom_index];
        const auto& neighbors = sorted_adjacency[atom_index];

        for (std::size_t left = 0u; left < neighbors.size(); ++left) {
            const auto left_atom_index = neighbors[left].atom_index;
            const auto left_class = symmetry_classes[left_atom_index];
            const auto left_bond_order = neighbors[left].bond_order;

            if (left_bond_order > 1.0 && left_atom_index > atom_index) {
                const auto connection_count =
                    left_bond_order * (left_bond_order - 1.0) / 2.0;
                const auto lower_class = std::min(hinge_class, left_class);
                const auto upper_class = std::max(hinge_class, left_class);
                connection_counts[{lower_class, upper_class}] += connection_count;
            }

            for (std::size_t right = left + 1u; right < neighbors.size(); ++right) {
                const auto right_atom_index = neighbors[right].atom_index;
                const auto right_class = symmetry_classes[right_atom_index];
                const auto right_bond_order = neighbors[right].bond_order;
                const auto lower_class = std::min(left_class, right_class);
                const auto upper_class = std::max(left_class, right_class);
                connection_counts[{lower_class, hinge_class, upper_class}] +=
                    left_bond_order * right_bond_order;
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
        total_connections * (info_entropy(connection_values) + std::log2(total_connections));
    const auto atom_type_ie = static_cast<double>(atom_count) * info_entropy(atom_type_values);
    return connection_ie + atom_type_ie;
}

std::vector<std::vector<std::int64_t>> compute_mordred_heavy_atom_distances(
    const MordredHeavyAtomGraph& graph) {
    const auto atom_count = graph.adjacency.size();
    std::vector<std::vector<std::int64_t>> distances(
        atom_count,
        std::vector<std::int64_t>(atom_count, kMordredDisconnectedDistance));

    for (std::size_t start = 0u; start < atom_count; ++start) {
        std::vector<std::size_t> queue;
        queue.reserve(atom_count);
        distances[start][start] = 0;
        queue.push_back(start);

        for (std::size_t head = 0u; head < queue.size(); ++head) {
            const auto current = queue[head];
            for (const auto neighbor : graph.adjacency[current]) {
                if (distances[start][neighbor.atom_index] != kMordredDisconnectedDistance) {
                    continue;
                }
                distances[start][neighbor.atom_index] = distances[start][current] + 1;
                queue.push_back(neighbor.atom_index);
            }
        }
    }

    return distances;
}

std::optional<std::vector<double>> compute_explicit_gasteiger_charge_values(
    const OEChem::OEMolBase& explicit_mol,
    const std::vector<unsigned int>& atom_ids) {
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = explicit_mol.GetAtoms(); atom; ++atom) {
        if (!has_mordred_gasteiger_parameters(
                static_cast<std::uint32_t>(atom->GetAtomicNum()))) {
            return std::nullopt;
        }
    }

    const auto charges = compute_gasteiger_atom_charges(explicit_mol, false);
    if (charges.charges.size() != charges.hydrogen_charges.size()
        || charges.charges.size() != charges.atom_ids.size()) {
        return std::nullopt;
    }

    std::unordered_map<unsigned int, std::size_t> charge_index_by_atom_id;
    charge_index_by_atom_id.reserve(charges.atom_ids.size());
    for (std::size_t index = 0u; index < charges.atom_ids.size(); ++index) {
        charge_index_by_atom_id.emplace(charges.atom_ids[index], index);
    }

    std::vector<double> values;
    values.reserve(atom_ids.size());
    for (const auto atom_id : atom_ids) {
        const auto found = charge_index_by_atom_id.find(atom_id);
        if (found == charge_index_by_atom_id.end()) {
            return std::nullopt;
        }

        const auto charge_index = found->second;
        const auto value = charges.charges[charge_index] + charges.hydrogen_charges[charge_index];
        if (!std::isfinite(value)) {
            return std::nullopt;
        }
        values.push_back(value);
    }
    return values;
}

std::optional<double> compute_cpsa_relative_charge(
    const std::vector<double>& charges,
    bool positive) {
    std::optional<double> selected_charge;
    double selected_abs_charge = 0.0;
    double charge_sum = 0.0;

    for (const auto charge : charges) {
        if (positive ? charge <= 0.0 : charge >= 0.0) {
            continue;
        }

        const auto abs_charge = std::abs(charge);
        if (!selected_charge.has_value() || abs_charge > selected_abs_charge) {
            selected_charge = charge;
            selected_abs_charge = abs_charge;
        }
        charge_sum += charge;
    }

    if (!selected_charge.has_value()) {
        return 0.0;
    }
    if (charge_sum == 0.0) {
        return std::nullopt;
    }
    return *selected_charge / charge_sum;
}

std::optional<MordredCPSAChargeValues> compute_cpsa_charge_values(
    const OEChem::OEMolBase& mol) {
    auto explicit_mol = explicit_hydrogen_copy(mol);
    std::vector<unsigned int> atom_ids;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = explicit_mol.GetAtoms(); atom; ++atom) {
        atom_ids.push_back(atom->GetIdx());
    }

    const auto charges = compute_explicit_gasteiger_charge_values(explicit_mol, atom_ids);
    if (!charges.has_value()) {
        return std::nullopt;
    }

    const auto rncg = compute_cpsa_relative_charge(*charges, false);
    const auto rpcg = compute_cpsa_relative_charge(*charges, true);
    if (!rncg.has_value() || !rpcg.has_value()) {
        return std::nullopt;
    }
    return MordredCPSAChargeValues{*rncg, *rpcg};
}

MordredExplicitAtomDistanceValues compute_mordred_explicit_atom_distances(
    const OEChem::OEMolBase& mol) {
    MordredExplicitAtomDistanceValues values;
    // Mordred Autocorrelation opts into explicit hydrogens, unlike most graph descriptors here.
    auto explicit_mol = explicit_hydrogen_copy(mol);
    std::vector<const OEChem::OEAtomBase*> atoms;
    std::unordered_map<unsigned int, std::size_t> atom_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = explicit_mol.GetAtoms(); atom; ++atom) {
        atom_indices.emplace(atom->GetIdx(), atoms.size());
        MordredAutocorrelationAtomSnapshot snapshot;
        snapshot.atomic_number = static_cast<std::uint32_t>(atom->GetAtomicNum());
        snapshot.formal_charge = atom->GetFormalCharge();
        snapshot.total_hydrogens = atom->GetTotalHCount();
        for (OESystem::OEIter<OEChem::OEAtomBase> neighbor = atom->GetAtoms(); neighbor;
             ++neighbor) {
            if (!is_hydrogen(*neighbor)) {
                ++snapshot.non_hydrogen_neighbors;
            }
        }
        values.atoms.push_back(snapshot);
        values.atomic_numbers.push_back(snapshot.atomic_number);
        values.atom_ids.push_back(atom->GetIdx());
        atoms.push_back(&*atom);
    }

    const auto atom_count = atoms.size();
    values.distances = std::vector<std::vector<std::int64_t>>(
        atom_count,
        std::vector<std::int64_t>(atom_count, kMordredDisconnectedDistance));
    for (std::size_t start = 0u; start < atom_count; ++start) {
        std::vector<std::size_t> queue;
        queue.reserve(atom_count);
        values.distances[start][start] = 0;
        queue.push_back(start);

        for (std::size_t head = 0u; head < queue.size(); ++head) {
            const auto current = queue[head];
            for (OESystem::OEIter<OEChem::OEAtomBase> neighbor = atoms[current]->GetAtoms();
                 neighbor;
                 ++neighbor) {
                const auto neighbor_index = atom_indices.find(neighbor->GetIdx());
                if (neighbor_index == atom_indices.end()) {
                    continue;
                }
                if (values.distances[start][neighbor_index->second]
                    != kMordredDisconnectedDistance) {
                    continue;
                }
                values.distances[start][neighbor_index->second] =
                    values.distances[start][current] + 1;
                queue.push_back(neighbor_index->second);
            }
        }
    }

    values.gasteiger_charges =
        compute_explicit_gasteiger_charge_values(explicit_mol, values.atom_ids);
    return values;
}

MordredAutocorrelationValues compute_autocorrelation_values_from_properties(
    const MordredExplicitAtomDistanceValues& distance_values,
    const char* suffix,
    const std::vector<double>& atom_properties,
    bool include_raw_values) {
    MordredAutocorrelationValues values;
    values.suffix = suffix;

    const auto atom_count = distance_values.atomic_numbers.size();
    if (atom_properties.size() != atom_count) {
        return values;
    }
    for (const auto atom_property : atom_properties) {
        if (!std::isfinite(atom_property)) {
            return values;
        }
    }

    std::array<double, 9> ats{};
    std::array<double, 9> atsc{};
    std::array<double, 9> squared_differences{};
    std::array<std::size_t, 9> pair_counts{};
    double atom_property_sum = 0.0;
    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        const auto atom_property = atom_properties[atom_index];
        if (include_raw_values) {
            ats[0] += atom_property * atom_property;
        }
        atom_property_sum += atom_property;
    }
    pair_counts[0] = atom_count;

    std::vector<double> centered_properties;
    centered_properties.reserve(atom_count);
    const auto atom_property_mean =
        atom_count == 0u ? 0.0 : atom_property_sum / static_cast<double>(atom_count);
    for (const auto atom_property : atom_properties) {
        const auto centered_property = atom_property - atom_property_mean;
        centered_properties.push_back(centered_property);
        atsc[0] += centered_property * centered_property;
    }

    for (std::size_t left = 0u; left < atom_count; ++left) {
        const auto left_property = atom_properties[left];
        const auto left_centered_property = centered_properties[left];
        for (std::size_t right = left + 1u; right < atom_count; ++right) {
            const auto distance = distance_values.distances[left][right];
            if (distance <= 0 || distance > 8) {
                continue;
            }
            const auto lag = static_cast<std::size_t>(distance);
            if (include_raw_values) {
                ats[lag] += left_property * atom_properties[right];
            }
            const auto property_difference = left_property - atom_properties[right];
            squared_differences[lag] += property_difference * property_difference;
            atsc[lag] += left_centered_property * centered_properties[right];
            ++pair_counts[lag];
        }
    }

    const auto centered_sum_squares = atsc[0];
    for (std::size_t lag = 0u; lag < values.aats.size(); ++lag) {
        if (include_raw_values) {
            values.ats[lag] = ats[lag];
        }
        values.atsc[lag] = atsc[lag];
        if (pair_counts[lag] != 0u) {
            if (include_raw_values) {
                values.aats[lag] = ats[lag] / static_cast<double>(pair_counts[lag]);
            }
            values.aatsc[lag] = atsc[lag] / static_cast<double>(pair_counts[lag]);
        }
    }
    if (centered_sum_squares == 0.0) {
        return values;
    }

    for (std::size_t lag = 1u; lag < values.mats.size(); ++lag) {
        if (!values.aatsc[lag].has_value()) {
            continue;
        }
        values.mats[lag] =
            static_cast<double>(atom_count) * *values.aatsc[lag] / centered_sum_squares;

        if (atom_count <= 1u || pair_counts[lag] == 0u) {
            continue;
        }
        // Mordred's GMat contains both directions for k > 0, while GSum is the
        // unordered pair count; this is the reduced unordered-pair form.
        const auto numerator =
            squared_differences[lag] / (2.0 * static_cast<double>(pair_counts[lag]));
        const auto denominator =
            centered_sum_squares / (static_cast<double>(atom_count) - 1.0);
        if (denominator != 0.0) {
            values.gats[lag] = numerator / denominator;
        }
    }

    return values;
}

MordredAutocorrelationValues compute_autocorrelation_values(
    const MordredExplicitAtomDistanceValues& distance_values,
    const MordredAutocorrelationProperty& property) {
    MordredAutocorrelationValues values;
    values.suffix = property.suffix;

    if (property.gasteiger_charge) {
        if (!distance_values.gasteiger_charges.has_value()) {
            return values;
        }
        return compute_autocorrelation_values_from_properties(
            distance_values, property.suffix, *distance_values.gasteiger_charges, false);
    }

    const auto atom_count = distance_values.atomic_numbers.size();
    std::vector<double> atom_properties;
    atom_properties.reserve(atom_count);
    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        const auto atom_property =
            property.atom_lookup != nullptr
                ? property.atom_lookup(distance_values.atoms[atom_index])
                : property.atomic_number_lookup(distance_values.atomic_numbers[atom_index]);
        if (!atom_property.has_value()) {
            return values;
        }
        atom_properties.push_back(*atom_property);
    }

    return compute_autocorrelation_values_from_properties(
        distance_values, property.suffix, atom_properties, true);
}

std::vector<MordredAutocorrelationValues> compute_autocorrelation_values(
    const OEChem::OEMolBase& mol) {
    const auto distance_values = compute_mordred_explicit_atom_distances(mol);
    std::vector<MordredAutocorrelationValues> values;
    values.reserve(mordred_autocorrelation_properties().size());
    for (const auto& property : mordred_autocorrelation_properties()) {
        values.push_back(compute_autocorrelation_values(distance_values, property));
    }
    return values;
}

MordredTopologicalChargeValues compute_topological_charge_values(
    const MordredHeavyAtomGraph& graph,
    const std::vector<std::vector<std::int64_t>>& distances) {
    MordredTopologicalChargeValues values;
    const auto atom_count = graph.atoms.size();
    if (atom_count < 2u) {
        return values;
    }

    std::vector<std::vector<double>> inverse_squared_distances(
        atom_count,
        std::vector<double>(atom_count, 0.0));
    for (std::size_t row = 0u; row < atom_count; ++row) {
        for (std::size_t column = 0u; column < atom_count; ++column) {
            const auto distance = distances[row][column];
            if (distance == 0) {
                continue;
            }
            const auto distance_value = static_cast<double>(distance);
            inverse_squared_distances[row][column] = 1.0 / (distance_value * distance_value);
        }
    }

    std::vector<std::vector<double>> charge_terms(atom_count, std::vector<double>(atom_count, 0.0));
    for (std::size_t row = 0u; row < atom_count; ++row) {
        for (std::size_t column = 0u; column < atom_count; ++column) {
            double forward = 0.0;
            for (const auto& neighbor : graph.adjacency[row]) {
                forward += inverse_squared_distances[neighbor.atom_index][column];
            }

            double reverse = 0.0;
            for (const auto& neighbor : graph.adjacency[column]) {
                reverse += inverse_squared_distances[neighbor.atom_index][row];
            }

            charge_terms[row][column] = forward - reverse;
        }
    }

    std::array<std::uint32_t, 11> counts{};
    for (std::size_t row = 1u; row < atom_count; ++row) {
        for (std::size_t column = 0u; column < row; ++column) {
            const auto distance = distances[row][column];
            if (distance < 1 || distance > 10) {
                continue;
            }

            const auto order = static_cast<std::size_t>(distance);
            values.raw[order] += std::abs(charge_terms[row][column]);
            ++counts[order];
        }
    }

    for (std::size_t order = 1u; order <= 10u; ++order) {
        if (counts[order] == 0u) {
            continue;
        }
        values.mean[order] =
            values.raw[order] / static_cast<double>(counts[order]);
        values.global10 += values.mean[order];
    }

    return values;
}

MordredABCIndexValues compute_abc_index_values(
    const MordredHeavyAtomGraph& graph,
    const std::vector<std::vector<std::int64_t>>& distances) {
    MordredABCIndexValues values;
    for (const auto [begin, end] : graph.bonds) {
        const auto begin_degree = static_cast<double>(graph.adjacency[begin].size());
        const auto end_degree = static_cast<double>(graph.adjacency[end].size());
        values.abc +=
            std::sqrt((begin_degree + end_degree - 2.0) / (begin_degree * end_degree));

        std::uint32_t begin_closer = 0u;
        std::uint32_t end_closer = 0u;
        for (std::size_t atom_index = 0u; atom_index < distances.size(); ++atom_index) {
            if (distances[begin][atom_index] < distances[end][atom_index]) {
                ++begin_closer;
            }
            if (distances[end][atom_index] < distances[begin][atom_index]) {
                ++end_closer;
            }
        }

        const auto nu = static_cast<double>(begin_closer);
        const auto nv = static_cast<double>(end_closer);
        values.abcgg += std::sqrt((nu + nv - 2.0) / (nu * nv));
    }
    return values;
}

void compute_molecular_distance_edge_family(
    const MordredHeavyAtomGraph& graph,
    const std::vector<std::vector<std::int64_t>>& distances,
    const int atomic_number,
    const std::size_t max_degree,
    std::array<std::array<std::optional<double>, 5>, 5>& output) {
    for (std::size_t valence1 = 1u; valence1 <= max_degree; ++valence1) {
        for (std::size_t valence2 = valence1; valence2 <= max_degree; ++valence2) {
            std::size_t selected_count = 0u;
            double log_distance_product = 0.0;

            for (std::size_t begin = 0u; begin < graph.atoms.size(); ++begin) {
                const auto* begin_atom = graph.atoms[begin];
                if (begin_atom->GetAtomicNum() != atomic_number) {
                    continue;
                }
                const auto begin_degree = graph.adjacency[begin].size();

                for (std::size_t end = begin + 1u; end < graph.atoms.size(); ++end) {
                    const auto* end_atom = graph.atoms[end];
                    if (end_atom->GetAtomicNum() != atomic_number) {
                        continue;
                    }
                    const auto end_degree = graph.adjacency[end].size();
                    const auto matches_order =
                        begin_degree == valence1 && end_degree == valence2;
                    const auto matches_reverse =
                        begin_degree == valence2 && end_degree == valence1;
                    if (!matches_order && !matches_reverse) {
                        continue;
                    }

                    log_distance_product += std::log(static_cast<double>(distances[begin][end]));
                    ++selected_count;
                }
            }

            if (selected_count == 0u) {
                continue;
            }
            output[valence1][valence2] =
                static_cast<double>(selected_count)
                / std::exp(log_distance_product / static_cast<double>(selected_count));
        }
    }
}

MordredMolecularDistanceEdgeValues compute_molecular_distance_edge_values(
    const MordredHeavyAtomGraph& graph,
    const std::vector<std::vector<std::int64_t>>& distances) {
    MordredMolecularDistanceEdgeValues values;
    compute_molecular_distance_edge_family(graph, distances, 6, 4u, values.carbon);
    compute_molecular_distance_edge_family(graph, distances, 8, 2u, values.oxygen);
    compute_molecular_distance_edge_family(graph, distances, 7, 3u, values.nitrogen);
    return values;
}

MordredZagrebValues compute_zagreb_values(const MordredHeavyAtomGraph& graph) {
    MordredZagrebValues values;
    double modified_zagreb1 = 0.0;
    bool has_zero_degree = false;

    for (const auto& neighbors : graph.adjacency) {
        const auto degree = static_cast<double>(neighbors.size());
        values.zagreb1 += degree * degree;
        if (degree == 0.0) {
            has_zero_degree = true;
        } else {
            modified_zagreb1 += 1.0 / (degree * degree);
        }
    }

    if (!has_zero_degree) {
        values.modified_zagreb1 = modified_zagreb1;
    }

    for (const auto [begin, end] : graph.bonds) {
        const auto begin_degree = static_cast<double>(graph.adjacency[begin].size());
        const auto end_degree = static_cast<double>(graph.adjacency[end].size());
        const auto degree_product = begin_degree * end_degree;
        values.zagreb2 += degree_product;
        values.modified_zagreb2 += 1.0 / degree_product;
    }

    return values;
}

MordredWienerValues compute_wiener_values(
    const std::vector<std::vector<std::int64_t>>& distances) {
    MordredWienerValues values;
    const auto atom_count = distances.size();
    for (std::size_t start = 0u; start < atom_count; ++start) {
        for (std::size_t end = start + 1u; end < atom_count; ++end) {
            const auto distance = distances[start][end];
            values.wpath += distance;
            if (distance == 3) {
                values.wpol += 1;
            }
        }
    }

    return values;
}

MordredTopologicalIndexValues compute_topological_index_values(
    const std::vector<std::vector<std::int64_t>>& distances) {
    MordredTopologicalIndexValues values;
    if (distances.empty()) {
        return values;
    }

    const auto first_eccentricity =
        *std::max_element(distances.front().begin(), distances.front().end());
    auto diameter = first_eccentricity;
    auto radius = first_eccentricity;

    for (const auto& atom_distances : distances) {
        const auto eccentricity =
            *std::max_element(atom_distances.begin(), atom_distances.end());
        diameter = std::max(diameter, eccentricity);
        radius = std::min(radius, eccentricity);
    }

    values.diameter = diameter;
    values.radius = radius;

    const auto diameter_value = static_cast<double>(diameter);
    const auto radius_value = static_cast<double>(radius);
    if (radius != 0) {
        values.topo_shape_index = (diameter_value - radius_value) / radius_value;
    }
    if (diameter != 0) {
        values.petitjean_index = (diameter_value - radius_value) / diameter_value;
    }

    return values;
}

std::int64_t compute_eccentric_connectivity_index(
    const MordredHeavyAtomGraph& graph,
    const std::vector<std::vector<std::int64_t>>& distances) {
    std::int64_t ec_index = 0;
    for (std::size_t atom_index = 0u; atom_index < graph.adjacency.size(); ++atom_index) {
        const auto eccentricity =
            *std::max_element(distances[atom_index].begin(), distances[atom_index].end());
        const auto valence = static_cast<std::int64_t>(graph.adjacency[atom_index].size());
        ec_index += eccentricity * valence;
    }
    return ec_index;
}

SimplePathWalkTotals count_simple_path_walks(
    const std::vector<std::vector<PathCountNeighbor>>& adjacency,
    std::size_t current,
    std::size_t remaining_bonds,
    double path_bond_order_product,
    std::vector<bool>& visited) {
    if (remaining_bonds == 0u) {
        return {1u, path_bond_order_product};
    }

    SimplePathWalkTotals totals;
    for (const auto neighbor : adjacency[current]) {
        if (visited[neighbor.atom_index]) {
            continue;
        }
        visited[neighbor.atom_index] = true;
        const auto child_totals = count_simple_path_walks(
            adjacency,
            neighbor.atom_index,
            remaining_bonds - 1u,
            path_bond_order_product * neighbor.bond_order,
            visited);
        totals.count += child_totals.count;
        totals.pi_sum += child_totals.pi_sum;
        visited[neighbor.atom_index] = false;
    }
    return totals;
}

MordredPathCountValues compute_path_count_values(const MordredHeavyAtomGraph& graph) {
    const auto& adjacency = graph.adjacency;
    MordredPathCountValues values;
    values.total_mpc10 = static_cast<std::uint32_t>(adjacency.size());
    values.total_pi_mpc10 = static_cast<double>(adjacency.size());
    for (std::size_t order = 1u; order <= 10u; ++order) {
        SimplePathWalkTotals oriented_paths;
        std::vector<bool> visited(adjacency.size(), false);
        for (std::size_t start = 0u; start < adjacency.size(); ++start) {
            visited[start] = true;
            const auto start_totals =
                count_simple_path_walks(adjacency, start, order, 1.0, visited);
            oriented_paths.count += start_totals.count;
            oriented_paths.pi_sum += start_totals.pi_sum;
            visited[start] = false;
        }

        // Each accepted RDKit/Mordred bond path is undirected; DFS sees both orientations.
        values.mpc[order] = oriented_paths.count / 2u;
        values.pi_mpc[order] = oriented_paths.pi_sum / 2.0;
        values.total_mpc10 += values.mpc[order];
        values.total_pi_mpc10 += values.pi_mpc[order];
    }
    return values;
}

std::optional<int> mordred_outer_electrons(std::uint32_t atomic_number) {
    // Mordred's dv property uses RDKit GetNOuterElecs; mirror those values
    // directly so valence-weighted Chi does not add an RDKit runtime dependency.
    constexpr std::array<int, 119> outer_electrons{{
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
        return std::nullopt;
    }
    return outer_electrons[atomic_number];
}

std::uint32_t mordred_principal_quantum_number(std::uint32_t atomic_number) {
    if (atomic_number <= 2u) {
        return 1u;
    }
    if (atomic_number <= 10u) {
        return 2u;
    }
    if (atomic_number <= 18u) {
        return 3u;
    }
    if (atomic_number <= 36u) {
        return 4u;
    }
    if (atomic_number <= 54u) {
        return 5u;
    }
    if (atomic_number <= 86u) {
        return 6u;
    }
    return 7u;
}

std::optional<double> eta_core_count(std::uint32_t atomic_number) {
    if (atomic_number == 1u) {
        return 0.0;
    }

    const auto outer_electrons = mordred_outer_electrons(atomic_number);
    if (!outer_electrons.has_value()) {
        return std::nullopt;
    }

    const auto period = mordred_principal_quantum_number(atomic_number);
    if (period <= 1u || *outer_electrons == 0) {
        return std::nullopt;
    }

    return (static_cast<double>(atomic_number) - static_cast<double>(*outer_electrons))
           / (static_cast<double>(*outer_electrons) * static_cast<double>(period - 1u));
}

std::optional<double> eta_epsilon(const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto outer_electrons = mordred_outer_electrons(atomic_number);
    const auto core_count = eta_core_count(atomic_number);
    if (!outer_electrons.has_value() || !core_count.has_value()) {
        return std::nullopt;
    }

    return 0.3 * static_cast<double>(*outer_electrons) - *core_count;
}

bool has_aromatic_bonds(const OEChem::OEMolBase& mol) {
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
        if (bond->IsAromatic()) {
            return true;
        }
    }
    return false;
}

void eta_kekulize_if_aromatic(OEChem::OEMolBase& mol) {
    if (has_aromatic_bonds(mol)) {
        OEChem::OEKekulize(mol);
    }
}

OEChem::OEGraphMol eta_prepared_mol(
    const OEChem::OEMolBase& mol,
    bool explicit_hydrogens) {
    OEChem::OEGraphMol working_mol(mol);
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    OEChem::OESuppressHydrogens(working_mol);
    if (explicit_hydrogens) {
        OEChem::OEAddExplicitHydrogens(working_mol, false, false);
    }
    OEChem::OEFindRingAtomsAndBonds(working_mol);
    OEChem::OEAssignAromaticFlags(working_mol);
    if (!explicit_hydrogens) {
        eta_kekulize_if_aromatic(working_mol);
    }
    return working_mol;
}

std::optional<OEChem::OEGraphMol> eta_altered_mol(
    const OEChem::OEMolBase& mol,
    bool explicit_hydrogens,
    bool saturated) {
    const auto source_mol = eta_prepared_mol(mol, false);
    OEChem::OEGraphMol altered_mol;
    std::unordered_map<unsigned int, OEChem::OEAtomBase*> atoms_by_source_id;

    for (OESystem::OEIter<OEChem::OEAtomBase> atom = source_mol.GetAtoms(); atom; ++atom) {
        if (is_hydrogen(*atom)) {
            continue;
        }

        auto* new_atom = altered_mol.NewAtom(
            saturated ? atom->GetAtomicNum() : static_cast<unsigned int>(6u));
        if (new_atom == nullptr) {
            return std::nullopt;
        }
        if (saturated) {
            new_atom->SetFormalCharge(atom->GetFormalCharge());
        }
        atoms_by_source_id.emplace(atom->GetIdx(), new_atom);
    }

    for (OESystem::OEIter<OEChem::OEBondBase> bond = source_mol.GetBonds(); bond; ++bond) {
        const auto* begin = bond->GetBgn();
        const auto* end = bond->GetEnd();
        if (begin == nullptr || end == nullptr || is_hydrogen(*begin) || is_hydrogen(*end)) {
            continue;
        }
        if (!saturated && (begin->GetDegree() > 4u || end->GetDegree() > 4u)) {
            return std::nullopt;
        }

        const auto new_begin = atoms_by_source_id.find(begin->GetIdx());
        const auto new_end = atoms_by_source_id.find(end->GetIdx());
        if (new_begin == atoms_by_source_id.end() || new_end == atoms_by_source_id.end()) {
            continue;
        }

        const auto order =
            saturated && (begin->GetAtomicNum() != 6u || end->GetAtomicNum() != 6u)
                ? bond->GetOrder()
                : 1u;
        if (altered_mol.NewBond(new_begin->second, new_end->second, order) == nullptr) {
            return std::nullopt;
        }
    }

    OEChem::OEFindRingAtomsAndBonds(altered_mol);
    OEChem::OEAssignAromaticFlags(altered_mol);
    eta_kekulize_if_aromatic(altered_mol);
    OEChem::OEAssignImplicitHydrogens(altered_mol);
    if (explicit_hydrogens) {
        OEChem::OEAddExplicitHydrogens(altered_mol, false, false);
        OEChem::OEFindRingAtomsAndBonds(altered_mol);
        OEChem::OEAssignAromaticFlags(altered_mol);
        eta_kekulize_if_aromatic(altered_mol);
    }

    return altered_mol;
}

MordredEtaGraph build_eta_graph(const OEChem::OEMolBase& mol) {
    MordredEtaGraph graph;
    std::unordered_map<unsigned int, std::size_t> atom_indices;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        atom_indices.emplace(atom->GetIdx(), graph.atoms.size());
        graph.atoms.push_back(&*atom);
    }

    graph.adjacency.resize(graph.atoms.size());
    for (OESystem::OEIter<OEChem::OEBondBase> bond = mol.GetBonds(); bond; ++bond) {
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

        const auto order = static_cast<std::uint32_t>(bond->GetOrder());
        const auto aromatic = bond->IsAromatic();
        graph.adjacency[begin_index->second].push_back({end_index->second, order, aromatic});
        graph.adjacency[end_index->second].push_back({begin_index->second, order, aromatic});
    }
    return graph;
}

bool is_connected_eta_graph(const MordredEtaGraph& graph) {
    const auto atom_count = graph.atoms.size();
    if (atom_count == 0u) {
        return false;
    }

    std::vector<bool> visited(atom_count, false);
    std::vector<std::size_t> queue;
    queue.reserve(atom_count);
    visited.front() = true;
    queue.push_back(0u);

    for (std::size_t head = 0u; head < queue.size(); ++head) {
        const auto current = queue[head];
        for (const auto& neighbor : graph.adjacency[current]) {
            if (visited[neighbor.atom_index]) {
                continue;
            }
            visited[neighbor.atom_index] = true;
            queue.push_back(neighbor.atom_index);
        }
    }

    return std::all_of(visited.begin(), visited.end(), [](bool value) { return value; });
}

std::vector<std::vector<std::int64_t>> compute_eta_distances(const MordredEtaGraph& graph) {
    const auto atom_count = graph.atoms.size();
    std::vector<std::vector<std::int64_t>> distances(
        atom_count,
        std::vector<std::int64_t>(atom_count, kMordredDisconnectedDistance));

    for (std::size_t start = 0u; start < atom_count; ++start) {
        std::vector<std::size_t> queue;
        queue.reserve(atom_count);
        distances[start][start] = 0;
        queue.push_back(start);

        for (std::size_t head = 0u; head < queue.size(); ++head) {
            const auto current = queue[head];
            for (const auto& neighbor : graph.adjacency[current]) {
                if (distances[start][neighbor.atom_index] != kMordredDisconnectedDistance) {
                    continue;
                }
                distances[start][neighbor.atom_index] = distances[start][current] + 1;
                queue.push_back(neighbor.atom_index);
            }
        }
    }

    return distances;
}

std::optional<double> sum_eta_core_count(const MordredEtaGraph& graph) {
    double value = 0.0;
    for (const auto* atom : graph.atoms) {
        const auto core_count = eta_core_count(static_cast<std::uint32_t>(atom->GetAtomicNum()));
        if (!core_count.has_value()) {
            return std::nullopt;
        }
        value += *core_count;
    }
    return value;
}

std::optional<double> eta_beta_sigma(const MordredEtaGraph& graph, std::size_t atom_index) {
    const auto epsilon = eta_epsilon(*graph.atoms[atom_index]);
    if (!epsilon.has_value()) {
        return std::nullopt;
    }

    double value = 0.0;
    for (const auto& neighbor : graph.adjacency[atom_index]) {
        const auto* neighbor_atom = graph.atoms[neighbor.atom_index];
        if (is_hydrogen(*neighbor_atom)) {
            continue;
        }
        const auto neighbor_epsilon = eta_epsilon(*neighbor_atom);
        if (!neighbor_epsilon.has_value()) {
            return std::nullopt;
        }
        value += std::abs(*neighbor_epsilon - *epsilon) <= 0.3 ? 0.5 : 0.75;
    }
    return value;
}

std::optional<double> eta_non_sigma_contribution(
    const MordredEtaGraph& graph,
    std::size_t atom_index,
    const MordredEtaNeighbor& neighbor) {
    if (neighbor.bond_order == 1u) {
        return 0.0;
    }

    const auto begin_epsilon = eta_epsilon(*graph.atoms[atom_index]);
    const auto end_epsilon = eta_epsilon(*graph.atoms[neighbor.atom_index]);
    if (!begin_epsilon.has_value() || !end_epsilon.has_value()) {
        return std::nullopt;
    }

    const auto order_factor = neighbor.bond_order == 3u ? 2.0 : 1.0;
    const auto epsilon_delta = std::abs(*begin_epsilon - *end_epsilon);
    double coefficient = 1.0;
    if (neighbor.aromatic) {
        coefficient = 2.0;
    } else if (epsilon_delta > 0.3) {
        coefficient = 1.5;
    }

    return coefficient * order_factor;
}

std::optional<double> eta_beta_non_sigma(const MordredEtaGraph& graph, std::size_t atom_index) {
    double value = 0.0;
    for (const auto& neighbor : graph.adjacency[atom_index]) {
        if (is_hydrogen(*graph.atoms[neighbor.atom_index])) {
            continue;
        }
        const auto contribution = eta_non_sigma_contribution(graph, atom_index, neighbor);
        if (!contribution.has_value()) {
            return std::nullopt;
        }
        value += *contribution;
    }
    return value;
}

std::optional<double> eta_beta_delta(const MordredEtaGraph& graph, std::size_t atom_index) {
    const auto* atom = graph.atoms[atom_index];
    const auto outer_electrons =
        mordred_outer_electrons(static_cast<std::uint32_t>(atom->GetAtomicNum()));
    if (!outer_electrons.has_value()) {
        return std::nullopt;
    }

    if (atom->IsAromatic() || atom->IsInRing()
        || *outer_electrons <= static_cast<int>(atom->GetValence())) {
        return 0.0;
    }

    for (const auto& neighbor : graph.adjacency[atom_index]) {
        if (graph.atoms[neighbor.atom_index]->IsAromatic()) {
            return 0.5;
        }
    }
    return 0.0;
}

std::optional<double> eta_gamma(const MordredEtaGraph& graph, std::size_t atom_index) {
    const auto sigma = eta_beta_sigma(graph, atom_index);
    const auto non_sigma = eta_beta_non_sigma(graph, atom_index);
    const auto delta = eta_beta_delta(graph, atom_index);
    const auto core_count =
        eta_core_count(static_cast<std::uint32_t>(graph.atoms[atom_index]->GetAtomicNum()));
    if (!sigma.has_value() || !non_sigma.has_value() || !delta.has_value()
        || !core_count.has_value()) {
        return std::nullopt;
    }

    const auto beta = *sigma + *non_sigma + *delta;
    if (beta == 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return *core_count / beta;
}

std::optional<double> eta_composite_index(const MordredEtaGraph& graph, bool local) {
    const auto atom_count = graph.atoms.size();
    std::vector<double> gamma;
    gamma.reserve(atom_count);
    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        const auto atom_gamma = eta_gamma(graph, atom_index);
        if (!atom_gamma.has_value()) {
            return std::nullopt;
        }
        gamma.push_back(*atom_gamma);
    }

    const auto distances = compute_eta_distances(graph);
    double value = 0.0;
    for (std::size_t left = 0u; left < atom_count; ++left) {
        for (std::size_t right = left + 1u; right < atom_count; ++right) {
            const auto distance = distances[left][right];
            if ((local && distance != 1) || (!local && distance == 0)) {
                continue;
            }
            const auto distance_value = static_cast<double>(distance);
            const auto contribution =
                std::sqrt(gamma[left] * gamma[right] / (distance_value * distance_value));
            if (!std::isfinite(contribution)) {
                return std::nullopt;
            }
            value += contribution;
        }
    }
    return value;
}

std::optional<double> eta_epsilon_average(const MordredEtaGraph& graph) {
    if (graph.atoms.empty()) {
        return std::nullopt;
    }

    double value = 0.0;
    for (const auto* atom : graph.atoms) {
        const auto epsilon = eta_epsilon(*atom);
        if (!epsilon.has_value()) {
            return std::nullopt;
        }
        value += *epsilon;
    }
    return value / static_cast<double>(graph.atoms.size());
}

std::optional<double> eta_epsilon_5_average(const MordredEtaGraph& graph) {
    double value = 0.0;
    std::size_t count = 0u;

    for (std::size_t atom_index = 0u; atom_index < graph.atoms.size(); ++atom_index) {
        const auto* atom = graph.atoms[atom_index];
        if (is_hydrogen(*atom)) {
            if (graph.adjacency[atom_index].empty()) {
                return std::nullopt;
            }
            const auto* neighbor = graph.atoms[graph.adjacency[atom_index].front().atom_index];
            if (neighbor->GetAtomicNum() == 6u) {
                continue;
            }
        }

        const auto epsilon = eta_epsilon(*atom);
        if (!epsilon.has_value()) {
            return std::nullopt;
        }
        value += *epsilon;
        ++count;
    }

    if (count == 0u) {
        return std::nullopt;
    }
    return value / static_cast<double>(count);
}

std::uint32_t eta_saturated_default_valence(
    const OEChem::OEAtomBase& atom,
    std::uint32_t bond_order_sum) {
    // Mirrors RDKit PeriodicTable::GetDefaultValence for AlterMolecule(...,
    // saturated=True) AddHs behavior; -1 means "do not add implicit Hs".
    static constexpr std::array<int, 119> default_valences{{
        -1, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3,
        2, 1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, 3, 4, 3, 2, 1, 0, 1, 2, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, 3, 2, 3, 2, 1, 0, 1, 2, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2,
        3, 2, 1, 0, 1, 2, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    }};

    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto formal_charge = atom.GetFormalCharge();

    if (atomic_number == 7u && formal_charge > 0) {
        return std::max<std::uint32_t>(4u, bond_order_sum);
    }
    if (atomic_number == 8u && formal_charge < 0) {
        return std::max<std::uint32_t>(1u, bond_order_sum);
    }

    const auto from_allowed_valences = [bond_order_sum](std::initializer_list<std::uint32_t> vals) {
        for (const auto valence : vals) {
            if (bond_order_sum <= valence) {
                return valence;
            }
        }
        return bond_order_sum;
    };

    if (atomic_number == 15u) {
        return from_allowed_valences({3u, 5u});
    }
    if (atomic_number == 16u || atomic_number == 34u) {
        return from_allowed_valences({2u, 4u, 6u});
    }

    if (atomic_number >= default_valences.size() || default_valences[atomic_number] < 0) {
        return bond_order_sum;
    }
    return std::max<std::uint32_t>(
        static_cast<std::uint32_t>(default_valences[atomic_number]),
        bond_order_sum);
}

std::optional<double> eta_saturated_epsilon_average(const OEChem::OEMolBase& mol) {
    const auto saturated_mol = eta_altered_mol(mol, false, true);
    if (!saturated_mol.has_value()) {
        return std::nullopt;
    }

    const auto graph = build_eta_graph(*saturated_mol);
    if (graph.atoms.empty()) {
        return std::nullopt;
    }

    double value = 0.0;
    std::size_t count = 0u;
    for (std::size_t atom_index = 0u; atom_index < graph.atoms.size(); ++atom_index) {
        const auto* atom = graph.atoms[atom_index];
        const auto epsilon = eta_epsilon(*atom);
        if (!epsilon.has_value()) {
            return std::nullopt;
        }
        value += *epsilon;
        ++count;

        std::uint32_t bond_order_sum = 0u;
        for (const auto& neighbor : graph.adjacency[atom_index]) {
            bond_order_sum += neighbor.bond_order;
        }
        const auto default_valence = eta_saturated_default_valence(*atom, bond_order_sum);
        if (default_valence > bond_order_sum) {
            const auto hydrogen_count = default_valence - bond_order_sum;
            value += 0.3 * static_cast<double>(hydrogen_count);
            count += hydrogen_count;
        }
    }

    return value / static_cast<double>(count);
}

std::optional<double> optional_difference(
    const std::optional<double>& left,
    const std::optional<double>& right) {
    if (!left.has_value() || !right.has_value()) {
        return std::nullopt;
    }
    return *left - *right;
}

std::optional<double> optional_positive_delta(
    const std::optional<double>& left,
    const std::optional<double>& right,
    double denominator) {
    const auto difference = optional_difference(left, right);
    if (!difference.has_value() || denominator == 0.0) {
        return std::nullopt;
    }
    return std::max(*difference / denominator, 0.0);
}

std::optional<double> optional_average(
    const std::optional<double>& value,
    double denominator) {
    if (!value.has_value() || denominator == 0.0) {
        return std::nullopt;
    }
    return *value / denominator;
}

void add_eta_value(
    MordredEtaValues& values,
    const char* name,
    const std::optional<double>& value) {
    values.push_back({name, value});
}

std::optional<MordredEtaValues> compute_eta_values(
    const OEChem::OEMolBase& mol,
    std::uint32_t ring_count) {
    const auto base_mol = eta_prepared_mol(mol, false);
    const auto base_graph = build_eta_graph(base_mol);
    if (!is_connected_eta_graph(base_graph)) {
        return std::nullopt;
    }

    const auto atom_count = base_graph.atoms.size();
    const auto atom_count_value = static_cast<double>(atom_count);
    const auto alpha = sum_eta_core_count(base_graph);

    std::optional<OEChem::OEGraphMol> reference_mol = eta_altered_mol(mol, false, false);
    std::optional<MordredEtaGraph> reference_graph;
    if (reference_mol.has_value()) {
        reference_graph = build_eta_graph(*reference_mol);
    }

    const auto alpha_reference =
        reference_graph.has_value() ? sum_eta_core_count(*reference_graph) : std::nullopt;

    double beta_sigma = 0.0;
    double beta_non_sigma_contribution = 0.0;
    double beta_delta_contribution = 0.0;
    bool beta_sigma_valid = true;
    bool beta_non_sigma_valid = true;
    bool beta_delta_valid = true;
    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        const auto sigma = eta_beta_sigma(base_graph, atom_index);
        if (sigma.has_value()) {
            beta_sigma += *sigma / 2.0;
        } else {
            beta_sigma_valid = false;
        }

        const auto non_sigma = eta_beta_non_sigma(base_graph, atom_index);
        if (non_sigma.has_value()) {
            beta_non_sigma_contribution += *non_sigma / 2.0;
        } else {
            beta_non_sigma_valid = false;
        }

        const auto delta = eta_beta_delta(base_graph, atom_index);
        if (delta.has_value()) {
            beta_delta_contribution += *delta;
        } else {
            beta_delta_valid = false;
        }
    }
    const std::optional<double> beta_sigma_value =
        beta_sigma_valid ? std::optional<double>{beta_sigma} : std::nullopt;
    const std::optional<double> beta_non_sigma_delta_value =
        beta_delta_valid ? std::optional<double>{beta_delta_contribution} : std::nullopt;
    const std::optional<double> beta_non_sigma_value =
        beta_non_sigma_valid && beta_delta_valid
            ? std::optional<double>{beta_non_sigma_contribution + beta_delta_contribution}
            : std::nullopt;
    const std::optional<double> beta =
        beta_sigma_value.has_value() && beta_non_sigma_value.has_value()
            ? std::optional<double>{*beta_sigma_value + *beta_non_sigma_value}
            : std::nullopt;

    const auto eta = eta_composite_index(base_graph, false);
    const auto eta_local = eta_composite_index(base_graph, true);
    const auto eta_reference =
        reference_graph.has_value() ? eta_composite_index(*reference_graph, false) : std::nullopt;
    const auto eta_reference_local =
        reference_graph.has_value() ? eta_composite_index(*reference_graph, true) : std::nullopt;
    const auto eta_functionality = optional_difference(eta_reference, eta);
    const auto eta_functionality_local = optional_difference(eta_reference_local, eta_local);

    std::optional<double> eta_branch;
    std::optional<double> eta_branch_ring;
    if (eta_reference_local.has_value() && atom_count > 1u) {
        const auto normal_local = atom_count == 2u
            ? 1.0
            : std::sqrt(2.0) + 0.5 * static_cast<double>(atom_count - 3u);
        eta_branch = normal_local - *eta_reference_local;
        eta_branch_ring = *eta_branch + 0.086 * static_cast<double>(ring_count);
    }

    const auto explicit_mol = eta_prepared_mol(mol, true);
    const auto explicit_graph = build_eta_graph(explicit_mol);
    const auto epsilon_1 = eta_epsilon_average(explicit_graph);
    const auto epsilon_2 = eta_epsilon_average(base_graph);
    const auto epsilon_5 = eta_epsilon_5_average(explicit_graph);

    const auto reference_explicit_mol = eta_altered_mol(mol, true, false);
    const auto epsilon_3 = reference_explicit_mol.has_value()
        ? eta_epsilon_average(build_eta_graph(*reference_explicit_mol))
        : std::optional<double>{};
    const auto epsilon_4 = eta_saturated_epsilon_average(mol);

    std::optional<double> psi;
    if (alpha.has_value() && epsilon_2.has_value() && *epsilon_2 != 0.0) {
        psi = *alpha / (atom_count_value * *epsilon_2);
    }

    MordredEtaValues values;
    values.reserve(45u);
    add_eta_value(values, "ETA_alpha", alpha);
    add_eta_value(values, "AETA_alpha", optional_average(alpha, atom_count_value));
    if (alpha.has_value() && *alpha != 0.0) {
        for (const auto [name, degree] : {
                 std::pair<const char*, std::size_t>{"ETA_shape_p", 1u},
                 std::pair<const char*, std::size_t>{"ETA_shape_y", 3u},
                 std::pair<const char*, std::size_t>{"ETA_shape_x", 4u},
             }) {
            double shape_alpha = 0.0;
            bool shape_valid = true;
            for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
                if (base_graph.adjacency[atom_index].size() != degree) {
                    continue;
                }
                const auto core_count = eta_core_count(
                    static_cast<std::uint32_t>(base_graph.atoms[atom_index]->GetAtomicNum()));
                if (!core_count.has_value()) {
                    shape_valid = false;
                    break;
                }
                shape_alpha += *core_count;
            }
            add_eta_value(values, name, shape_valid ? std::optional<double>{shape_alpha / *alpha}
                                                    : std::nullopt);
        }
    } else {
        add_eta_value(values, "ETA_shape_p", std::nullopt);
        add_eta_value(values, "ETA_shape_y", std::nullopt);
        add_eta_value(values, "ETA_shape_x", std::nullopt);
    }

    add_eta_value(values, "ETA_beta", beta);
    add_eta_value(values, "AETA_beta", optional_average(beta, atom_count_value));
    add_eta_value(values, "ETA_beta_s", beta_sigma_value);
    add_eta_value(values, "AETA_beta_s", optional_average(beta_sigma_value, atom_count_value));
    add_eta_value(values, "ETA_beta_ns", beta_non_sigma_value);
    add_eta_value(
        values,
        "AETA_beta_ns",
        optional_average(beta_non_sigma_value, atom_count_value));
    add_eta_value(values, "ETA_beta_ns_d", beta_non_sigma_delta_value);
    add_eta_value(
        values,
        "AETA_beta_ns_d",
        optional_average(beta_non_sigma_delta_value, atom_count_value));
    add_eta_value(values, "ETA_eta", eta);
    add_eta_value(values, "AETA_eta", optional_average(eta, atom_count_value));
    add_eta_value(values, "ETA_eta_L", eta_local);
    add_eta_value(values, "AETA_eta_L", optional_average(eta_local, atom_count_value));
    add_eta_value(values, "ETA_eta_R", eta_reference);
    add_eta_value(values, "AETA_eta_R", optional_average(eta_reference, atom_count_value));
    add_eta_value(values, "ETA_eta_RL", eta_reference_local);
    add_eta_value(
        values,
        "AETA_eta_RL",
        optional_average(eta_reference_local, atom_count_value));
    add_eta_value(values, "ETA_eta_F", eta_functionality);
    add_eta_value(
        values,
        "AETA_eta_F",
        optional_average(eta_functionality, atom_count_value));
    add_eta_value(values, "ETA_eta_FL", eta_functionality_local);
    add_eta_value(
        values,
        "AETA_eta_FL",
        optional_average(eta_functionality_local, atom_count_value));
    add_eta_value(values, "ETA_eta_B", eta_branch);
    add_eta_value(values, "AETA_eta_B", optional_average(eta_branch, atom_count_value));
    add_eta_value(values, "ETA_eta_BR", eta_branch_ring);
    add_eta_value(values, "AETA_eta_BR", optional_average(eta_branch_ring, atom_count_value));
    add_eta_value(
        values,
        "ETA_dAlpha_A",
        optional_positive_delta(alpha, alpha_reference, atom_count_value));
    add_eta_value(
        values,
        "ETA_dAlpha_B",
        optional_positive_delta(alpha_reference, alpha, atom_count_value));
    add_eta_value(values, "ETA_epsilon_1", epsilon_1);
    add_eta_value(values, "ETA_epsilon_2", epsilon_2);
    add_eta_value(values, "ETA_epsilon_3", epsilon_3);
    add_eta_value(values, "ETA_epsilon_4", epsilon_4);
    add_eta_value(values, "ETA_epsilon_5", epsilon_5);
    add_eta_value(values, "ETA_dEpsilon_A", optional_difference(epsilon_1, epsilon_3));
    add_eta_value(values, "ETA_dEpsilon_B", optional_difference(epsilon_1, epsilon_4));
    add_eta_value(values, "ETA_dEpsilon_C", optional_difference(epsilon_3, epsilon_4));
    add_eta_value(values, "ETA_dEpsilon_D", optional_difference(epsilon_2, epsilon_5));
    const auto delta_beta = optional_difference(beta_non_sigma_value, beta_sigma_value);
    add_eta_value(values, "ETA_dBeta", delta_beta);
    add_eta_value(values, "AETA_dBeta", optional_average(delta_beta, atom_count_value));
    add_eta_value(values, "ETA_psi_1", psi);
    add_eta_value(
        values,
        "ETA_dPsi_A",
        psi.has_value() ? std::optional<double>{std::max(0.714 - *psi, 0.0)} : std::nullopt);
    add_eta_value(
        values,
        "ETA_dPsi_B",
        psi.has_value() ? std::optional<double>{std::max(*psi - 0.714, 0.0)} : std::nullopt);
    return values;
}

double mordred_estate_intrinsic_state(
    const OEChem::OEAtomBase& atom,
    std::size_t heavy_atom_degree) {
    const auto degree = static_cast<std::uint32_t>(heavy_atom_degree);
    if (degree == 0u) {
        return 0.0;
    }

    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    const auto outer_electrons = mordred_outer_electrons(atomic_number);
    if (!outer_electrons.has_value()) {
        return 0.0;
    }

    const auto principal_quantum_number =
        static_cast<double>(mordred_principal_quantum_number(atomic_number));
    const auto valence_delta =
        static_cast<double>(*outer_electrons) - static_cast<double>(atom.GetTotalHCount());
    const auto intrinsic_state =
        4.0 / (principal_quantum_number * principal_quantum_number) * valence_delta + 1.0;
    return intrinsic_state / static_cast<double>(degree);
}

std::vector<double> compute_mordred_estate_indices(const MordredHeavyAtomGraph& graph) {
    const auto atom_count = graph.atoms.size();
    std::vector<double> intrinsic_states(atom_count, 0.0);
    std::vector<double> accumulators(atom_count, 0.0);
    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        intrinsic_states[atom_index] = mordred_estate_intrinsic_state(
            *graph.atoms[atom_index],
            graph.adjacency[atom_index].size());
    }

    const auto distances = compute_mordred_heavy_atom_distances(graph);
    for (std::size_t left = 0u; left < atom_count; ++left) {
        for (std::size_t right = left + 1u; right < atom_count; ++right) {
            const auto distance = distances[left][right];
            if (distance >= kMordredDisconnectedDistance) {
                continue;
            }
            const auto path_length = static_cast<double>(distance + 1);
            const auto contribution =
                (intrinsic_states[left] - intrinsic_states[right])
                / (path_length * path_length);
            accumulators[left] += contribution;
            accumulators[right] -= contribution;
        }
    }

    std::vector<double> indices(atom_count, 0.0);
    for (std::size_t atom_index = 0u; atom_index < atom_count; ++atom_index) {
        indices[atom_index] = accumulators[atom_index] + intrinsic_states[atom_index];
    }
    return indices;
}

bool labute_values_align_with_heavy_atom_graph(
    const MordredLabuteAsaValues& labute_values,
    const MordredHeavyAtomGraph& graph) {
    if (labute_values.atom_contributions.size() != graph.atoms.size()
        || labute_values.atom_ids.size() != graph.atoms.size()) {
        return false;
    }

    for (std::size_t atom_index = 0u; atom_index < graph.atoms.size(); ++atom_index) {
        if (labute_values.atom_ids[atom_index] != graph.atoms[atom_index]->GetIdx()) {
            return false;
        }
    }
    return true;
}

bool labute_values_align_with_crippen_atom_contributions(
    const MordredLabuteAsaValues& labute_values,
    const MordredCrippenAtomContributions& crippen_contributions) {
    if (labute_values.atom_contributions.size() != crippen_contributions.mr.size()
        || labute_values.atom_ids.size() != crippen_contributions.atom_ids.size()) {
        return false;
    }

    for (std::size_t atom_index = 0u; atom_index < labute_values.atom_ids.size(); ++atom_index) {
        if (labute_values.atom_ids[atom_index] != crippen_contributions.atom_ids[atom_index]) {
            return false;
        }
    }
    return true;
}

bool labute_values_align_with_gasteiger_atom_charges(
    const MordredLabuteAsaValues& labute_values,
    const MordredGasteigerAtomCharges& gasteiger_charges) {
    if (labute_values.atom_contributions.size() != gasteiger_charges.charges.size()
        || labute_values.atom_ids.size() != gasteiger_charges.atom_ids.size()) {
        return false;
    }

    for (std::size_t atom_index = 0u; atom_index < labute_values.atom_ids.size(); ++atom_index) {
        if (labute_values.atom_ids[atom_index] != gasteiger_charges.atom_ids[atom_index]) {
            return false;
        }
    }
    return true;
}

template <std::size_t BinCount>
std::optional<std::array<double, BinCount>> compute_crippen_vsa_values(
    const OEChem::OEMolBase& mol,
    const MordredLabuteAsaValues& labute_values,
    const std::array<double, BinCount>& bins,
    const std::vector<double> MordredCrippenAtomContributions::*contribution_member) {
    const auto crippen_contributions = compute_crippen_atom_contributions(mol);
    if (!crippen_contributions.has_value()
        || !labute_values_align_with_crippen_atom_contributions(
            labute_values,
            *crippen_contributions)) {
        return std::nullopt;
    }

    const auto& property_contributions = (*crippen_contributions).*contribution_member;
    if (property_contributions.size() != labute_values.atom_contributions.size()) {
        return std::nullopt;
    }

    std::array<double, BinCount> values{};
    for (std::size_t atom_index = 0u; atom_index < property_contributions.size(); ++atom_index) {
        const auto property_contribution = property_contributions[atom_index];
        const auto atom_contribution = labute_values.atom_contributions[atom_index];
        if (!std::isfinite(property_contribution) || !std::isfinite(atom_contribution)) {
            return std::nullopt;
        }

        // RDKit MOE VSA descriptors use upper_bound/bisect_right and hide the tail bin.
        const auto bin = static_cast<std::size_t>(
            std::upper_bound(bins.begin(), bins.end(), property_contribution) - bins.begin());
        if (bin < values.size()) {
            values[bin] += atom_contribution;
        }
    }
    return values;
}

std::optional<std::array<double, 9>> compute_smr_vsa_values(
    const OEChem::OEMolBase& mol,
    const MordredLabuteAsaValues& labute_values) {
    static constexpr std::array<double, 9> mr_bins{{
        1.29,
        1.82,
        2.24,
        2.45,
        2.75,
        3.05,
        3.63,
        3.8,
        4.0,
    }};

    return compute_crippen_vsa_values(
        mol,
        labute_values,
        mr_bins,
        &MordredCrippenAtomContributions::mr);
}

std::optional<std::array<double, 11>> compute_slogp_vsa_values(
    const OEChem::OEMolBase& mol,
    const MordredLabuteAsaValues& labute_values) {
    static constexpr std::array<double, 11> logp_bins{{
        -0.4,
        -0.2,
        0.0,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
    }};

    return compute_crippen_vsa_values(
        mol,
        labute_values,
        logp_bins,
        &MordredCrippenAtomContributions::logp);
}

std::optional<std::array<double, 14>> compute_peoe_vsa_values(const OEChem::OEMolBase& mol) {
    static constexpr std::array<double, 13> charge_bins{{
        -0.3,
        -0.25,
        -0.20,
        -0.15,
        -0.10,
        -0.05,
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    }};

    const auto labute_values = compute_peoe_labute_asa_values(mol);
    const auto charges = compute_gasteiger_atom_charges(mol);
    if (!labute_values_align_with_gasteiger_atom_charges(labute_values, charges)) {
        return std::nullopt;
    }

    std::array<double, 14> values{};
    for (std::size_t atom_index = 0u; atom_index < charges.charges.size(); ++atom_index) {
        const auto charge = charges.charges[atom_index];
        const auto atom_contribution = labute_values.atom_contributions[atom_index];
        // RDKit keeps the full tail bin. Mordred's preset exposes only
        // PEOE_VSA1..13, so NaN charge/surface from bonded dummy atoms is hidden there.
        const auto bin = static_cast<std::size_t>(
            std::upper_bound(charge_bins.begin(), charge_bins.end(), charge)
            - charge_bins.begin());
        values[bin] += atom_contribution;
    }
    return values;
}

std::optional<std::array<double, 9>> compute_vsa_estate_values(
    const OEChem::OEMolBase& mol,
    const MordredLabuteAsaValues& labute_values) {
    static constexpr std::array<double, 9> vsa_bins{{
        4.78,
        5.00,
        5.410,
        5.740,
        6.00,
        6.07,
        6.45,
        7.00,
        11.0,
    }};

    const auto working_mol = implicit_hydrogen_estate_mol(mol);
    const auto graph = build_mordred_heavy_atom_graph(working_mol);
    // Labute and EState each work on their own hydrogen-suppressed molecule copy;
    // the atom-id check keeps their per-heavy-atom vectors aligned before binning.
    if (!labute_values_align_with_heavy_atom_graph(labute_values, graph)) {
        return std::nullopt;
    }

    const auto estate_indices = compute_mordred_estate_indices(graph);
    std::array<double, 9> values{};
    for (std::size_t atom_index = 0u; atom_index < graph.atoms.size(); ++atom_index) {
        const auto estate_index = estate_indices[atom_index];
        const auto atom_contribution = labute_values.atom_contributions[atom_index];
        if (!std::isfinite(estate_index) || !std::isfinite(atom_contribution)) {
            return std::nullopt;
        }

        // RDKit uses bisect_right: exact boundary contributions move to the higher bin.
        const auto bin = static_cast<std::size_t>(
            std::upper_bound(vsa_bins.begin(), vsa_bins.end(), atom_contribution)
            - vsa_bins.begin());
        if (bin < values.size()) {
            values[bin] += estate_index;
        }
    }
    return values;
}

std::optional<std::array<double, 10>> compute_estate_vsa_values(
    const OEChem::OEMolBase& mol,
    const MordredLabuteAsaValues& labute_values) {
    static constexpr std::array<double, 10> estate_bins{{
        -0.390,
        0.290,
        0.717,
        1.165,
        1.540,
        1.807,
        2.05,
        4.69,
        9.17,
        15.0,
    }};

    const auto working_mol = implicit_hydrogen_estate_mol(mol);
    const auto graph = build_mordred_heavy_atom_graph(working_mol);
    if (!labute_values_align_with_heavy_atom_graph(labute_values, graph)) {
        return std::nullopt;
    }

    const auto estate_indices = compute_mordred_estate_indices(graph);
    std::array<double, 10> values{};
    for (std::size_t atom_index = 0u; atom_index < graph.atoms.size(); ++atom_index) {
        const auto estate_index = estate_indices[atom_index];
        const auto atom_contribution = labute_values.atom_contributions[atom_index];
        if (!std::isfinite(estate_index) || !std::isfinite(atom_contribution)) {
            return std::nullopt;
        }

        // RDKit EState_VSA_ uses bisect_right on EState indices and adds
        // per-heavy-atom Labute contributions; Mordred exposes only bins 0..9.
        const auto bin = static_cast<std::size_t>(
            std::upper_bound(estate_bins.begin(), estate_bins.end(), estate_index)
            - estate_bins.begin());
        if (bin < values.size()) {
            values[bin] += atom_contribution;
        }
    }
    return values;
}

MordredEStateValues compute_mordred_estate_values(const OEChem::OEMolBase& mol) {
    const auto working_mol = implicit_hydrogen_estate_mol(mol);
    const auto graph = build_mordred_heavy_atom_graph(working_mol);
    const auto indices = compute_mordred_estate_indices(graph);
    const auto& patterns = mordred_estate_patterns();

    MordredEStateValues values;
    values.counts.reserve(patterns.size());
    values.sums.reserve(patterns.size());
    values.maxima.reserve(patterns.size());
    values.minima.reserve(patterns.size());

    for (const auto& pattern : patterns) {
        const auto matched_atom_indices =
            mordred_estate_pattern_root_atom_indices(working_mol, pattern.smarts);
        double sum = 0.0;
        std::optional<double> maximum;
        std::optional<double> minimum;

        for (std::size_t graph_index = 0u; graph_index < graph.atoms.size(); ++graph_index) {
            const auto atom_id = graph.atoms[graph_index]->GetIdx();
            if (matched_atom_indices.find(atom_id) == matched_atom_indices.end()) {
                continue;
            }
            const auto value = indices[graph_index];
            sum += value;
            maximum = maximum.has_value() ? std::max(*maximum, value) : value;
            minimum = minimum.has_value() ? std::min(*minimum, value) : value;
        }

        values.counts.push_back(static_cast<std::uint32_t>(matched_atom_indices.size()));
        values.sums.push_back(sum);
        values.maxima.push_back(maximum);
        values.minima.push_back(minimum);
    }
    return values;
}

double chi_sigma_electrons(const OEChem::OEAtomBase& atom) {
    double sigma_electrons = 0.0;
    for (OESystem::OEIter<OEChem::OEAtomBase> neighbor = atom.GetAtoms(); neighbor; ++neighbor) {
        if (!is_hydrogen(*neighbor)) {
            sigma_electrons += 1.0;
        }
    }
    return sigma_electrons;
}

std::optional<double> chi_valence_electrons(const OEChem::OEAtomBase& atom) {
    const auto atomic_number = static_cast<std::uint32_t>(atom.GetAtomicNum());
    if (atomic_number == 1u) {
        return 0.0;
    }

    const auto outer_electrons = mordred_outer_electrons(atomic_number);
    if (!outer_electrons.has_value()) {
        return std::nullopt;
    }

    const auto formal_charge = atom.GetFormalCharge();
    const auto zv = static_cast<double>(*outer_electrons - formal_charge);
    const auto z = static_cast<double>(static_cast<int>(atomic_number) - formal_charge);
    const auto hydrogens = static_cast<double>(atom.GetTotalHCount());
    const auto denominator = z - zv - 1.0;
    if (denominator == 0.0) {
        return std::nullopt;
    }
    return (zv - hydrogens) / denominator;
}

ChiPathWalkTotals count_chi_path_walks(
    const std::vector<std::vector<PathCountNeighbor>>& adjacency,
    const std::vector<double>& atom_properties,
    std::size_t current,
    std::size_t remaining_bonds,
    double property_product,
    std::vector<bool>& visited) {
    if (remaining_bonds == 0u) {
        if (property_product <= 0.0) {
            return {0u, 0.0, false};
        }
        return {1u, std::pow(property_product, -0.5), true};
    }

    ChiPathWalkTotals totals;
    for (const auto neighbor : adjacency[current]) {
        if (visited[neighbor.atom_index]) {
            continue;
        }
        visited[neighbor.atom_index] = true;
        const auto child_totals = count_chi_path_walks(
            adjacency,
            atom_properties,
            neighbor.atom_index,
            remaining_bonds - 1u,
            property_product * atom_properties[neighbor.atom_index],
            visited);
        if (!child_totals.valid) {
            return child_totals;
        }
        totals.count += child_totals.count;
        totals.sum += child_totals.sum;
        visited[neighbor.atom_index] = false;
    }
    return totals;
}

std::optional<double> compute_chi_path_value(
    const MordredHeavyAtomGraph& graph,
    const std::vector<double>& atom_properties,
    std::size_t order,
    bool averaged) {
    ChiPathWalkTotals totals;

    if (order == 0u) {
        totals.count = static_cast<std::uint32_t>(atom_properties.size());
        for (const auto atom_property : atom_properties) {
            if (atom_property <= 0.0) {
                return std::nullopt;
            }
            totals.sum += std::pow(atom_property, -0.5);
        }
    } else {
        std::vector<bool> visited(graph.adjacency.size(), false);
        for (std::size_t start = 0u; start < graph.adjacency.size(); ++start) {
            visited[start] = true;
            const auto start_totals = count_chi_path_walks(
                graph.adjacency,
                atom_properties,
                start,
                order,
                atom_properties[start],
                visited);
            if (!start_totals.valid) {
                return std::nullopt;
            }
            totals.count += start_totals.count;
            totals.sum += start_totals.sum;
            visited[start] = false;
        }
        totals.count /= 2u;
        totals.sum /= 2.0;
    }

    if (averaged) {
        if (totals.count == 0u) {
            return std::nullopt;
        }
        return totals.sum / static_cast<double>(totals.count);
    }
    return totals.sum;
}

MordredChiPathValues compute_chi_path_values(const MordredHeavyAtomGraph& graph) {
    std::vector<double> sigma_properties;
    std::vector<double> valence_properties;
    sigma_properties.reserve(graph.atoms.size());
    valence_properties.reserve(graph.atoms.size());

    for (const auto* atom : graph.atoms) {
        sigma_properties.push_back(chi_sigma_electrons(*atom));
        const auto valence = chi_valence_electrons(*atom);
        valence_properties.push_back(valence.value_or(0.0));
    }

    MordredChiPathValues values;
    for (std::size_t order = 0u; order < 8u; ++order) {
        values.xp_d[order] = compute_chi_path_value(graph, sigma_properties, order, false);
        values.axp_d[order] = compute_chi_path_value(graph, sigma_properties, order, true);
        values.xp_dv[order] = compute_chi_path_value(graph, valence_properties, order, false);
        values.axp_dv[order] = compute_chi_path_value(graph, valence_properties, order, true);
    }
    return values;
}

bool contains_index(const std::vector<std::size_t>& values, std::size_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void collect_connected_bond_subgraphs(
    const MordredHeavyAtomGraph& graph,
    std::size_t target_bond_count,
    std::vector<std::size_t> selected_bonds,
    std::vector<std::size_t> candidate_bonds,
    std::set<std::vector<std::size_t>>& subgraphs) {
    if (selected_bonds.size() == target_bond_count) {
        std::sort(selected_bonds.begin(), selected_bonds.end());
        subgraphs.insert(selected_bonds);
        return;
    }

    while (!candidate_bonds.empty()) {
        const auto next_bond = candidate_bonds.back();
        candidate_bonds.pop_back();
        if (contains_index(selected_bonds, next_bond)) {
            continue;
        }

        auto next_selected = selected_bonds;
        next_selected.push_back(next_bond);

        auto next_candidates = candidate_bonds;
        for (const auto neighbor_bond : graph.bond_neighbors[next_bond]) {
            if (!contains_index(next_selected, neighbor_bond)) {
                next_candidates.push_back(neighbor_bond);
            }
        }

        collect_connected_bond_subgraphs(
            graph, target_bond_count, next_selected, next_candidates, subgraphs);
    }
}

std::set<std::vector<std::size_t>> find_connected_bond_subgraphs(
    const MordredHeavyAtomGraph& graph,
    std::size_t target_bond_count) {
    std::set<std::vector<std::size_t>> subgraphs;
    if (target_bond_count == 0u || graph.bonds.size() < target_bond_count) {
        return subgraphs;
    }

    for (std::size_t bond_index = 0u; bond_index < graph.bonds.size(); ++bond_index) {
        collect_connected_bond_subgraphs(
            graph, target_bond_count, {bond_index}, graph.bond_neighbors[bond_index], subgraphs);
    }
    return subgraphs;
}

enum class ChiNonPathType {
    Chain,
    Cluster,
    PathCluster,
    Path,
};

struct ChiClassifiedSubgraph {
    ChiNonPathType type;
    std::vector<std::size_t> atoms;
};

ChiClassifiedSubgraph classify_chi_bond_subgraph(
    const MordredHeavyAtomGraph& graph,
    const std::vector<std::size_t>& selected_bonds) {
    std::vector<std::uint32_t> selected_degrees(graph.atoms.size(), 0u);
    for (const auto bond_index : selected_bonds) {
        const auto [begin, end] = graph.bonds[bond_index];
        ++selected_degrees[begin];
        ++selected_degrees[end];
    }

    std::vector<std::size_t> selected_atoms;
    bool has_degree_two = false;
    bool has_only_path_degrees = true;
    for (std::size_t atom_index = 0u; atom_index < selected_degrees.size(); ++atom_index) {
        const auto degree = selected_degrees[atom_index];
        if (degree == 0u) {
            continue;
        }
        selected_atoms.push_back(atom_index);
        has_degree_two = has_degree_two || degree == 2u;
        has_only_path_degrees = has_only_path_degrees && (degree == 1u || degree == 2u);
    }

    // Mordred classifies connected selected-bond subgraphs by the subgraph
    // topology itself; cycles win before degree-pattern checks.
    if (selected_bonds.size() >= selected_atoms.size()) {
        return {ChiNonPathType::Chain, selected_atoms};
    }
    if (has_only_path_degrees) {
        return {ChiNonPathType::Path, selected_atoms};
    }
    if (has_degree_two) {
        return {ChiNonPathType::PathCluster, selected_atoms};
    }
    return {ChiNonPathType::Cluster, selected_atoms};
}

std::optional<double> compute_chi_subgraph_value(
    const std::vector<ChiClassifiedSubgraph>& classified_subgraphs,
    const std::vector<double>& atom_properties,
    ChiNonPathType type) {
    double sum = 0.0;
    for (const auto& subgraph : classified_subgraphs) {
        if (subgraph.type != type) {
            continue;
        }

        double property_product = 1.0;
        for (const auto atom_index : subgraph.atoms) {
            property_product *= atom_properties[atom_index];
        }
        if (property_product <= 0.0) {
            return std::nullopt;
        }
        sum += std::pow(property_product, -0.5);
    }
    return sum;
}

MordredChiNonPathValues compute_chi_non_path_values(const MordredHeavyAtomGraph& graph) {
    std::vector<double> sigma_properties;
    std::vector<double> valence_properties;
    sigma_properties.reserve(graph.atoms.size());
    valence_properties.reserve(graph.atoms.size());

    for (const auto* atom : graph.atoms) {
        sigma_properties.push_back(chi_sigma_electrons(*atom));
        const auto valence = chi_valence_electrons(*atom);
        valence_properties.push_back(valence.value_or(0.0));
    }

    MordredChiNonPathValues values;
    for (std::size_t order = 1u; order <= 7u; ++order) {
        std::vector<ChiClassifiedSubgraph> classified_subgraphs;
        const auto subgraphs = find_connected_bond_subgraphs(graph, order);
        classified_subgraphs.reserve(subgraphs.size());
        for (const auto& subgraph : subgraphs) {
            classified_subgraphs.push_back(classify_chi_bond_subgraph(graph, subgraph));
        }

        if (order >= 3u) {
            values.xch_d[order] = compute_chi_subgraph_value(
                classified_subgraphs, sigma_properties, ChiNonPathType::Chain);
            values.xch_dv[order] = compute_chi_subgraph_value(
                classified_subgraphs, valence_properties, ChiNonPathType::Chain);
        }
        if (order >= 3u && order <= 6u) {
            values.xc_d[order] = compute_chi_subgraph_value(
                classified_subgraphs, sigma_properties, ChiNonPathType::Cluster);
            values.xc_dv[order] = compute_chi_subgraph_value(
                classified_subgraphs, valence_properties, ChiNonPathType::Cluster);
        }
        if (order >= 4u && order <= 6u) {
            values.xpc_d[order] = compute_chi_subgraph_value(
                classified_subgraphs, sigma_properties, ChiNonPathType::PathCluster);
            values.xpc_dv[order] = compute_chi_subgraph_value(
                classified_subgraphs, valence_properties, ChiNonPathType::PathCluster);
        }
    }
    return values;
}

std::optional<double> kappa_shape_index(
    std::uint32_t heavy_atom_count,
    const MordredPathCountValues& path_count_values,
    std::size_t order) {
    const auto path_count = path_count_values.mpc[order];
    if (path_count == 0u) {
        return std::nullopt;
    }

    const auto atom_count = static_cast<double>(heavy_atom_count);
    const auto path_count_value = static_cast<double>(path_count);
    const auto minimum_path_count = atom_count - static_cast<double>(order);
    if (order == 1u) {
        const auto maximum_path_count = 0.5 * atom_count * (atom_count - 1.0);
        return 2.0 * maximum_path_count * minimum_path_count
            / (path_count_value * path_count_value);
    }
    if (order == 2u) {
        const auto maximum_path_count = 0.5 * (atom_count - 1.0) * (atom_count - 2.0);
        return 2.0 * maximum_path_count * minimum_path_count
            / (path_count_value * path_count_value);
    }

    const auto maximum_path_count =
        heavy_atom_count % 2u == 0u
            ? 0.25 * (atom_count - 2.0) * (atom_count - 2.0)
            : 0.25 * (atom_count - 1.0) * (atom_count - 3.0);
    return 4.0 * maximum_path_count * minimum_path_count
        / (path_count_value * path_count_value);
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

void set_geometrical_index_values(
    DescriptorSetBuilder& builder,
    const MordredGeometricalIndexValues& values) {
    set_float(builder, "GeomDiameter", values.diameter);
    set_float(builder, "GeomRadius", values.radius);
    set_optional_float(builder, "GeomShapeIndex", values.shape_index);
    set_optional_float(builder, "GeomPetitjeanIndex", values.petitjean_index);
}

void set_gravitational_index_values(
    DescriptorSetBuilder& builder,
    const MordredGravitationalIndexValues& values) {
    set_optional_float(builder, "GRAV", values.heavy);
    set_optional_float(builder, "GRAVH", values.all);
    set_optional_float(builder, "GRAVp", values.heavy_pair);
    set_optional_float(builder, "GRAVHp", values.all_pair);
}

void set_moment_of_inertia_values(
    DescriptorSetBuilder& builder,
    const MordredMomentOfInertiaValues& values) {
    set_float(builder, "MOMI-X", values.axes[0]);
    set_float(builder, "MOMI-Y", values.axes[1]);
    set_float(builder, "MOMI-Z", values.axes[2]);
}

void set_low_count_3d_values(
    DescriptorSetBuilder& builder,
    const MordredLowCount3DValues& values) {
    if (values.geometrical.has_value()) {
        set_geometrical_index_values(builder, *values.geometrical);
    }
    set_gravitational_index_values(builder, values.gravitational);
    if (values.moment_of_inertia.has_value()) {
        set_moment_of_inertia_values(builder, *values.moment_of_inertia);
    }
    set_optional_float(builder, "PBF", values.pbf);
}

std::string mordred_morse_descriptor_name(std::size_t distance_index, const char* suffix) {
    std::ostringstream stream;
    stream << "Mor" << std::setw(2) << std::setfill('0') << distance_index << suffix;
    return stream.str();
}

void set_mordred_morse_values(
    DescriptorSetBuilder& builder,
    const std::vector<MordredMoRSEValues>& values) {
    for (const auto& property_values : values) {
        for (std::size_t distance_index = 1u; distance_index <= 32u; ++distance_index) {
            set_optional_float(
                builder,
                mordred_morse_descriptor_name(distance_index, property_values.suffix),
                property_values.values[distance_index]);
        }
    }
}

void set_chi_path_values(DescriptorSetBuilder& builder, const MordredChiPathValues& values) {
    for (std::size_t order = 0u; order < 8u; ++order) {
        const auto order_text = std::to_string(order);
        set_optional_float(builder, "Xp-" + order_text + "d", values.xp_d[order]);
        set_optional_float(builder, "AXp-" + order_text + "d", values.axp_d[order]);
        set_optional_float(builder, "Xp-" + order_text + "dv", values.xp_dv[order]);
        set_optional_float(builder, "AXp-" + order_text + "dv", values.axp_dv[order]);
    }
}

void set_chi_non_path_values(
    DescriptorSetBuilder& builder,
    const MordredChiNonPathValues& values) {
    for (std::size_t order = 3u; order <= 7u; ++order) {
        const auto order_text = std::to_string(order);
        set_optional_float(builder, "Xch-" + order_text + "d", values.xch_d[order]);
        set_optional_float(builder, "Xch-" + order_text + "dv", values.xch_dv[order]);
    }
    for (std::size_t order = 3u; order <= 6u; ++order) {
        const auto order_text = std::to_string(order);
        set_optional_float(builder, "Xc-" + order_text + "d", values.xc_d[order]);
        set_optional_float(builder, "Xc-" + order_text + "dv", values.xc_dv[order]);
    }
    for (std::size_t order = 4u; order <= 6u; ++order) {
        const auto order_text = std::to_string(order);
        set_optional_float(builder, "Xpc-" + order_text + "d", values.xpc_d[order]);
        set_optional_float(builder, "Xpc-" + order_text + "dv", values.xpc_dv[order]);
    }
}

void set_information_content_values(
    DescriptorSetBuilder& builder,
    const MordredInformationContentValues& values) {
    for (std::size_t order = 0u; order <= 5u; ++order) {
        const auto order_text = std::to_string(order);
        set_optional_float(builder, "IC" + order_text, values.ic[order]);
        set_optional_float(builder, "TIC" + order_text, values.tic[order]);
        set_optional_float(builder, "SIC" + order_text, values.sic[order]);
        set_optional_float(builder, "BIC" + order_text, values.bic[order]);
        set_optional_float(builder, "CIC" + order_text, values.cic[order]);
        set_optional_float(builder, "MIC" + order_text, values.mic[order]);
        set_optional_float(builder, "ZMIC" + order_text, values.zmic[order]);
    }
}

void set_zagreb_values(DescriptorSetBuilder& builder, const MordredZagrebValues& values) {
    set_float(builder, "Zagreb1", values.zagreb1);
    set_float(builder, "Zagreb2", values.zagreb2);
    set_optional_float(builder, "mZagreb1", values.modified_zagreb1);
    set_float(builder, "mZagreb2", values.modified_zagreb2);
}

void set_abc_index_values(DescriptorSetBuilder& builder, const MordredABCIndexValues& values) {
    set_float(builder, "ABC", values.abc);
    set_float(builder, "ABCGG", values.abcgg);
}

void set_cpsa_charge_values(
    DescriptorSetBuilder& builder,
    const MordredCPSAChargeValues& values) {
    set_float(builder, "RNCG", values.rncg);
    set_float(builder, "RPCG", values.rpcg);
}

void set_autocorrelation_values(
    DescriptorSetBuilder& builder,
    const std::vector<MordredAutocorrelationValues>& value_sets) {
    for (const auto& values : value_sets) {
        for (std::size_t lag = 0u; lag < values.ats.size(); ++lag) {
            const auto lag_text = std::to_string(lag);
            set_optional_float(builder, "ATS" + lag_text + values.suffix, values.ats[lag]);
            set_optional_float(builder, "AATS" + lag_text + values.suffix, values.aats[lag]);
            set_optional_float(builder, "ATSC" + lag_text + values.suffix, values.atsc[lag]);
            set_optional_float(builder, "AATSC" + lag_text + values.suffix, values.aatsc[lag]);
            if (lag != 0u) {
                set_optional_float(builder, "MATS" + lag_text + values.suffix, values.mats[lag]);
                set_optional_float(builder, "GATS" + lag_text + values.suffix, values.gats[lag]);
            }
        }
    }
}

void set_molecular_distance_edge_values(
    DescriptorSetBuilder& builder,
    const MordredMolecularDistanceEdgeValues& values) {
    const auto set_family =
        [&builder](
            const std::string& prefix,
            const std::size_t max_degree,
            const std::array<std::array<std::optional<double>, 5>, 5>& family_values) {
            for (std::size_t valence1 = 1u; valence1 <= max_degree; ++valence1) {
                for (std::size_t valence2 = valence1; valence2 <= max_degree; ++valence2) {
                    set_optional_float(
                        builder,
                        prefix + "-" + std::to_string(valence1) + std::to_string(valence2),
                        family_values[valence1][valence2]);
                }
            }
        };

    for (std::size_t valence1 = 1u; valence1 <= 4u; ++valence1) {
        for (std::size_t valence2 = valence1; valence2 <= 4u; ++valence2) {
            set_optional_float(
                builder,
                "MDEC-" + std::to_string(valence1) + std::to_string(valence2),
                values.carbon[valence1][valence2]);
        }
    }
    set_family("MDEO", 2u, values.oxygen);
    set_family("MDEN", 3u, values.nitrogen);
}

void set_wiener_values(DescriptorSetBuilder& builder, const MordredWienerValues& values) {
    builder.Set("WPath", DescriptorValue::Int(values.wpath));
    builder.Set("WPol", DescriptorValue::Int(values.wpol));
}

void set_topological_index_values(
    DescriptorSetBuilder& builder,
    const MordredTopologicalIndexValues& values) {
    if (values.diameter.has_value()) {
        builder.Set("Diameter", DescriptorValue::Int(*values.diameter));
    }
    if (values.radius.has_value()) {
        builder.Set("Radius", DescriptorValue::Int(*values.radius));
    }
    set_optional_float(builder, "TopoShapeIndex", values.topo_shape_index);
    set_optional_float(builder, "PetitjeanIndex", values.petitjean_index);
}

void set_topological_charge_values(
    DescriptorSetBuilder& builder,
    const MordredTopologicalChargeValues& values) {
    for (std::size_t order = 1u; order <= 10u; ++order) {
        const auto order_text = std::to_string(order);
        set_float(builder, "GGI" + order_text, values.raw[order]);
        set_float(builder, "JGI" + order_text, values.mean[order]);
    }
    set_float(builder, "JGT10", values.global10);
}

void set_molecular_id_pair(
    DescriptorSetBuilder& builder,
    const std::string& qualifier,
    double value,
    std::size_t atom_count) {
    set_float(builder, "MID" + qualifier, value);
    set_float(builder, "AMID" + qualifier, value / static_cast<double>(atom_count));
}

void set_molecular_id_values(
    DescriptorSetBuilder& builder,
    const MordredMolecularIdValues& values) {
    set_molecular_id_pair(builder, "", values.any, values.atom_count);
    set_molecular_id_pair(builder, "_h", values.hetero, values.atom_count);
    set_molecular_id_pair(builder, "_C", values.carbon, values.atom_count);
    set_molecular_id_pair(builder, "_N", values.nitrogen, values.atom_count);
    set_molecular_id_pair(builder, "_O", values.oxygen, values.atom_count);
    set_molecular_id_pair(builder, "_X", values.halogen, values.atom_count);
}

void set_adjacency_matrix_eigenvalue_values(
    DescriptorSetBuilder& builder,
    const MordredMatrixEigenvalueValues& values) {
    set_float(builder, "SpAbs_A", values.spectral_absolute);
    set_float(builder, "SpMax_A", values.spectral_max);
    set_float(builder, "SpDiam_A", values.spectral_diameter);
    set_float(builder, "SpAD_A", values.spectral_absolute_deviation);
    set_float(builder, "SpMAD_A", values.spectral_mean_absolute_deviation);
    set_float(builder, "LogEE_A", values.log_estrada_like);
    set_float(builder, "VE1_A", values.eigenvector_coefficient_sum);
    set_float(builder, "VE2_A", values.eigenvector_coefficient_mean);
    set_float(builder, "VE3_A", values.eigenvector_coefficient_log);
    set_float(builder, "VR1_A", values.randic_eigenvector_sum);
    set_float(builder, "VR2_A", values.randic_eigenvector_mean);
    set_optional_float(builder, "VR3_A", values.randic_eigenvector_log);
}

void set_distance_matrix_eigenvalue_values(
    DescriptorSetBuilder& builder,
    const MordredMatrixEigenvalueValues& values) {
    set_float(builder, "SpAbs_D", values.spectral_absolute);
    set_float(builder, "SpMax_D", values.spectral_max);
    set_float(builder, "SpDiam_D", values.spectral_diameter);
    set_float(builder, "SpAD_D", values.spectral_absolute_deviation);
    set_float(builder, "SpMAD_D", values.spectral_mean_absolute_deviation);
    set_float(builder, "LogEE_D", values.log_estrada_like);
    set_float(builder, "VE1_D", values.eigenvector_coefficient_sum);
    set_float(builder, "VE2_D", values.eigenvector_coefficient_mean);
    set_float(builder, "VE3_D", values.eigenvector_coefficient_log);
    set_float(builder, "VR1_D", values.randic_eigenvector_sum);
    set_float(builder, "VR2_D", values.randic_eigenvector_mean);
    set_optional_float(builder, "VR3_D", values.randic_eigenvector_log);
}

void set_detour_matrix_values(
    DescriptorSetBuilder& builder,
    const MordredDetourMatrixValues& values) {
    set_float(builder, "SpAbs_Dt", values.matrix.spectral_absolute);
    set_float(builder, "SpMax_Dt", values.matrix.spectral_max);
    set_float(builder, "SpDiam_Dt", values.matrix.spectral_diameter);
    set_float(builder, "SpAD_Dt", values.matrix.spectral_absolute_deviation);
    set_float(builder, "SpMAD_Dt", values.matrix.spectral_mean_absolute_deviation);
    set_float(builder, "LogEE_Dt", values.matrix.log_estrada_like);
    set_float(builder, "SM1_Dt", values.matrix.spectral_moment);
    set_float(builder, "VE1_Dt", values.matrix.eigenvector_coefficient_sum);
    set_float(builder, "VE2_Dt", values.matrix.eigenvector_coefficient_mean);
    set_float(builder, "VE3_Dt", values.matrix.eigenvector_coefficient_log);
    set_float(builder, "VR1_Dt", values.matrix.randic_eigenvector_sum);
    set_float(builder, "VR2_Dt", values.matrix.randic_eigenvector_mean);
    set_optional_float(builder, "VR3_Dt", values.matrix.randic_eigenvector_log);
    builder.Set("DetourIndex", DescriptorValue::Int(values.detour_index));
}

std::string barysz_descriptor_name(const std::string& method, const char* suffix) {
    return method + "_Dz" + suffix;
}

void set_barysz_matrix_values(
    DescriptorSetBuilder& builder,
    const MordredMatrixEigenvalueValues& values,
    const char* suffix) {
    set_float(builder, barysz_descriptor_name("SpAbs", suffix), values.spectral_absolute);
    set_float(builder, barysz_descriptor_name("SpMax", suffix), values.spectral_max);
    set_float(builder, barysz_descriptor_name("SpDiam", suffix), values.spectral_diameter);
    set_float(
        builder,
        barysz_descriptor_name("SpAD", suffix),
        values.spectral_absolute_deviation);
    set_float(
        builder,
        barysz_descriptor_name("SpMAD", suffix),
        values.spectral_mean_absolute_deviation);
    set_float(builder, barysz_descriptor_name("LogEE", suffix), values.log_estrada_like);
    set_float(builder, barysz_descriptor_name("SM1", suffix), values.spectral_moment);
    set_float(
        builder,
        barysz_descriptor_name("VE1", suffix),
        values.eigenvector_coefficient_sum);
    set_float(
        builder,
        barysz_descriptor_name("VE2", suffix),
        values.eigenvector_coefficient_mean);
    set_float(
        builder,
        barysz_descriptor_name("VE3", suffix),
        values.eigenvector_coefficient_log);
    set_float(builder, barysz_descriptor_name("VR1", suffix), values.randic_eigenvector_sum);
    set_float(builder, barysz_descriptor_name("VR2", suffix), values.randic_eigenvector_mean);
    set_optional_float(
        builder,
        barysz_descriptor_name("VR3", suffix),
        values.randic_eigenvector_log);
}

void set_bcut_values(DescriptorSetBuilder& builder, const std::vector<MordredBCUTValues>& values) {
    for (const auto& value : values) {
        const std::string prefix = std::string("BCUT") + value.suffix + "-1";
        set_optional_float(builder, prefix + "h", value.high);
        set_optional_float(builder, prefix + "l", value.low);
    }
}

void set_eccentric_connectivity_index(DescriptorSetBuilder& builder, std::int64_t value) {
    builder.Set("ECIndex", DescriptorValue::Int(value));
}

void set_ring_count_summary(
    DescriptorSetBuilder& builder,
    const MordredRingCountSummary& values,
    const std::string& qualifier,
    std::size_t minimum_order = 3u) {
    set_int(builder, "n" + qualifier + "Ring", values.total);
    for (std::size_t order = minimum_order; order <= 12u; ++order) {
        set_int(
            builder,
            "n" + std::to_string(order) + qualifier + "Ring",
            values.by_size[order]);
    }
    set_int(builder, "nG12" + qualifier + "Ring", values.greater_or_equal_12);
}

void set_ring_count_values(DescriptorSetBuilder& builder, const MordredRingCountValues& values) {
    set_ring_count_summary(builder, values.all, "");
    set_ring_count_summary(builder, values.hetero, "H");
    set_ring_count_summary(builder, values.aromatic, "a");
    set_ring_count_summary(builder, values.aromatic_hetero, "aH");
    set_ring_count_summary(builder, values.aliphatic, "A");
    set_ring_count_summary(builder, values.aliphatic_hetero, "AH");
}

void set_fused_ring_count_values(DescriptorSetBuilder& builder, const MordredRingCountValues& values) {
    set_ring_count_summary(builder, values.all, "F", 4u);
    set_ring_count_summary(builder, values.hetero, "FH", 4u);
    set_ring_count_summary(builder, values.aromatic, "Fa", 4u);
    set_ring_count_summary(builder, values.aromatic_hetero, "FaH", 4u);
    set_ring_count_summary(builder, values.aliphatic, "FA", 4u);
    set_ring_count_summary(builder, values.aliphatic_hetero, "FAH", 4u);
}

void set_eta_values(DescriptorSetBuilder& builder, const MordredEtaValues& values) {
    for (const auto& value : values) {
        set_optional_float(builder, value.name, value.value);
    }
}

} // namespace

DescriptorSet MakeMordredDescriptors(const OEChem::OEMolBase& mol) {
    const auto values = compute_first_batch_values(mol);
    const auto additive_values = compute_additive_property_values(mol);
    const auto walk_count_values = compute_walk_count_values(mol);
    const auto heavy_atom_graph = build_mordred_heavy_atom_graph(mol);
    const auto path_count_values = compute_path_count_values(heavy_atom_graph);
    const auto chi_path_values = compute_chi_path_values(heavy_atom_graph);
    const auto chi_non_path_values = compute_chi_non_path_values(heavy_atom_graph);
    const auto zagreb_values = compute_zagreb_values(heavy_atom_graph);
    const auto vertex_adjacency_information =
        compute_vertex_adjacency_information(heavy_atom_graph);
    const auto heavy_atom_distances = compute_mordred_heavy_atom_distances(heavy_atom_graph);
    const auto balaban_j = compute_balaban_j(heavy_atom_graph, heavy_atom_distances);
    const auto bertz_ct = compute_bertz_ct(heavy_atom_graph);
    const auto topological_charge_values =
        compute_topological_charge_values(heavy_atom_graph, heavy_atom_distances);
    const auto molecular_id_values = compute_molecular_id_values(heavy_atom_graph);
    const auto adjacency_matrix_eigenvalue_values =
        compute_adjacency_matrix_eigenvalue_values(heavy_atom_graph);
    const auto distance_matrix_eigenvalue_values =
        compute_distance_matrix_eigenvalue_values(heavy_atom_graph, heavy_atom_distances);
    const auto detour_matrix_values = compute_detour_matrix_values(heavy_atom_graph);
    std::vector<std::pair<const char*, std::optional<MordredMatrixEigenvalueValues>>>
        barysz_matrix_values;
    barysz_matrix_values.reserve(mordred_barysz_matrix_properties().size());
    for (const auto& property : mordred_barysz_matrix_properties()) {
        barysz_matrix_values.emplace_back(
            property.suffix,
            compute_barysz_matrix_values(
                heavy_atom_graph,
                property.lookup,
                property.carbon_reference));
    }
    const auto bcut_values = compute_bcut_values(mol, heavy_atom_graph);
    const auto molecular_distance_edge_values =
        compute_molecular_distance_edge_values(heavy_atom_graph, heavy_atom_distances);
    const auto abc_index_values =
        compute_abc_index_values(heavy_atom_graph, heavy_atom_distances);
    const auto cpsa_charge_values = compute_cpsa_charge_values(mol);
    const auto autocorrelation_values = compute_autocorrelation_values(mol);
    const auto information_content_values = compute_information_content_values(mol);
    const auto wiener_values = compute_wiener_values(heavy_atom_distances);
    const auto topological_index_values = compute_topological_index_values(heavy_atom_distances);
    const auto ec_index =
        compute_eccentric_connectivity_index(heavy_atom_graph, heavy_atom_distances);
    const auto ring_count_values = compute_ring_count_value_sets(mol);
    const auto eta_values = compute_eta_values(mol, ring_count_values.base.all.total);
    const auto estate_values = compute_mordred_estate_values(mol);
    const auto filter_it_log_s = compute_filter_it_log_s(mol);
    const auto labute_asa_values = compute_labute_asa_values(mol);
    const auto peoe_vsa_values = compute_peoe_vsa_values(mol);
    const auto smr_vsa_values =
        labute_asa_values.has_value()
            ? compute_smr_vsa_values(mol, *labute_asa_values)
            : std::optional<std::array<double, 9>>{};
    const auto slogp_vsa_values =
        labute_asa_values.has_value()
            ? compute_slogp_vsa_values(mol, *labute_asa_values)
            : std::optional<std::array<double, 11>>{};
    const auto vsa_estate_values =
        labute_asa_values.has_value()
            ? compute_vsa_estate_values(mol, *labute_asa_values)
            : std::optional<std::array<double, 9>>{};
    const auto estate_vsa_values =
        labute_asa_values.has_value()
            ? compute_estate_vsa_values(mol, *labute_asa_values)
            : std::optional<std::array<double, 10>>{};
    const auto three_d_context = build_mordred_3d_context(mol);
    const auto low_count_3d_values = compute_low_count_3d_values(three_d_context);
    const auto morse_values = compute_mordred_morse_values(three_d_context);
    DescriptorSetBuilder builder(MordredDescriptorSchema());

    const auto all_atoms = values.heavy_atoms + values.hydrogens;
    const auto all_bonds = values.heavy_bonds + values.hydrogens;
    const auto all_single_bonds = values.single_heavy_bonds + values.hydrogens;
    const auto framework_atom_count = count_atoms(explicit_hydrogen_copy(mol));
    const auto framework_ratio = compute_framework_ratio(heavy_atom_graph, framework_atom_count);

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
    set_float(builder, "fragCpx", compute_fragment_complexity(values));
    set_int(builder, "nHBAcc", values.hbond_acceptors);
    set_int(builder, "nHBDon", values.hbond_donors);
    for (std::size_t index = 0u; index < mordred_estate_patterns().size(); ++index) {
        set_int(
            builder,
            mordred_estate_patterns()[index].descriptor_name,
            estate_values.counts[index]);
    }
    for (std::size_t index = 0u; index < mordred_estate_patterns().size(); ++index) {
        std::string descriptor_name = mordred_estate_patterns()[index].descriptor_name;
        descriptor_name[0] = 'S';
        set_float(builder, descriptor_name, estate_values.sums[index]);
    }
    for (std::size_t index = 0u; index < mordred_estate_patterns().size(); ++index) {
        std::string descriptor_name = mordred_estate_patterns()[index].descriptor_name;
        descriptor_name = "MAX" + descriptor_name.substr(1);
        set_optional_float(builder, descriptor_name, estate_values.maxima[index]);
    }
    for (std::size_t index = 0u; index < mordred_estate_patterns().size(); ++index) {
        std::string descriptor_name = mordred_estate_patterns()[index].descriptor_name;
        descriptor_name = "MIN" + descriptor_name.substr(1);
        set_optional_float(builder, descriptor_name, estate_values.minima[index]);
    }

    set_int(builder, "nRot", values.rotatable_bonds);
    if (values.heavy_bonds != 0u) {
        set_float(
            builder,
            "RotRatio",
            static_cast<double>(values.rotatable_bonds) / static_cast<double>(values.heavy_bonds));
    }
    set_float(builder, "SLogP", values.crippen_logp);
    set_float(builder, "SMR", values.crippen_mr);
    set_optional_float(
        builder,
        "LabuteASA",
        labute_asa_values.has_value() ? std::optional<double>{labute_asa_values->total}
                                      : std::nullopt);
    if (peoe_vsa_values.has_value()) {
        for (std::size_t index = 0u; index < peoe_vsa_values->size() - 1u; ++index) {
            set_float(
                builder,
                "PEOE_VSA" + std::to_string(index + 1u),
                (*peoe_vsa_values)[index]);
        }
    }
    if (smr_vsa_values.has_value()) {
        for (std::size_t index = 0u; index < smr_vsa_values->size(); ++index) {
            set_float(
                builder,
                "SMR_VSA" + std::to_string(index + 1u),
                (*smr_vsa_values)[index]);
        }
    }
    if (slogp_vsa_values.has_value()) {
        for (std::size_t index = 0u; index < slogp_vsa_values->size(); ++index) {
            set_float(
                builder,
                "SlogP_VSA" + std::to_string(index + 1u),
                (*slogp_vsa_values)[index]);
        }
    }
    if (estate_vsa_values.has_value()) {
        for (std::size_t index = 0u; index < estate_vsa_values->size(); ++index) {
            set_float(
                builder,
                "EState_VSA" + std::to_string(index + 1u),
                (*estate_vsa_values)[index]);
        }
    }
    if (vsa_estate_values.has_value()) {
        for (std::size_t index = 0u; index < vsa_estate_values->size(); ++index) {
            set_float(
                builder,
                "VSA_EState" + std::to_string(index + 1u),
                (*vsa_estate_values)[index]);
        }
    }
    if (low_count_3d_values.has_value()) {
        set_low_count_3d_values(builder, *low_count_3d_values);
    }
    set_mordred_morse_values(builder, morse_values);
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
    set_float(builder, "MWC01", walk_count_values.mwc[1]);
    set_float(builder, "MWC02", walk_count_values.mwc[2]);
    set_float(builder, "MWC03", walk_count_values.mwc[3]);
    set_float(builder, "MWC04", walk_count_values.mwc[4]);
    set_float(builder, "MWC05", walk_count_values.mwc[5]);
    set_float(builder, "MWC06", walk_count_values.mwc[6]);
    set_float(builder, "MWC07", walk_count_values.mwc[7]);
    set_float(builder, "MWC08", walk_count_values.mwc[8]);
    set_float(builder, "MWC09", walk_count_values.mwc[9]);
    set_float(builder, "MWC10", walk_count_values.mwc[10]);
    set_float(builder, "TMWC10", walk_count_values.total_mwc10);
    set_float(builder, "SRW02", walk_count_values.srw[2]);
    set_float(builder, "SRW03", walk_count_values.srw[3]);
    set_float(builder, "SRW04", walk_count_values.srw[4]);
    set_float(builder, "SRW05", walk_count_values.srw[5]);
    set_float(builder, "SRW06", walk_count_values.srw[6]);
    set_float(builder, "SRW07", walk_count_values.srw[7]);
    set_float(builder, "SRW08", walk_count_values.srw[8]);
    set_float(builder, "SRW09", walk_count_values.srw[9]);
    set_float(builder, "SRW10", walk_count_values.srw[10]);
    set_float(builder, "TSRW10", walk_count_values.total_srw10);
    set_int(builder, "MPC2", path_count_values.mpc[2]);
    set_int(builder, "MPC3", path_count_values.mpc[3]);
    set_int(builder, "MPC4", path_count_values.mpc[4]);
    set_int(builder, "MPC5", path_count_values.mpc[5]);
    set_int(builder, "MPC6", path_count_values.mpc[6]);
    set_int(builder, "MPC7", path_count_values.mpc[7]);
    set_int(builder, "MPC8", path_count_values.mpc[8]);
    set_int(builder, "MPC9", path_count_values.mpc[9]);
    set_int(builder, "MPC10", path_count_values.mpc[10]);
    set_int(builder, "TMPC10", path_count_values.total_mpc10);
    set_float(builder, "piPC1", std::log(path_count_values.pi_mpc[1] + 1.0));
    set_float(builder, "piPC2", std::log(path_count_values.pi_mpc[2] + 1.0));
    set_float(builder, "piPC3", std::log(path_count_values.pi_mpc[3] + 1.0));
    set_float(builder, "piPC4", std::log(path_count_values.pi_mpc[4] + 1.0));
    set_float(builder, "piPC5", std::log(path_count_values.pi_mpc[5] + 1.0));
    set_float(builder, "piPC6", std::log(path_count_values.pi_mpc[6] + 1.0));
    set_float(builder, "piPC7", std::log(path_count_values.pi_mpc[7] + 1.0));
    set_float(builder, "piPC8", std::log(path_count_values.pi_mpc[8] + 1.0));
    set_float(builder, "piPC9", std::log(path_count_values.pi_mpc[9] + 1.0));
    set_float(builder, "piPC10", std::log(path_count_values.pi_mpc[10] + 1.0));
    set_float(builder, "TpiPC10", std::log(path_count_values.total_pi_mpc10 + 1.0));
    set_optional_float(
        builder,
        "Kier1",
        kappa_shape_index(values.heavy_atoms, path_count_values, 1u));
    set_optional_float(
        builder,
        "Kier2",
        kappa_shape_index(values.heavy_atoms, path_count_values, 2u));
    set_optional_float(
        builder,
        "Kier3",
        kappa_shape_index(values.heavy_atoms, path_count_values, 3u));
    set_chi_path_values(builder, chi_path_values);
    set_chi_non_path_values(builder, chi_non_path_values);
    set_information_content_values(builder, information_content_values);
    set_zagreb_values(builder, zagreb_values);
    set_molecular_distance_edge_values(builder, molecular_distance_edge_values);
    set_abc_index_values(builder, abc_index_values);
    if (cpsa_charge_values.has_value()) {
        set_cpsa_charge_values(builder, *cpsa_charge_values);
    }
    set_autocorrelation_values(builder, autocorrelation_values);
    set_optional_float(builder, "BalabanJ", balaban_j);
    set_float(builder, "BertzCT", bertz_ct);
    set_optional_float(builder, "VAdjMat", vertex_adjacency_information);
    set_wiener_values(builder, wiener_values);
    set_topological_index_values(builder, topological_index_values);
    set_topological_charge_values(builder, topological_charge_values);
    if (molecular_id_values.has_value()) {
        set_molecular_id_values(builder, *molecular_id_values);
    }
    if (adjacency_matrix_eigenvalue_values.has_value()) {
        set_adjacency_matrix_eigenvalue_values(builder, *adjacency_matrix_eigenvalue_values);
    }
    if (distance_matrix_eigenvalue_values.has_value()) {
        set_distance_matrix_eigenvalue_values(builder, *distance_matrix_eigenvalue_values);
    }
    if (detour_matrix_values.has_value()) {
        set_detour_matrix_values(builder, *detour_matrix_values);
    }
    for (const auto& [suffix, property_values] : barysz_matrix_values) {
        if (property_values.has_value()) {
            set_barysz_matrix_values(builder, *property_values, suffix);
        }
    }
    set_bcut_values(builder, bcut_values);
    set_eccentric_connectivity_index(builder, ec_index);
    set_ring_count_values(builder, ring_count_values.base);
    set_fused_ring_count_values(builder, ring_count_values.fused);
    if (eta_values.has_value()) {
        set_eta_values(builder, *eta_values);
    }
    set_optional_float(builder, "fMF", framework_ratio);
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
    set_optional_float(builder, "FilterItLogS", filter_it_log_s);

    return builder.Build();
}

namespace test {

DescriptorSet MakeMordredDetourDescriptorsForTesting(
    const OEChem::OEMolBase& mol,
    std::uint64_t max_search_operations) {
    const auto heavy_atom_graph = build_mordred_heavy_atom_graph(mol);
    const auto detour_matrix_values =
        compute_detour_matrix_values(heavy_atom_graph, max_search_operations);
    DescriptorSetBuilder builder(MordredDescriptorSchema());
    if (detour_matrix_values.has_value()) {
        set_detour_matrix_values(builder, *detour_matrix_values);
    }
    return builder.Build();
}

} // namespace test

} // namespace OEFP
