#include <gtest/gtest.h>

#include "oefp/atom_pair.h"

#include <oechem.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace OEFP {
namespace test {
namespace {

OEChem::OEGraphMol mol_from_smiles(const std::string& smiles) {
    OEChem::OEGraphMol mol;
    if (!OEChem::OESmilesToMol(mol, smiles)) {
        throw std::runtime_error("Failed to parse test SMILES: " + smiles);
    }
    return mol;
}

} // namespace

TEST(AtomPairTest, DefaultOptionsMatchRdkitGeneratorDefaults) {
    const AtomPairOptions options;

    EXPECT_EQ(options.min_distance, 1u);
    EXPECT_EQ(options.max_distance, 30u);
    EXPECT_EQ(options.num_bits, 2048u);
    EXPECT_FALSE(options.use_chirality);
    EXPECT_TRUE(options.use_2d);
    EXPECT_TRUE(options.count_simulation);
    EXPECT_EQ(options.count_bounds, std::vector<std::uint32_t>({1u, 2u, 4u, 8u}));
}

TEST(AtomPairTest, RejectsInvalidOptions) {
    const auto mol = mol_from_smiles("CCO");

    AtomPairOptions zero_bits;
    zero_bits.num_bits = 0;
    EXPECT_THROW(MakeAtomPairFingerprint(mol, zero_bits), std::invalid_argument);

    AtomPairOptions inverted_distances;
    inverted_distances.min_distance = 3;
    inverted_distances.max_distance = 2;
    EXPECT_THROW(MakeAtomPairFingerprint(mol, inverted_distances), std::invalid_argument);

    AtomPairOptions too_long_distance;
    too_long_distance.max_distance = 31;
    EXPECT_THROW(MakeAtomPairFingerprint(mol, too_long_distance), std::invalid_argument);

    AtomPairOptions chiral;
    chiral.use_chirality = true;
    EXPECT_THROW(MakeAtomPairFingerprint(mol, chiral), std::invalid_argument);

    AtomPairOptions three_dimensional;
    three_dimensional.use_2d = false;
    EXPECT_THROW(MakeAtomPairFingerprint(mol, three_dimensional), std::invalid_argument);

    AtomPairOptions empty_count_bounds;
    empty_count_bounds.count_bounds.clear();
    EXPECT_THROW(MakeAtomPairFingerprint(mol, empty_count_bounds), std::invalid_argument);

    AtomPairOptions too_many_count_bounds;
    too_many_count_bounds.num_bits = 4;
    too_many_count_bounds.count_bounds = {1u, 2u, 4u, 8u};
    EXPECT_THROW(MakeAtomPairFingerprint(mol, too_many_count_bounds), std::invalid_argument);
}

TEST(AtomPairTest, GeneratedFingerprintCarriesStrictAtomPairSpec) {
    const auto mol = mol_from_smiles("CCO");
    AtomPairOptions options;
    options.min_distance = 1;
    options.max_distance = 2;
    options.num_bits = 128;
    options.count_simulation = false;

    const auto fp = MakeAtomPairFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 128u);
    EXPECT_EQ(spec.size_bits, 128u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Binary);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "AtomPair");
    EXPECT_EQ(spec.source_version, "AtomPair-1.1.0");
    EXPECT_EQ(
        spec.parameters,
        "min_distance=1;max_distance=2;num_bits=128;use_chirality=false;"
        "use_2d=true;count_simulation=false;count_bounds=1,2,4,8");
}

TEST(AtomPairTest, GeneratesNonEmptyFingerprintForSimpleMolecule) {
    const auto mol = mol_from_smiles("CCO");

    const auto fp = MakeAtomPairFingerprint(mol);

    EXPECT_GT(fp.CountOnBits(), 0u);
}

TEST(AtomPairTest, GeneratedCountFingerprintCarriesStrictAtomPairSpec) {
    const auto mol = mol_from_smiles("CCO");
    AtomPairOptions options;
    options.min_distance = 1;
    options.max_distance = 2;
    options.num_bits = 128;

    const auto fp = MakeAtomPairCountFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 128u);
    EXPECT_EQ(spec.size_bits, 128u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Counted);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "AtomPair");
    EXPECT_EQ(spec.source_version, "AtomPair-1.1.0");
    EXPECT_EQ(
        spec.parameters,
        "min_distance=1;max_distance=2;num_bits=128;use_chirality=false;"
        "use_2d=true;count_simulation=false;count_bounds=1,2,4,8");
}

TEST(AtomPairTest, GeneratesNonEmptyCountFingerprintForSimpleMolecule) {
    const auto mol = mol_from_smiles("CCO");

    const auto fp = MakeAtomPairCountFingerprint(mol);

    EXPECT_GT(fp.NonzeroCount(), 0u);
    EXPECT_EQ(fp.TotalCount(), 3u);
}

TEST(AtomPairTest, GeneratedSparseFingerprintCarriesStrictAtomPairSpec) {
    const auto mol = mol_from_smiles("CCO");
    AtomPairOptions options;
    options.min_distance = 1;
    options.max_distance = 2;
    options.count_simulation = false;

    const auto fp = MakeAtomPairSparseFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 8388608u);
    EXPECT_EQ(spec.size_bits, 8388608u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Binary);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "AtomPair");
    EXPECT_EQ(spec.source_version, "AtomPair-1.1.0");
    EXPECT_EQ(
        spec.parameters,
        "min_distance=1;max_distance=2;use_chirality=false;"
        "use_2d=true;count_simulation=false;count_bounds=1,2,4,8;"
        "output=sparse_binary");
}

TEST(AtomPairTest, GeneratesNonEmptySparseFingerprintForSimpleMolecule) {
    const auto mol = mol_from_smiles("CCO");

    const auto fp = MakeAtomPairSparseFingerprint(mol);

    EXPECT_GT(fp.CountOnBits(), 0u);
}

} // namespace test
} // namespace OEFP
