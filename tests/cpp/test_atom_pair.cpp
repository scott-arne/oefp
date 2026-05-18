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

TEST(AtomPairGeneratorTest, FingerprintMatchesFunctionalApiForDefaultOptions) {
    const auto mol = mol_from_smiles("CC(=O)Oc1ccccc1C(=O)O");
    const AtomPairOptions options;
    const AtomPairGenerator generator(options);

    const auto generated = generator.Fingerprint(mol);
    const auto functional = MakeAtomPairFingerprint(mol, options);

    EXPECT_EQ(generated.Spec(), functional.Spec());
    EXPECT_EQ(generated.Words(), functional.Words());
    EXPECT_EQ(generated.CountOnBits(), functional.CountOnBits());
}

TEST(AtomPairGeneratorTest, FingerprintMatchesFunctionalApiForNonDefaultOptions) {
    const auto mol = mol_from_smiles("c1ccc(O)cc1");
    AtomPairOptions options;
    options.min_distance = 1;
    options.max_distance = 4;
    options.num_bits = 512;
    options.count_simulation = false;

    const AtomPairGenerator generator(options);

    const auto generated = generator.Fingerprint(mol);
    const auto functional = MakeAtomPairFingerprint(mol, options);

    EXPECT_EQ(generated.SizeBits(), 512u);
    EXPECT_EQ(generated.Spec(), functional.Spec());
    EXPECT_EQ(generated.Words(), functional.Words());
}

TEST(AtomPairGeneratorTest, CountSimulationMatchesFunctionalApi) {
    const auto mol = mol_from_smiles("CCC(CC)CO");
    AtomPairOptions options;
    options.num_bits = 256;
    options.count_simulation = true;
    options.count_bounds = {1u, 2u, 4u};

    const AtomPairGenerator generator(options);

    const auto generated = generator.Fingerprint(mol);
    const auto functional = MakeAtomPairFingerprint(mol, options);

    EXPECT_EQ(generated.Spec(), functional.Spec());
    EXPECT_EQ(generated.Words(), functional.Words());
}

TEST(AtomPairGeneratorTest, ProfileReportsStageTimingsAndGeneratedBits) {
    const auto mol = mol_from_smiles("CC(=O)Oc1ccccc1C(=O)O");
    AtomPairOptions options;
    options.num_bits = 256;

    const auto profile = ProfileAtomPairFingerprint(mol, options);
    const auto fingerprint = MakeAtomPairFingerprint(mol, options);

    EXPECT_EQ(profile.atom_count, mol.NumAtoms());
    EXPECT_GT(profile.event_count, 0u);
    EXPECT_EQ(profile.on_bit_count, fingerprint.CountOnBits());
    EXPECT_GE(profile.molecule_preparation_seconds, 0.0);
    EXPECT_GE(profile.graph_seconds, 0.0);
    EXPECT_GE(profile.atom_code_seconds, 0.0);
    EXPECT_GE(profile.distance_seconds, 0.0);
    EXPECT_GE(profile.pair_enumeration_seconds, 0.0);
    EXPECT_GE(profile.bit_folding_seconds, 0.0);
    EXPECT_GT(profile.TotalSeconds(), 0.0);
}

TEST(AtomPairGeneratorTest, ConstructorRejectsInvalidOptions) {
    AtomPairOptions zero_bits;
    zero_bits.num_bits = 0;
    EXPECT_THROW(static_cast<void>(AtomPairGenerator{zero_bits}), std::invalid_argument);

    AtomPairOptions chiral;
    chiral.use_chirality = true;
    EXPECT_THROW(static_cast<void>(AtomPairGenerator{chiral}), std::invalid_argument);

    AtomPairOptions empty_count_bounds;
    empty_count_bounds.count_bounds.clear();
    EXPECT_THROW(static_cast<void>(AtomPairGenerator{empty_count_bounds}), std::invalid_argument);
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

TEST(AtomPairTest, GeneratedSparseCountFingerprintCarriesStrictAtomPairSpec) {
    const auto mol = mol_from_smiles("CCO");
    AtomPairOptions options;
    options.min_distance = 1;
    options.max_distance = 2;

    const auto fp = MakeAtomPairSparseCountFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 8388608u);
    EXPECT_EQ(spec.size_bits, 8388608u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Counted);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "AtomPair");
    EXPECT_EQ(spec.source_version, "AtomPair-1.1.0");
    EXPECT_EQ(
        spec.parameters,
        "min_distance=1;max_distance=2;use_chirality=false;"
        "use_2d=true;output=sparse_count");
}

TEST(AtomPairTest, GeneratesNonEmptySparseCountFingerprintForSimpleMolecule) {
    const auto mol = mol_from_smiles("CCO");

    const auto fp = MakeAtomPairSparseCountFingerprint(mol);

    EXPECT_GT(fp.NonzeroCount(), 0u);
    EXPECT_EQ(fp.TotalCount(), 3u);
}

TEST(AtomPairTest, GeneratedDescriptorsCarryStrictAtomPairSpec) {
    const auto mol = mol_from_smiles("CCO");
    AtomPairOptions options;
    options.min_distance = 1;
    options.max_distance = 2;
    options.num_bits = 128;
    options.count_simulation = false;

    const auto descriptors = MakeAtomPairDescriptors(mol, options);
    const auto& spec = descriptors.Spec();

    EXPECT_EQ(descriptors.ValueType(), DescriptorValueType::String);
    EXPECT_EQ(spec.value_type, DescriptorValueType::String);
    EXPECT_EQ(spec.source_name, "OEFP");
    EXPECT_EQ(spec.source_type, "AtomPair");
    EXPECT_EQ(spec.source_version, "AtomPair-1.1.0");
    EXPECT_EQ(
        spec.parameters,
        "min_distance=1;max_distance=2;use_chirality=false;"
        "use_2d=true;output=descriptors");
}

TEST(AtomPairTest, GeneratedDescriptorsUseRawAtomPairKeysAndCounts) {
    const auto ethane = mol_from_smiles("CC");
    const auto ethane_descriptors = MakeAtomPairDescriptors(ethane);

    EXPECT_EQ(ethane_descriptors.TotalCount(), 1u);
    EXPECT_EQ(ethane_descriptors.StringKeys(), std::vector<std::string>({"33_1_33"}));
    EXPECT_EQ(ethane_descriptors.Counts(), std::vector<std::uint32_t>({1u}));

    const auto ethanol = mol_from_smiles("CCO");
    const auto ethanol_descriptors = MakeAtomPairDescriptors(ethanol);

    EXPECT_EQ(ethanol_descriptors.TotalCount(), 3u);
    EXPECT_EQ(
        ethanol_descriptors.StringKeys(),
        std::vector<std::string>({"33_1_34", "33_2_97", "34_1_97"}));
    EXPECT_EQ(
        ethanol_descriptors.Counts(),
        std::vector<std::uint32_t>({1u, 1u, 1u}));
}

TEST(AtomPairTest, DescriptorDistanceOptionsFilterRawAtomPairs) {
    const auto mol = mol_from_smiles("CCO");

    AtomPairOptions options;
    options.min_distance = 1;
    options.max_distance = 1;

    const auto descriptors = MakeAtomPairDescriptors(mol, options);

    EXPECT_EQ(descriptors.TotalCount(), 2u);
    EXPECT_EQ(
        descriptors.StringKeys(),
        std::vector<std::string>({"33_1_34", "34_1_97"}));
    EXPECT_EQ(
        descriptors.Counts(),
        std::vector<std::uint32_t>({1u, 1u}));
}

TEST(AtomPairTest, GeneratedDescriptorsCoverAromaticityValenceAndBranching) {
    const auto benzene_descriptors = MakeAtomPairDescriptors(mol_from_smiles("c1ccccc1"));
    EXPECT_EQ(benzene_descriptors.TotalCount(), 15u);
    EXPECT_EQ(
        benzene_descriptors.StringKeys(),
        std::vector<std::string>({"42_1_42", "42_2_42", "42_3_42"}));
    EXPECT_EQ(
        benzene_descriptors.Counts(),
        std::vector<std::uint32_t>({6u, 6u, 3u}));

    const auto pyridine_descriptors = MakeAtomPairDescriptors(mol_from_smiles("c1ccncc1"));
    EXPECT_EQ(pyridine_descriptors.TotalCount(), 15u);
    EXPECT_EQ(
        pyridine_descriptors.StringKeys(),
        std::vector<std::string>(
            {"42_1_42", "42_1_74", "42_2_42", "42_2_74", "42_3_42", "42_3_74"}));
    EXPECT_EQ(
        pyridine_descriptors.Counts(),
        std::vector<std::uint32_t>({4u, 2u, 4u, 2u, 2u, 1u}));

    const auto acetaldehyde_descriptors = MakeAtomPairDescriptors(mol_from_smiles("CC=O"));
    EXPECT_EQ(acetaldehyde_descriptors.TotalCount(), 3u);
    EXPECT_EQ(
        acetaldehyde_descriptors.StringKeys(),
        std::vector<std::string>({"33_1_42", "33_2_105", "42_1_105"}));
    EXPECT_EQ(
        acetaldehyde_descriptors.Counts(),
        std::vector<std::uint32_t>({1u, 1u, 1u}));

    const auto acetonitrile_descriptors = MakeAtomPairDescriptors(mol_from_smiles("CC#N"));
    EXPECT_EQ(acetonitrile_descriptors.TotalCount(), 3u);
    EXPECT_EQ(
        acetonitrile_descriptors.StringKeys(),
        std::vector<std::string>({"33_1_50", "33_2_81", "50_1_81"}));
    EXPECT_EQ(
        acetonitrile_descriptors.Counts(),
        std::vector<std::uint32_t>({1u, 1u, 1u}));

    const auto tertiary_chloride_descriptors =
        MakeAtomPairDescriptors(mol_from_smiles("CC(C)(C)Cl"));
    EXPECT_EQ(tertiary_chloride_descriptors.TotalCount(), 10u);
    EXPECT_EQ(
        tertiary_chloride_descriptors.StringKeys(),
        std::vector<std::string>(
            {"33_1_36", "33_2_257", "33_2_33", "36_1_257"}));
    EXPECT_EQ(
        tertiary_chloride_descriptors.Counts(),
        std::vector<std::uint32_t>({3u, 3u, 3u, 1u}));
}

TEST(AtomPairTest, DescriptorGenerationRejectsUnsupportedOptions) {
    const auto mol = mol_from_smiles("CCO");

    AtomPairOptions inverted_distances;
    inverted_distances.min_distance = 3;
    inverted_distances.max_distance = 2;
    EXPECT_THROW(MakeAtomPairDescriptors(mol, inverted_distances), std::invalid_argument);

    AtomPairOptions too_long_distance;
    too_long_distance.max_distance = 31;
    EXPECT_THROW(MakeAtomPairDescriptors(mol, too_long_distance), std::invalid_argument);

    AtomPairOptions chiral;
    chiral.use_chirality = true;
    EXPECT_THROW(MakeAtomPairDescriptors(mol, chiral), std::invalid_argument);

    AtomPairOptions three_dimensional;
    three_dimensional.use_2d = false;
    EXPECT_THROW(MakeAtomPairDescriptors(mol, three_dimensional), std::invalid_argument);
}

} // namespace test
} // namespace OEFP
