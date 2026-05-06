#include <gtest/gtest.h>

#include "oefp/batch.h"
#include "oefp/morgan.h"

#include <oechem.h>

#include <algorithm>
#include <cstdint>
#include <limits>
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

TEST(MorganTest, DefaultOptionsMatchExpectedPublicDefaults) {
    const MorganOptions options;

    EXPECT_EQ(options.radius, 2u);
    EXPECT_EQ(options.num_bits, 2048u);
    EXPECT_FALSE(options.use_chirality);
    EXPECT_TRUE(options.use_bond_types);
    EXPECT_FALSE(options.only_nonzero_invariants);
    EXPECT_TRUE(options.include_ring_membership);
    EXPECT_FALSE(options.include_redundant_environments);
    EXPECT_FALSE(options.count_simulation);
    EXPECT_EQ(options.count_bounds, std::vector<std::uint32_t>({1u, 2u, 4u, 8u}));
}

TEST(MorganTest, RejectsInvalidOptions) {
    const auto mol = mol_from_smiles("CCO");

    MorganOptions zero_bits;
    zero_bits.num_bits = 0;
    EXPECT_THROW(MakeMorganFingerprint(mol, zero_bits), std::invalid_argument);

    MorganOptions chiral;
    chiral.use_chirality = true;
    EXPECT_THROW(MakeMorganFingerprint(mol, chiral), std::invalid_argument);

    MorganOptions empty_count_bounds;
    empty_count_bounds.count_simulation = true;
    empty_count_bounds.count_bounds.clear();
    EXPECT_THROW(MakeMorganFingerprint(mol, empty_count_bounds), std::invalid_argument);

    MorganOptions too_many_count_bounds;
    too_many_count_bounds.count_simulation = true;
    too_many_count_bounds.num_bits = 4;
    too_many_count_bounds.count_bounds = {1u, 2u, 4u, 8u};
    EXPECT_THROW(MakeMorganFingerprint(mol, too_many_count_bounds), std::invalid_argument);

    MorganOptions simulated_count_fp;
    simulated_count_fp.count_simulation = true;
    EXPECT_THROW(MakeMorganCountFingerprint(mol, simulated_count_fp), std::invalid_argument);
}

TEST(MorganTest, GeneratedFingerprintCarriesStrictMorganSpec) {
    const auto mol = mol_from_smiles("c1ccccc1");
    MorganOptions options;
    options.radius = 1;
    options.num_bits = 128;
    options.use_bond_types = false;
    options.only_nonzero_invariants = true;
    options.include_ring_membership = false;
    options.include_redundant_environments = true;
    options.count_simulation = true;
    options.count_bounds = {1u, 3u, 7u};

    const auto fp = MakeMorganFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 128u);
    EXPECT_EQ(spec.size_bits, 128u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Binary);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "Morgan");
    EXPECT_EQ(spec.source_version, "Morgan-2026.03.1");
    EXPECT_EQ(
        spec.parameters,
        "radius=1;num_bits=128;use_chirality=false;use_bond_types=false;"
        "only_nonzero_invariants=true;include_ring_membership=false;"
        "include_redundant_environments=true;count_simulation=true;"
        "count_bounds=1,3,7");
}

TEST(MorganTest, MatchingOptionsBatchAndDifferentOptionsReject) {
    const auto mol_a = mol_from_smiles("CCO");
    const auto mol_b = mol_from_smiles("CCN");

    MorganOptions options;
    options.num_bits = 128;
    const auto fp_a = MakeMorganFingerprint(mol_a, options);
    const auto fp_b = MakeMorganFingerprint(mol_b, options);

    const auto batch = OEFPBatch::FromFingerprints({fp_a, fp_b});
    EXPECT_EQ(batch.Size(), 2u);
    EXPECT_EQ(batch.Spec(), fp_a.Spec());

    MorganOptions different_radius = options;
    different_radius.radius = 1;
    EXPECT_THROW(
        OEFPBatch::FromFingerprints(
            {fp_a, MakeMorganFingerprint(mol_a, different_radius)}),
        std::invalid_argument);

    MorganOptions different_bond_types = options;
    different_bond_types.use_bond_types = false;
    EXPECT_THROW(
        OEFPBatch::FromFingerprints(
            {fp_a, MakeMorganFingerprint(mol_a, different_bond_types)}),
        std::invalid_argument);

    MorganOptions different_nonzero = options;
    different_nonzero.only_nonzero_invariants = true;
    EXPECT_THROW(
        OEFPBatch::FromFingerprints(
            {fp_a, MakeMorganFingerprint(mol_a, different_nonzero)}),
        std::invalid_argument);

    MorganOptions different_ring_membership = options;
    different_ring_membership.include_ring_membership = false;
    EXPECT_THROW(
        OEFPBatch::FromFingerprints(
            {fp_a, MakeMorganFingerprint(mol_a, different_ring_membership)}),
        std::invalid_argument);

    MorganOptions different_redundant_environments = options;
    different_redundant_environments.include_redundant_environments = true;
    EXPECT_THROW(
        OEFPBatch::FromFingerprints(
            {fp_a, MakeMorganFingerprint(mol_a, different_redundant_environments)}),
        std::invalid_argument);

    MorganOptions different_count_simulation = options;
    different_count_simulation.count_simulation = true;
    EXPECT_THROW(
        OEFPBatch::FromFingerprints(
            {fp_a, MakeMorganFingerprint(mol_a, different_count_simulation)}),
        std::invalid_argument);

    MorganOptions different_count_bounds = options;
    different_count_bounds.count_bounds = {1u, 4u, 16u};
    EXPECT_THROW(
        OEFPBatch::FromFingerprints(
            {fp_a, MakeMorganFingerprint(mol_a, different_count_bounds)}),
        std::invalid_argument);
}

TEST(MorganTest, GeneratesNonEmptyFingerprintForSimpleMolecule) {
    const auto mol = mol_from_smiles("CCO");

    const auto fp = MakeMorganFingerprint(mol);

    EXPECT_GT(fp.CountOnBits(), 0u);
}

TEST(MorganTest, GeneratesFoldedBinaryFingerprintWithBitMappings) {
    const auto mol = mol_from_smiles("CCC(CC)CO");
    MorganOptions options;
    options.radius = 1;
    options.num_bits = 2048;

    const auto result = MakeMorganFingerprintWithMapping(mol, options);
    const auto fp = result.Fingerprint();
    const auto mappings = result.Mapping();
    const auto bit_ids = mappings.BitIds(0);

    EXPECT_EQ(fp.CountOnBits(), bit_ids.size());
    for (const auto bit_id : bit_ids) {
        EXPECT_TRUE(fp.TestBit(bit_id));
        EXPECT_FALSE(mappings.EnvironmentsForBit(0, bit_id).empty());
    }

    const auto environments = mappings.EnvironmentsForBit(0, 80);
    ASSERT_EQ(environments.size(), 3u);
    EXPECT_EQ(environments[0].AtomId(), 1u);
    EXPECT_EQ(environments[0].Radius(), 0u);
    EXPECT_EQ(environments[1].AtomId(), 3u);
    EXPECT_EQ(environments[1].Radius(), 0u);
    EXPECT_EQ(environments[2].AtomId(), 5u);
    EXPECT_EQ(environments[2].Radius(), 0u);
}

TEST(MorganTest, GeneratesSparseBinaryFingerprintWithRawBitMappings) {
    const auto mol = mol_from_smiles("CCC(CC)CO");
    MorganOptions options;
    options.radius = 1;

    const auto result = MakeMorganSparseFingerprintWithMapping(mol, options);
    const auto fp = result.Fingerprint();
    const auto mappings = result.Mapping();
    const auto bit_ids = mappings.BitIds(0);

    EXPECT_EQ(fp.CountOnBits(), bit_ids.size());
    for (const auto bit_id : bit_ids) {
        EXPECT_NE(std::find(fp.Indices().begin(), fp.Indices().end(), bit_id), fp.Indices().end());
        EXPECT_FALSE(mappings.EnvironmentsForBit(0, bit_id).empty());
    }
    EXPECT_FALSE(mappings.EnvironmentsForBit(0, 2245384272u).empty());
}

TEST(MorganTest, CountSimulationSetsThresholdBits) {
    const auto mol = mol_from_smiles("CC");
    MorganOptions options;
    options.radius = 0;
    options.num_bits = 16;
    options.count_simulation = true;
    options.count_bounds = {1u, 2u, 4u};

    const auto fp = MakeMorganFingerprint(mol, options);
    const auto bit_count = options.count_bounds.size();

    EXPECT_EQ(fp.CountOnBits(), 2u);

    std::vector<std::uint64_t> on_bits;
    for (std::uint64_t bit = 0; bit < fp.SizeBits(); ++bit) {
        if (fp.TestBit(bit)) {
            on_bits.push_back(bit);
        }
    }
    ASSERT_EQ(on_bits.size(), 2u);
    EXPECT_EQ(on_bits[0] % bit_count, 0u);
    EXPECT_EQ(on_bits[1], on_bits[0] + 1u);
    EXPECT_FALSE(fp.TestBit(on_bits[0] + 2u));
}

TEST(MorganTest, GeneratesCountFingerprintWithStrictMorganSpec) {
    const auto mol = mol_from_smiles("CCO");
    MorganOptions options;
    options.num_bits = 128;

    const auto fp = MakeMorganCountFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 128u);
    EXPECT_EQ(spec.size_bits, 128u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Counted);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "Morgan");
    EXPECT_EQ(spec.source_version, "Morgan-2026.03.1");
    EXPECT_NE(spec.parameters.find("num_bits=128"), std::string::npos);
    EXPECT_GT(fp.NonzeroCount(), 0u);
    EXPECT_GE(fp.TotalCount(), fp.NonzeroCount());
}

TEST(MorganTest, GeneratesSparseCountFingerprintWithRawIdentifiers) {
    const auto mol = mol_from_smiles("CCO");
    MorganOptions options;
    options.num_bits = 128;

    const auto fp = MakeMorganSparseCountFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(spec.size_bits, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(spec.value_type, FingerprintValueType::Counted);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "Morgan");
    EXPECT_EQ(spec.source_version, "Morgan-2026.03.1");
    EXPECT_EQ(
        spec.parameters,
        "radius=2;use_chirality=false;use_bond_types=true;"
        "only_nonzero_invariants=false;include_ring_membership=true;"
        "include_redundant_environments=false;output=sparse_count");
    EXPECT_GT(fp.NonzeroCount(), 0u);
    EXPECT_GE(fp.TotalCount(), fp.NonzeroCount());
    EXPECT_GT(fp.Indices().back(), options.num_bits);
}

TEST(MorganTest, GeneratesSparseFingerprintWithRawIdentifiers) {
    const auto mol = mol_from_smiles("CCO");
    MorganOptions options;
    options.num_bits = 128;

    const auto fp = MakeMorganSparseFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(spec.size_bits, std::numeric_limits<std::uint64_t>::max());
    EXPECT_EQ(spec.value_type, FingerprintValueType::Binary);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "Morgan");
    EXPECT_EQ(spec.source_version, "Morgan-2026.03.1");
    EXPECT_EQ(
        spec.parameters,
        "radius=2;use_chirality=false;use_bond_types=true;"
        "only_nonzero_invariants=false;include_ring_membership=true;"
        "include_redundant_environments=false;output=sparse_binary");
    EXPECT_GT(fp.CountOnBits(), 0u);
    EXPECT_GT(fp.Indices().back(), options.num_bits);
}

} // namespace test
} // namespace OEFP
