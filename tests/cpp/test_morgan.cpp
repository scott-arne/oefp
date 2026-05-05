#include <gtest/gtest.h>

#include "oefp/batch.h"
#include "oefp/morgan.h"

#include <oechem.h>

#include <stdexcept>
#include <string>

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
}

TEST(MorganTest, RejectsInvalidOptions) {
    const auto mol = mol_from_smiles("CCO");

    MorganOptions zero_bits;
    zero_bits.num_bits = 0;
    EXPECT_THROW(MakeMorganFingerprint(mol, zero_bits), std::invalid_argument);

    MorganOptions chiral;
    chiral.use_chirality = true;
    EXPECT_THROW(MakeMorganFingerprint(mol, chiral), std::invalid_argument);
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
        "include_redundant_environments=true");
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
}

TEST(MorganTest, GeneratesNonEmptyFingerprintForSimpleMolecule) {
    const auto mol = mol_from_smiles("CCO");

    const auto fp = MakeMorganFingerprint(mol);

    EXPECT_GT(fp.CountOnBits(), 0u);
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

} // namespace test
} // namespace OEFP
