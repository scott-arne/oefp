#include <gtest/gtest.h>

#include "oefp/topological_torsions.h"

#include <oechem.h>

#include <cstdint>
#include <numeric>
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

const CountedStringKeyValues& torsion_values(const DescriptorSet& descriptors) {
    return descriptors.Value("topological_torsions").CountedStringKeys();
}

std::uint64_t total_count(const CountedStringKeyValues& values) {
    return std::accumulate(values.counts.begin(), values.counts.end(), std::uint64_t{0});
}

} // namespace

TEST(TopologicalTorsionsTest, DefaultOptionsMatchRdkitGeneratorDefaults) {
    const TopologicalTorsionsOptions options;

    EXPECT_EQ(options.torsion_atom_count, 4u);
    EXPECT_EQ(options.num_bits, 2048u);
    EXPECT_FALSE(options.use_chirality);
    EXPECT_TRUE(options.count_simulation);
    EXPECT_EQ(options.count_bounds, std::vector<std::uint32_t>({1u, 2u, 4u, 8u}));
}

TEST(TopologicalTorsionsTest, RejectsInvalidOptions) {
    const auto mol = mol_from_smiles("CCCC");

    TopologicalTorsionsOptions zero_bits;
    zero_bits.num_bits = 0;
    EXPECT_THROW(MakeTopologicalTorsionsFingerprint(mol, zero_bits), std::invalid_argument);

    TopologicalTorsionsOptions zero_atoms;
    zero_atoms.torsion_atom_count = 0;
    EXPECT_THROW(MakeTopologicalTorsionsFingerprint(mol, zero_atoms), std::invalid_argument);

    TopologicalTorsionsOptions too_many_atoms;
    too_many_atoms.torsion_atom_count = 8;
    EXPECT_THROW(MakeTopologicalTorsionsFingerprint(mol, too_many_atoms), std::invalid_argument);

    // Chirality widens each atom code by two bits, so it can only be combined
    // with torsion lengths whose total raw code stays within 64 bits.
    TopologicalTorsionsOptions chiral_overflow;
    chiral_overflow.use_chirality = true;
    chiral_overflow.torsion_atom_count = 6;
    EXPECT_THROW(
        MakeTopologicalTorsionsSparseCountFingerprint(mol, chiral_overflow),
        std::invalid_argument);

    TopologicalTorsionsOptions empty_count_bounds;
    empty_count_bounds.count_bounds.clear();
    EXPECT_THROW(MakeTopologicalTorsionsFingerprint(mol, empty_count_bounds), std::invalid_argument);
}

TEST(TopologicalTorsionsTest, GeneratedFingerprintCarriesStrictSpec) {
    const auto mol = mol_from_smiles("CCCC");
    TopologicalTorsionsOptions options;
    options.torsion_atom_count = 4;
    options.num_bits = 128;
    options.count_simulation = false;

    const auto fp = MakeTopologicalTorsionsFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 128u);
    EXPECT_EQ(spec.size_bits, 128u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Binary);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "TopologicalTorsions");
    EXPECT_EQ(spec.source_version, "TopologicalTorsions-1.0.0");
    EXPECT_EQ(
        spec.parameters,
        "torsion_atom_count=4;num_bits=128;use_chirality=false;"
        "count_simulation=false;count_bounds=1,2,4,8");
}

TEST(TopologicalTorsionsGeneratorTest, FingerprintMatchesFunctionalApi) {
    const auto mol = mol_from_smiles("CC(C)CCO");
    TopologicalTorsionsOptions options;
    options.num_bits = 512;

    const TopologicalTorsionsGenerator generator(options);

    const auto generated = generator.Fingerprint(mol);
    const auto functional = MakeTopologicalTorsionsFingerprint(mol, options);

    EXPECT_EQ(generated.Spec(), functional.Spec());
    EXPECT_EQ(generated.Words(), functional.Words());
}

TEST(TopologicalTorsionsTest, GeneratedCountFingerprintCarriesStrictSpec) {
    const auto mol = mol_from_smiles("CCCC");
    TopologicalTorsionsOptions options;
    options.num_bits = 128;

    const auto fp = MakeTopologicalTorsionsCountFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 128u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Counted);
    EXPECT_EQ(spec.source_type, "TopologicalTorsions");
    EXPECT_EQ(
        spec.parameters,
        "torsion_atom_count=4;num_bits=128;use_chirality=false;"
        "count_simulation=false;count_bounds=1,2,4,8");
}

TEST(TopologicalTorsionsTest, GeneratedSparseFingerprintCarriesStrictSpec) {
    const auto mol = mol_from_smiles("CCCC");
    TopologicalTorsionsOptions options;
    options.count_simulation = false;

    const auto fp = MakeTopologicalTorsionsSparseFingerprint(mol, options);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 4294967295u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Binary);
    EXPECT_EQ(spec.source_type, "TopologicalTorsions");
    EXPECT_EQ(
        spec.parameters,
        "torsion_atom_count=4;use_chirality=false;"
        "count_simulation=false;count_bounds=1,2,4,8;output=sparse_binary");
}

TEST(TopologicalTorsionsTest, GeneratedSparseCountFingerprintCarriesStrictSpec) {
    const auto mol = mol_from_smiles("CCCC");

    const auto fp = MakeTopologicalTorsionsSparseCountFingerprint(mol);
    const auto& spec = fp.Spec();

    EXPECT_EQ(fp.SizeBits(), 1ULL << 36u);
    EXPECT_EQ(spec.value_type, FingerprintValueType::Counted);
    EXPECT_EQ(spec.source_name, "RDKit-compatible");
    EXPECT_EQ(spec.source_type, "TopologicalTorsions");
    EXPECT_EQ(spec.source_version, "TopologicalTorsions-1.0.0");
    EXPECT_EQ(
        spec.parameters,
        "torsion_atom_count=4;use_chirality=false;output=sparse_count");
}

TEST(TopologicalTorsionsTest, GeneratedSparseCountFingerprintUsesRawRdkitIds) {
    const auto butane = MakeTopologicalTorsionsSparseCountFingerprint(mol_from_smiles("CCCC"));

    EXPECT_EQ(butane.TotalCount(), 1u);
    EXPECT_EQ(butane.Indices(), std::vector<std::uint64_t>({4303372320ULL}));
    EXPECT_EQ(butane.Counts(), std::vector<std::uint32_t>({1u}));

    const auto pentane = MakeTopologicalTorsionsSparseCountFingerprint(mol_from_smiles("CCCCC"));

    EXPECT_EQ(pentane.TotalCount(), 2u);
    EXPECT_EQ(pentane.Indices(), std::vector<std::uint64_t>({4437590048ULL}));
    EXPECT_EQ(pentane.Counts(), std::vector<std::uint32_t>({2u}));
}

TEST(TopologicalTorsionsTest, GeneratedDescriptorsCarryStrictSpec) {
    const auto mol = mol_from_smiles("CCCC");
    TopologicalTorsionsOptions options;
    options.torsion_atom_count = 4;

    const auto descriptors = MakeTopologicalTorsionsDescriptors(mol, options);
    const auto& schema = descriptors.Schema();
    const auto& definition = schema.Definition(schema.IndexOf("topological_torsions"));

    EXPECT_EQ(definition.value_kind, DescriptorValueKind::CountedStringKeys);
    EXPECT_EQ(definition.source_name, "OEFP");
    EXPECT_EQ(definition.source_type, "TopologicalTorsions");
    EXPECT_EQ(definition.source_version, "TopologicalTorsions-1.0.0");
    EXPECT_EQ(
        definition.parameters,
        "torsion_atom_count=4;use_chirality=false;output=descriptors");
    EXPECT_EQ(definition.prerequisites, kDescriptorPrerequisiteGraph);
    EXPECT_EQ(definition.prerequisites & kDescriptorPrerequisiteCoordinates2D, 0u);
    EXPECT_EQ(definition.prerequisites & kDescriptorPrerequisiteCoordinates3D, 0u);
}

TEST(TopologicalTorsionsTest, GeneratedDescriptorsUseCanonicalPathCodeKeys) {
    const auto butane = MakeTopologicalTorsionsDescriptors(mol_from_smiles("CCCC"));
    const auto& butane_values = torsion_values(butane);

    EXPECT_EQ(total_count(butane_values), 1u);
    EXPECT_EQ(butane_values.keys, std::vector<std::string>({"32_32_32_32"}));
    EXPECT_EQ(butane_values.counts, std::vector<std::uint32_t>({1u}));

    const auto pentane = MakeTopologicalTorsionsDescriptors(mol_from_smiles("CCCCC"));
    const auto& pentane_values = torsion_values(pentane);

    EXPECT_EQ(total_count(pentane_values), 2u);
    EXPECT_EQ(pentane_values.keys, std::vector<std::string>({"32_32_32_33"}));
    EXPECT_EQ(pentane_values.counts, std::vector<std::uint32_t>({2u}));

    const auto cyclopropane = MakeTopologicalTorsionsDescriptors(mol_from_smiles("C1CC1"));
    const auto& cyclopropane_values = torsion_values(cyclopropane);

    EXPECT_EQ(total_count(cyclopropane_values), 1u);
    EXPECT_EQ(cyclopropane_values.keys, std::vector<std::string>({"33_32_32_33"}));
    EXPECT_EQ(cyclopropane_values.counts, std::vector<std::uint32_t>({1u}));
}

TEST(TopologicalTorsionsTest, ChiralSparseCountMatchesRdkitRawIds) {
    TopologicalTorsionsOptions options;
    options.use_chirality = true;

    const auto r_form =
        MakeTopologicalTorsionsSparseCountFingerprint(mol_from_smiles("C[C@](F)(Cl)CC"), options);

    EXPECT_EQ(
        r_form.Indices(),
        std::vector<std::uint64_t>({279315546144ULL, 1103949266976ULL, 2203460894752ULL}));
    EXPECT_EQ(r_form.Counts(), std::vector<std::uint32_t>({1u, 1u, 1u}));
}

TEST(TopologicalTorsionsTest, ChiralEncodingDistinguishesEnantiomers) {
    TopologicalTorsionsOptions options;
    options.use_chirality = true;

    const auto r_form =
        MakeTopologicalTorsionsSparseCountFingerprint(mol_from_smiles("C[C@](F)(Cl)CC"), options);
    const auto s_form =
        MakeTopologicalTorsionsSparseCountFingerprint(mol_from_smiles("C[C@@](F)(Cl)CC"), options);

    EXPECT_NE(r_form.Indices(), s_form.Indices());
}

TEST(TopologicalTorsionsTest, ChiralityLeavesFoldedAchiralOutputUnchanged) {
    const auto mol = mol_from_smiles("CCCC");

    TopologicalTorsionsOptions with_chirality;
    with_chirality.use_chirality = true;
    with_chirality.num_bits = 128;

    TopologicalTorsionsOptions without_chirality;
    without_chirality.num_bits = 128;

    const auto chiral_fp = MakeTopologicalTorsionsCountFingerprint(mol, with_chirality);
    const auto achiral_fp = MakeTopologicalTorsionsCountFingerprint(mol, without_chirality);

    EXPECT_EQ(chiral_fp.Indices(), achiral_fp.Indices());
    EXPECT_EQ(chiral_fp.Counts(), achiral_fp.Counts());
}

} // namespace test
} // namespace OEFP
