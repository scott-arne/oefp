#include <gtest/gtest.h>

#include "oefp/mordred.h"

#include <oechem.h>

#include <cstdint>
#include <map>
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

const std::vector<std::string>& supported_count_names() {
    static const std::vector<std::string> names{
        "nAromAtom", "nAromBond", "nAtom",    "nB",       "nBonds",     "nBondsA",
        "nBondsD",   "nBondsKD",  "nBondsKS", "nBondsM",  "nBondsO",    "nBondsS",
        "nBondsT",   "nBr",       "nC",       "nCl",      "nF",         "nH",
        "nHeavyAtom", "nHetero",   "nI",       "nN",       "nO",         "nP",
        "nS",        "nX",
    };
    return names;
}

std::uint32_t count_or_zero(
    const std::map<std::string, std::uint32_t>& counts,
    const std::string& key) {
    const auto found = counts.find(key);
    return found == counts.end() ? 0u : found->second;
}

} // namespace

TEST(MordredDescriptorTest, DescriptorRowCarriesFullSchema) {
    const auto descriptors = MakeMordredDescriptors(mol_from_smiles("c1ccncc1"));

    EXPECT_EQ(descriptors.Schema().Size(), 1826u);
    EXPECT_EQ(descriptors.Schema().SchemaId(), MordredDescriptorSchema()->SchemaId());
    EXPECT_TRUE(descriptors.Schema().Contains("Lipinski"));
    EXPECT_TRUE(descriptors.Schema().Contains("GhoseFilter"));
    EXPECT_TRUE(descriptors.Has("nAtom"));
    EXPECT_TRUE(descriptors.Has("MW"));
    EXPECT_FALSE(descriptors.Has("Lipinski"));
    EXPECT_FALSE(descriptors.Has("GhoseFilter"));
    EXPECT_FALSE(descriptors.Has("ABC"));
}

TEST(MordredDescriptorTest, CountSubsetMatchesCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, std::uint32_t> nonzero;
    };

    const std::vector<Case> cases{
        {
            "CCO",
            {
                {"nAtom", 9},       {"nBonds", 8},    {"nBondsKS", 8},
                {"nBondsO", 2},     {"nBondsS", 8},   {"nC", 2},
                {"nH", 6},          {"nHeavyAtom", 3}, {"nHetero", 1},
                {"nO", 1},
            },
        },
        {
            "CC(C)(C)Cl",
            {
                {"nAtom", 14},      {"nBonds", 13},   {"nBondsKS", 13},
                {"nBondsO", 4},     {"nBondsS", 13},  {"nC", 4},
                {"nCl", 1},         {"nH", 9},        {"nHeavyAtom", 5},
                {"nHetero", 1},     {"nX", 1},
            },
        },
        {
            "c1ccccc1",
            {
                {"nAromAtom", 6},   {"nAromBond", 6}, {"nAtom", 12},
                {"nBonds", 12},     {"nBondsA", 6},   {"nBondsKD", 3},
                {"nBondsKS", 9},    {"nBondsM", 6},   {"nBondsO", 6},
                {"nBondsS", 6},     {"nC", 6},        {"nH", 6},
                {"nHeavyAtom", 6},
            },
        },
        {
            "c1ccncc1",
            {
                {"nAromAtom", 6},   {"nAromBond", 6}, {"nAtom", 11},
                {"nBonds", 11},     {"nBondsA", 6},   {"nBondsKD", 3},
                {"nBondsKS", 8},    {"nBondsM", 6},   {"nBondsO", 6},
                {"nBondsS", 5},     {"nC", 5},        {"nH", 5},
                {"nHeavyAtom", 6},  {"nHetero", 1},   {"nN", 1},
            },
        },
        {
            "CC=O",
            {
                {"nAtom", 7},       {"nBonds", 6},    {"nBondsD", 1},
                {"nBondsKD", 1},    {"nBondsKS", 5},  {"nBondsM", 1},
                {"nBondsO", 2},     {"nBondsS", 5},   {"nC", 2},
                {"nH", 4},          {"nHeavyAtom", 3}, {"nHetero", 1},
                {"nO", 1},
            },
        },
        {
            "CC#N",
            {
                {"nAtom", 6},       {"nBonds", 5},    {"nBondsKS", 4},
                {"nBondsM", 1},     {"nBondsO", 2},   {"nBondsS", 4},
                {"nBondsT", 1},     {"nC", 2},        {"nH", 3},
                {"nHeavyAtom", 3},  {"nHetero", 1},   {"nN", 1},
            },
        },
        {
            "C1CCCCC1",
            {
                {"nAtom", 18},      {"nBonds", 18},   {"nBondsKS", 18},
                {"nBondsO", 6},     {"nBondsS", 18},  {"nC", 6},
                {"nH", 12},         {"nHeavyAtom", 6},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& name : supported_count_names()) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_EQ(
                descriptors.Int(name),
                static_cast<std::int64_t>(count_or_zero(expected.nonzero, name)))
                << name;
        }
    }
}

TEST(MordredDescriptorTest, CrippenDescriptorsMatchCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        double slogp;
        double smr;
    };

    const std::vector<Case> cases{
        {"CCO", -0.0014000000000000123, 12.759800000000002},
        {"c1ccncc1", 1.0816, 24.236999999999995},
        {"FC(F)(F)c1ccc(Br)cc1", 3.467900000000001, 39.14400000000001},
        {"O=[Se]=O", -0.6184000000000001, 7.126999999999999},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        EXPECT_TRUE(descriptors.Has("SLogP"));
        EXPECT_TRUE(descriptors.Has("SMR"));
        EXPECT_NEAR(descriptors.Float("SLogP"), expected.slogp, 1.0e-8);
        EXPECT_NEAR(descriptors.Float("SMR"), expected.smr, 1.0e-8);
    }
}

} // namespace test
} // namespace OEFP
