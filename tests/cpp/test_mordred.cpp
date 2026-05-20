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
    EXPECT_TRUE(descriptors.Has("Lipinski"));
    EXPECT_TRUE(descriptors.Has("GhoseFilter"));
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

TEST(MordredDescriptorTest, AdditivePropertyDescriptorsMatchCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "CCO",
            {
                {"SZ", 4.333333333333334},
                {"Sm", 3.83556739655316},
                {"Sv", 4.340282515774477},
                {"Sse", 8.994173343044428},
                {"Spe", 8.525490196078433},
                {"Sare", 8.680000000000001},
                {"Sp", 4.875902994011976},
                {"Si", 10.455255010967736},
                {"MZ", 0.48148148148148157},
                {"Mm", 0.4261741551725733},
                {"Mv", 0.4822536128638307},
                {"Mse", 0.9993525936716031},
                {"Mpe", 0.9472766884531593},
                {"Mare", 0.9644444444444447},
                {"Mp", 0.541766999334664},
                {"Mi", 1.1616950012186373},
                {"VMcGowan", 44.91000000000002},
                {"apol", 8.142758},
                {"bpol", 6.019242},
                {"Vabc", 51.93865634694684},
            },
        },
        {
            "O=S(=O)(N)C1=CC=CC=C1",
            {
                {"SZ", 13.666666666666666},
                {"Sm", 13.08692032303722},
                {"Sv", 11.27101384083045},
                {"Sse", 17.50873998543336},
                {"Spe", 16.941176470588236},
                {"Sare", 17.163999999999998},
                {"Sp", 12.150629341317366},
                {"Si", 19.083071587790734},
                {"MZ", 0.8039215686274509},
                {"Mm", 0.7698188425316012},
                {"Mv", 0.6630008141664971},
                {"Mse", 1.0299258814960799},
                {"Mpe", 0.9965397923875433},
                {"Mare", 1.0096470588235293},
                {"Mp", 0.7147429024304333},
                {"Mi", 1.1225336228112197},
                {"VMcGowan", 109.71000000000002},
                {"apol", 20.291551},
                {"bpol", 13.108449},
                {"Vabc", 128.25277347493352},
            },
        },
        {
            "C1=CC2=C(C=C1)C=CC=C2",
            {
                {"Vabc", 121.82109855212238},
            },
        },
        {
            "CI",
            {
                {"SZ", 10.333333333333332},
                {"Sm", 11.817456498209976},
                {"Sv", 3.392711581518422},
                {"Sse", 4.843408594319009},
                {"Spe", 4.631372549019608},
                {"Sare", 4.524},
                {"Sp", 5.401424550898202},
                {"Si", 5.551085583865438},
                {"MZ", 2.0666666666666664},
                {"Mm", 2.3634912996419954},
                {"Mv", 0.6785423163036844},
                {"Mse", 0.9686817188638018},
                {"Mpe", 0.9262745098039217},
                {"Mare", 0.9048},
                {"Mp", 1.0802849101796403},
                {"Mi", 1.1102171167730877},
                {"VMcGowan", 50.780000000000015},
                {"apol", 9.020379},
                {"bpol", 6.689621},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
        }
    }
}

TEST(MordredDescriptorTest, VabcDocumentsOpenEyeRingPrimitiveDivergence) {
    const auto descriptors =
        MakeMordredDescriptors(mol_from_smiles("C12C3C4C1C5C2C3C45"));

    EXPECT_TRUE(descriptors.Has("Vabc"));
    EXPECT_NEAR(descriptors.Float("Vabc"), 85.14204599989137, 1.0e-8);
}

TEST(MordredDescriptorTest, WalkCountDescriptorsMatchCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "CCC",
            {
                {"MWC01", 2.0},
                {"MWC05", 2.833213344056216},
                {"SRW04", 2.1972245773362196},
                {"TMWC10", 33.89759936075361},
                {"TSRW10", 17.310770665188652},
            },
        },
        {
            "C1CCCCC1",
            {
                {"MWC01", 6.0},
                {"MWC05", 5.262690188904886},
                {"SRW04", 3.6109179126442243},
                {"TMWC10", 65.63782292184975},
                {"TSRW10", 30.941316689854872},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
        }
    }
}

TEST(MordredDescriptorTest, NonPiPathCountDescriptorsMatchMordredPathSemantics) {
    struct Case {
        std::string smiles;
        std::map<std::string, std::int64_t> expected_values;
    };

    const std::vector<Case> cases{
        {
            "CCCCCC",
            {
                {"MPC2", 4},
                {"MPC3", 3},
                {"MPC5", 1},
                {"MPC10", 0},
                {"TMPC10", 21},
            },
        },
        {
            "CCC(C)CC",
            {
                {"MPC2", 5},
                {"MPC3", 4},
                {"MPC4", 1},
                {"MPC10", 0},
                {"TMPC10", 21},
            },
        },
        {
            "C1CCCCC1",
            {
                {"MPC2", 6},
                {"MPC3", 6},
                {"MPC5", 6},
                {"MPC6", 0},
                {"MPC10", 0},
                {"TMPC10", 36},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_EQ(descriptors.Int(name), value) << name;
        }
    }
}

TEST(MordredDescriptorTest, FilterDescriptorsMatchCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        bool lipinski;
        bool ghose_filter;
    };

    const std::vector<Case> cases{
        {"CCO", true, false},
        {"c1ccncc1", true, false},
        {"CCCCCCCCCCCCCCCC", false, false},
        {"CCOC(=O)c1ccc(OCC)c(O)c1C(=O)OCC", true, true},
        {"COC(=O)c1ccc(OCC)c(O)c1C(=O)OCC", true, true},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        EXPECT_TRUE(descriptors.Has("Lipinski"));
        EXPECT_TRUE(descriptors.Has("GhoseFilter"));
        EXPECT_EQ(descriptors.Bool("Lipinski"), expected.lipinski);
        EXPECT_EQ(descriptors.Bool("GhoseFilter"), expected.ghose_filter);
    }
}

} // namespace test
} // namespace OEFP
