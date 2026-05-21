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
    EXPECT_TRUE(descriptors.Has("ABC"));
    EXPECT_TRUE(descriptors.Has("SpAbs_A"));
    EXPECT_TRUE(descriptors.Has("SpAbs_D"));
    EXPECT_TRUE(descriptors.Has("VE1_A"));
    EXPECT_TRUE(descriptors.Has("VE1_D"));
    EXPECT_TRUE(descriptors.Has("VR1_A"));
    EXPECT_TRUE(descriptors.Has("VR1_D"));
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

TEST(MordredDescriptorTest, BaseRingCountDescriptorsMatchMordredSymmSSSRReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, std::int64_t> expected_values;
    };

    const std::vector<std::string> names{
        "nRing",   "n3Ring",  "n4Ring",  "n5Ring",   "n6Ring",  "n7Ring",
        "n8Ring",  "n9Ring",  "n10Ring", "n11Ring",  "n12Ring", "nG12Ring",
    };
    const std::vector<Case> cases{
        {"CCO", {}},
        {"C1CC1C", {{"nRing", 1}, {"n3Ring", 1}}},
        {"c1ccncc1", {{"nRing", 1}, {"n6Ring", 1}}},
        {"C1=CC2=C(C=C1)C=CC=C2", {{"nRing", 2}, {"n6Ring", 2}}},
        {"C12C3C4C1C5C2C3C45", {{"nRing", 6}, {"n4Ring", 6}}},
        {"C12C3C1C23", {{"nRing", 4}, {"n3Ring", 4}}},
        {"C123C45C16C24C356", {{"nRing", 6}, {"n3Ring", 3}, {"n4Ring", 2}, {"n5Ring", 1}}},
        {"C123C45C16C24C356.CCO", {{"nRing", 6}, {"n3Ring", 3}, {"n4Ring", 2}, {"n5Ring", 1}}},
        {"C1CCCCCCCCCCC1", {{"nRing", 1}, {"n12Ring", 1}, {"nG12Ring", 1}}},
        {"C1CCCCCCCCCCCC1", {{"nRing", 1}, {"nG12Ring", 1}}},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& name : names) {
            const auto found = expected.expected_values.find(name);
            const auto expected_value =
                found == expected.expected_values.end() ? 0 : found->second;
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_EQ(descriptors.Int(name), expected_value) << name;
        }
    }
}

TEST(MordredDescriptorTest, FilteredRingCountDescriptorsMatchMordredSymmSSSRReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, std::int64_t> expected_values;
    };

    const std::vector<std::string> names{
        "nHRing",     "n3HRing",    "n4HRing",    "n5HRing",    "n6HRing",
        "n7HRing",    "n8HRing",    "n9HRing",    "n10HRing",   "n11HRing",
        "n12HRing",   "nG12HRing",  "naRing",     "n3aRing",    "n4aRing",
        "n5aRing",    "n6aRing",    "n7aRing",    "n8aRing",    "n9aRing",
        "n10aRing",   "n11aRing",   "n12aRing",   "nG12aRing",  "naHRing",
        "n3aHRing",   "n4aHRing",   "n5aHRing",   "n6aHRing",   "n7aHRing",
        "n8aHRing",   "n9aHRing",   "n10aHRing",  "n11aHRing",  "n12aHRing",
        "nG12aHRing", "nARing",     "n3ARing",    "n4ARing",    "n5ARing",
        "n6ARing",    "n7ARing",    "n8ARing",    "n9ARing",    "n10ARing",
        "n11ARing",   "n12ARing",   "nG12ARing",  "nAHRing",    "n3AHRing",
        "n4AHRing",   "n5AHRing",   "n6AHRing",   "n7AHRing",   "n8AHRing",
        "n9AHRing",   "n10AHRing",  "n11AHRing",  "n12AHRing",  "nG12AHRing",
    };
    const std::vector<Case> cases{
        {"CCO", {}},
        {"C1CC1", {{"nARing", 1}, {"n3ARing", 1}}},
        {"c1ccccc1", {{"naRing", 1}, {"n6aRing", 1}}},
        {
            "c1ccncc1",
            {
                {"nHRing", 1},
                {"n6HRing", 1},
                {"naRing", 1},
                {"n6aRing", 1},
                {"naHRing", 1},
                {"n6aHRing", 1},
            },
        },
        {
            "C1CCOCC1",
            {
                {"nHRing", 1},
                {"n6HRing", 1},
                {"nARing", 1},
                {"n6ARing", 1},
                {"nAHRing", 1},
                {"n6AHRing", 1},
            },
        },
        {
            "C1CCCCCCCCCCC1",
            {{"nARing", 1}, {"n12ARing", 1}, {"nG12ARing", 1}},
        },
        {
            "O1CCCCCCCCCCCC1",
            {
                {"nHRing", 1},
                {"nG12HRing", 1},
                {"nARing", 1},
                {"nG12ARing", 1},
                {"nAHRing", 1},
                {"nG12AHRing", 1},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& name : names) {
            const auto found = expected.expected_values.find(name);
            const auto expected_value =
                found == expected.expected_values.end() ? 0 : found->second;
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_EQ(descriptors.Int(name), expected_value) << name;
        }
    }
}

TEST(MordredDescriptorTest, FusedRingCountDescriptorsMatchMordredGroupingReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, std::int64_t> expected_values;
    };

    const std::vector<std::string> names{
        "nFRing",     "n4FRing",    "n5FRing",    "n6FRing",    "n7FRing",
        "n8FRing",    "n9FRing",    "n10FRing",   "n11FRing",   "n12FRing",
        "nG12FRing",  "nFHRing",    "n4FHRing",   "n5FHRing",   "n6FHRing",
        "n7FHRing",   "n8FHRing",   "n9FHRing",   "n10FHRing",  "n11FHRing",
        "n12FHRing",  "nG12FHRing", "nFaRing",    "n4FaRing",   "n5FaRing",
        "n6FaRing",   "n7FaRing",   "n8FaRing",   "n9FaRing",   "n10FaRing",
        "n11FaRing",  "n12FaRing",  "nG12FaRing", "nFaHRing",   "n4FaHRing",
        "n5FaHRing",  "n6FaHRing",  "n7FaHRing",  "n8FaHRing",  "n9FaHRing",
        "n10FaHRing", "n11FaHRing", "n12FaHRing", "nG12FaHRing", "nFARing",
        "n4FARing",   "n5FARing",   "n6FARing",   "n7FARing",   "n8FARing",
        "n9FARing",   "n10FARing",  "n11FARing",  "n12FARing",  "nG12FARing",
        "nFAHRing",   "n4FAHRing",  "n5FAHRing",  "n6FAHRing",  "n7FAHRing",
        "n8FAHRing",  "n9FAHRing",  "n10FAHRing", "n11FAHRing", "n12FAHRing",
        "nG12FAHRing",
    };
    const std::vector<Case> cases{
        {"CCO", {}},
        {"C1CCCCC1.C1CCCCC1", {}},
        {
            "C1=CC2=C(C=C1)C=CC=C2",
            {{"nFRing", 1}, {"n10FRing", 1}, {"nFaRing", 1}, {"n10FaRing", 1}},
        },
        {
            "c1ccc2ncccc2c1",
            {
                {"nFRing", 1},
                {"n10FRing", 1},
                {"nFHRing", 1},
                {"n10FHRing", 1},
                {"nFaRing", 1},
                {"n10FaRing", 1},
                {"nFaHRing", 1},
                {"n10FaHRing", 1},
            },
        },
        {
            "C12C3C4C1C5C2C3C45",
            {{"nFRing", 1}, {"n8FRing", 1}, {"nFARing", 1}, {"n8FARing", 1}},
        },
        {
            "C1C2CC3CC1CC(C2)C3",
            {{"nFRing", 1}, {"n10FRing", 1}, {"nFARing", 1}, {"n10FARing", 1}},
        },
        {
            "O1CC2CCC1C2",
            {
                {"nFRing", 1},
                {"n7FRing", 1},
                {"nFHRing", 1},
                {"n7FHRing", 1},
                {"nFARing", 1},
                {"n7FARing", 1},
                {"nFAHRing", 1},
                {"n7FAHRing", 1},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& name : names) {
            const auto found = expected.expected_values.find(name);
            const auto expected_value =
                found == expected.expected_values.end() ? 0 : found->second;
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_EQ(descriptors.Int(name), expected_value) << name;
        }
    }
}

TEST(MordredDescriptorTest, ZagrebDescriptorsUseHeavyAtomGraphDegrees) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
        std::vector<std::string> missing_values;
    };

    const std::vector<Case> cases{
        {
            "CCO",
            {
                {"Zagreb1", 6.0},
                {"Zagreb2", 4.0},
                {"mZagreb1", 2.25},
                {"mZagreb2", 1.0},
            },
            {},
        },
        {
            "c1ccncc1",
            {
                {"Zagreb1", 24.0},
                {"Zagreb2", 24.0},
                {"mZagreb1", 1.5},
                {"mZagreb2", 1.5},
            },
            {},
        },
        {
            "C",
            {
                {"Zagreb1", 0.0},
                {"Zagreb2", 0.0},
                {"mZagreb2", 0.0},
            },
            {"mZagreb1"},
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
        }
        for (const auto& name : expected.missing_values) {
            EXPECT_FALSE(descriptors.Has(name)) << name;
        }
    }
}

TEST(MordredDescriptorTest, WienerDescriptorsUseHeavyAtomShortestPaths) {
    struct Case {
        std::string smiles;
        std::map<std::string, std::int64_t> expected_values;
    };

    const std::vector<Case> cases{
        {
            "CCO",
            {
                {"WPath", 4},
                {"WPol", 0},
            },
        },
        {
            "c1ccncc1",
            {
                {"WPath", 27},
                {"WPol", 3},
            },
        },
        {
            "CCCCCC",
            {
                {"WPath", 35},
                {"WPol", 3},
            },
        },
        {
            "C.CC",
            {
                {"WPath", 200000001},
                {"WPol", 0},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            if (descriptors.Has(name)) {
                EXPECT_EQ(descriptors.Int(name), value) << name;
            }
        }
    }
}

TEST(MordredDescriptorTest, EccentricConnectivityIndexUsesHeavyAtomEccentricityAndDegree) {
    struct Case {
        std::string smiles;
        std::int64_t expected_value;
    };

    const std::vector<Case> cases{
        {"CCO", 6},
        {"c1ccncc1", 36},
        {"CCCCCC", 38},
        {"C1CCCCC1", 36},
        {"C", 0},
        {"C.CC", 200000000},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("ECIndex"));
        EXPECT_EQ(descriptors.Int("ECIndex"), expected.expected_value);
    }
}

TEST(MordredDescriptorTest, TopologicalIndexDescriptorsUseHeavyAtomShortestPaths) {
    struct Case {
        std::string smiles;
        std::int64_t diameter;
        std::int64_t radius;
        std::map<std::string, double> shape_values;
    };

    const std::vector<Case> cases{
        {"CCO", 2, 1, {{"TopoShapeIndex", 1.0}, {"PetitjeanIndex", 0.5}}},
        {"c1ccncc1", 3, 3, {{"TopoShapeIndex", 0.0}, {"PetitjeanIndex", 0.0}}},
        {"CCCCCC", 5, 3, {{"TopoShapeIndex", 2.0 / 3.0}, {"PetitjeanIndex", 0.4}}},
        {"C", 0, 0, {}},
        {"C.CC", 100000000, 100000000, {{"TopoShapeIndex", 0.0}, {"PetitjeanIndex", 0.0}}},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("Diameter"));
        ASSERT_TRUE(descriptors.Has("Radius"));
        EXPECT_EQ(descriptors.Int("Diameter"), expected.diameter);
        EXPECT_EQ(descriptors.Int("Radius"), expected.radius);

        for (const auto& [name, value] : expected.shape_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            if (descriptors.Has(name)) {
                EXPECT_NEAR(descriptors.Float(name), value, 1.0e-12) << name;
            }
        }
        if (expected.shape_values.empty()) {
            EXPECT_FALSE(descriptors.Has("TopoShapeIndex"));
            EXPECT_FALSE(descriptors.Has("PetitjeanIndex"));
        }
    }
}

TEST(MordredDescriptorTest, TopologicalIndexDescriptorsAreMissingForEmptyMolecule) {
    const OEChem::OEGraphMol mol;
    const auto descriptors = MakeMordredDescriptors(mol);

    EXPECT_FALSE(descriptors.Has("Diameter"));
    EXPECT_FALSE(descriptors.Has("Radius"));
    EXPECT_FALSE(descriptors.Has("TopoShapeIndex"));
    EXPECT_FALSE(descriptors.Has("PetitjeanIndex"));
}

TEST(MordredDescriptorTest, VertexAdjacencyInformationUsesHeavyHeavyBondCount) {
    struct Case {
        std::string smiles;
        double expected_value;
    };

    const std::vector<Case> cases{
        {"CCO", 2.0},
        {"c1ccncc1", 3.584962500721156},
        {"C.CC", 1.0},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("VAdjMat"));
        EXPECT_NEAR(descriptors.Float("VAdjMat"), expected.expected_value, 1.0e-12);
    }

    EXPECT_FALSE(MakeMordredDescriptors(mol_from_smiles("C")).Has("VAdjMat"));

    const OEChem::OEGraphMol empty_mol;
    EXPECT_FALSE(MakeMordredDescriptors(empty_mol).Has("VAdjMat"));
}

TEST(MordredDescriptorTest, BalabanJUsesHeavyAtomDistanceRowSumsAndBonds) {
    struct Case {
        std::string smiles;
        double expected_value;
    };

    const std::vector<Case> cases{
        {"C", 0.0},
        {"CC", 1.0},
        {"C.CC", 0.0},
        {"CCO", 1.6329931618554523},
        {"c1ccncc1", 2.0},
        {"CCCCCC", 2.3390923149762908},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("BalabanJ"));
        EXPECT_NEAR(descriptors.Float("BalabanJ"), expected.expected_value, 1.0e-12);
    }

    const OEChem::OEGraphMol empty_mol;
    EXPECT_FALSE(MakeMordredDescriptors(empty_mol).Has("BalabanJ"));
}

TEST(MordredDescriptorTest, BertzCTUsesHeavyAtomSymmetryClassesAndBondOrders) {
    struct Case {
        std::string smiles;
        double expected_value;
    };

    const std::vector<Case> cases{
        {"[H][H]", 0.0},
        {"C", 0.0},
        {"CC", 0.0},
        {"CCC", 0.0},
        {"CCCCCC", 12.0},
        {"CCO", 2.754887502163469},
        {"CC#N", 24.264662506490406},
        {"c1ccncc1", 75.86113958768547},
        {"CC(=O)O", 27.019550008653873},
        {"OP(=O)(O)O", 49.78353986569366},
        {"C[N+](C)(C)CC(=O)[O-]", 93.09201719290463},
        {"c1ccccc1[N+](=O)[O-]", 207.5576620543373},
        {"O=S(=O)(N)C1=CC=CC=C1", 303.7795180972205},
        {"C.CC", 0.0},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("BertzCT"));
        EXPECT_NEAR(descriptors.Float("BertzCT"), expected.expected_value, 1.0e-12);
    }

    const OEChem::OEGraphMol empty_mol;
    const auto descriptors = MakeMordredDescriptors(empty_mol);
    ASSERT_TRUE(descriptors.Has("BertzCT"));
    EXPECT_NEAR(descriptors.Float("BertzCT"), 0.0, 1.0e-12);
}

TEST(MordredDescriptorTest, AdjacencyMatrixEigenvalueDescriptorsMatchMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "C",
            {
                {"SpAbs_A", 0.0},
                {"SpMax_A", 0.0},
                {"SpDiam_A", 0.0},
                {"SpAD_A", 0.0},
                {"SpMAD_A", 0.0},
                {"LogEE_A", 0.6931471805599453},
                {"VE1_A", 1.0},
                {"VE2_A", 1.0},
                {"VE3_A", -2.3025850929940455},
                {"VR1_A", 0.0},
                {"VR2_A", 0.0},
            },
        },
        {
            "CC",
            {
                {"SpAbs_A", 2.0},
                {"SpMax_A", 1.0},
                {"SpDiam_A", 2.0},
                {"SpAD_A", 2.0},
                {"SpMAD_A", 1.0},
                {"LogEE_A", 1.4076059644443804},
                {"VE1_A", 1.414213562373095},
                {"VE2_A", 0.7071067811865475},
                {"VE3_A", -1.2628643221541278},
                {"VR1_A", 1.4142135623730951},
                {"VR2_A", 0.7071067811865476},
                {"VR3_A", -1.2628643221541276},
            },
        },
        {
            "CCO",
            {
                {"SpAbs_A", 2.82842712474619},
                {"SpMax_A", 1.414213562373095},
                {"SpDiam_A", 2.82842712474619},
                {"SpAD_A", 2.82842712474619},
                {"SpMAD_A", 0.9428090415820632},
                {"LogEE_A", 1.8494570055365824},
                {"VE1_A", 1.7071067811865475},
                {"VE2_A", 0.5690355937288492},
                {"VE3_A", -0.6691728075863654},
                {"VR1_A", 3.3635856610148585},
                {"VR2_A", 1.1211952203382862},
                {"VR3_A", 0.009034761653968602},
            },
        },
        {
            "c1ccncc1",
            {
                {"SpAbs_A", 7.999999999999998},
                {"SpMax_A", 1.9999999999999998},
                {"SpDiam_A", 3.999999999999999},
                {"SpAD_A", 7.999999999999998},
                {"SpMAD_A", 1.333333333333333},
                {"LogEE_A", 2.6876239260352994},
                {"VE1_A", 2.4494897427831788},
                {"VE2_A", 0.40824829046386313},
                {"VE3_A", 0.3850541108480373},
                {"VR1_A", 14.696938456699066},
                {"VR2_A", 2.4494897427831774},
                {"VR3_A", 2.1768135800760917},
            },
        },
        {
            "CCCCCC",
            {
                {"SpAbs_A", 6.987918414869867},
                {"SpMax_A", 1.8019377358048383},
                {"SpDiam_A", 3.603875471609676},
                {"SpAD_A", 6.987918414869867},
                {"SpMAD_A", 1.164653069144978},
                {"LogEE_A", 2.579830499327949},
                {"VE1_A", 2.3418960180704147},
                {"VE2_A", 0.3903160030117358},
                {"VE3_A", 0.34013524164950576},
                {"VR1_A", 12.628859982958033},
                {"VR2_A", 2.104809997159672},
                {"VR3_A", 2.0251590458905078},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            if (descriptors.Has(name)) {
                EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
            }
        }
        if (expected.smiles == "C") {
            EXPECT_FALSE(descriptors.Has("VR3_A"));
        }
    }
}

TEST(MordredDescriptorTest, AdjacencyMatrixEigenvalueDescriptorsAreMissingWhenRequiredMatrixIsMissing) {
    const std::vector<std::string> descriptor_names{
        "SpAbs_A",
        "SpMax_A",
        "SpDiam_A",
        "SpAD_A",
        "SpMAD_A",
        "LogEE_A",
        "VE1_A",
        "VE2_A",
        "VE3_A",
        "VR1_A",
        "VR2_A",
        "VR3_A",
    };

    const OEChem::OEGraphMol empty_mol;
    const std::vector<std::pair<std::string, OEChem::OEGraphMol>> cases{
        {"empty", empty_mol},
        {"[H][H]", mol_from_smiles("[H][H]")},
        {"C.CC", mol_from_smiles("C.CC")},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.first);
        const auto descriptors = MakeMordredDescriptors(expected.second);

        for (const auto& name : descriptor_names) {
            EXPECT_FALSE(descriptors.Has(name)) << name;
        }
    }
}

TEST(MordredDescriptorTest, DistanceMatrixEigenvalueDescriptorsMatchMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "C",
            {
                {"SpAbs_D", 0.0},
                {"SpMax_D", 0.0},
                {"SpDiam_D", 0.0},
                {"SpAD_D", 0.0},
                {"SpMAD_D", 0.0},
                {"LogEE_D", 0.6931471805599453},
                {"VE1_D", 1.0},
                {"VE2_D", 1.0},
                {"VE3_D", -2.3025850929940455},
                {"VR1_D", 0.0},
                {"VR2_D", 0.0},
            },
        },
        {
            "CC",
            {
                {"SpAbs_D", 2.0},
                {"SpMax_D", 1.0},
                {"SpDiam_D", 2.0},
                {"SpAD_D", 2.0},
                {"SpMAD_D", 1.0},
                {"LogEE_D", 1.4076059644443804},
                {"VE1_D", 1.414213562373095},
                {"VE2_D", 0.7071067811865475},
                {"VE3_D", -1.2628643221541278},
                {"VR1_D", 1.4142135623730951},
                {"VR2_D", 0.7071067811865476},
                {"VR3_D", -1.2628643221541276},
            },
        },
        {
            "CCO",
            {
                {"SpAbs_D", 5.464101615137755},
                {"SpMax_D", 2.7320508075688776},
                {"SpDiam_D", 4.732050807568878},
                {"SpAD_D", 5.464101615137755},
                {"SpMAD_D", 1.8213672050459184},
                {"LogEE_D", 2.832072756761435},
                {"VE1_D", 1.7156269037800915},
                {"VE2_D", 0.5718756345933639},
                {"VE3_D", -0.6641942489393373},
                {"VR1_D", 3.7224194364083996},
                {"VR2_D", 1.2408064788028},
                {"VR3_D", 0.11040103868100991},
            },
        },
        {
            "c1ccncc1",
            {
                {"SpAbs_D", 18.000000000000007},
                {"SpMax_D", 9.000000000000002},
                {"SpDiam_D", 13.000000000000002},
                {"SpAD_D", 18.000000000000007},
                {"SpMAD_D", 3.0000000000000013},
                {"LogEE_D", 9.000420061762542},
                {"VE1_D", 2.4494897427831788},
                {"VE2_D", 0.40824829046386313},
                {"VE3_D", 0.3850541108480373},
                {"VR1_D", 14.696938456699066},
                {"VR2_D", 2.4494897427831774},
                {"VR3_D", 2.1768135800760917},
            },
        },
        {
            "CCCCCC",
            {
                {"SpAbs_D", 24.21862300782596},
                {"SpMax_D", 12.109311503912979},
                {"SpDiam_D", 19.573413119050734},
                {"SpAD_D", 24.21862300782596},
                {"SpMAD_D", 4.036437167970994},
                {"LogEE_D", 12.109325540282484},
                {"VE1_D", 2.4117970446931674},
                {"VE2_D", 0.4019661741155279},
                {"VE3_D", 0.3695465075674201},
                {"VR1_D", 13.316549998387089},
                {"VR2_D", 2.2194249997311815},
                {"VR3_D", 2.078181998667496},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            if (descriptors.Has(name)) {
                EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
            }
        }
        if (expected.smiles == "C") {
            EXPECT_FALSE(descriptors.Has("VR3_D"));
        }
    }
}

TEST(MordredDescriptorTest, DistanceMatrixEigenvalueDescriptorsAreMissingWhenRequiredMatrixIsMissing) {
    const std::vector<std::string> descriptor_names{
        "SpAbs_D",
        "SpMax_D",
        "SpDiam_D",
        "SpAD_D",
        "SpMAD_D",
        "LogEE_D",
        "VE1_D",
        "VE2_D",
        "VE3_D",
        "VR1_D",
        "VR2_D",
        "VR3_D",
    };

    const OEChem::OEGraphMol empty_mol;
    const std::vector<std::pair<std::string, OEChem::OEGraphMol>> cases{
        {"empty", empty_mol},
        {"[H][H]", mol_from_smiles("[H][H]")},
        {"C.CC", mol_from_smiles("C.CC")},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.first);
        const auto descriptors = MakeMordredDescriptors(expected.second);

        for (const auto& name : descriptor_names) {
            EXPECT_FALSE(descriptors.Has(name)) << name;
        }
    }
}

TEST(MordredDescriptorTest, TopologicalChargeDescriptorsReturnZeroForEmptySelections) {
    const std::vector<std::string> descriptor_names{
        "GGI1", "GGI2",  "GGI3",  "GGI4",  "GGI5",  "GGI6",  "GGI7",
        "GGI8", "GGI9",  "GGI10", "JGI1",  "JGI2",  "JGI3",  "JGI4",
        "JGI5", "JGI6",  "JGI7",  "JGI8",  "JGI9",  "JGI10", "JGT10",
    };

    const OEChem::OEGraphMol empty_mol;
    const std::vector<std::pair<std::string, OEChem::OEGraphMol>> cases{
        {"empty", empty_mol},
        {"[H][H]", mol_from_smiles("[H][H]")},
        {"C", mol_from_smiles("C")},
        {"CC", mol_from_smiles("CC")},
        {"c1ccncc1", mol_from_smiles("c1ccncc1")},
        {"C.CC", mol_from_smiles("C.CC")},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.first);
        const auto descriptors = MakeMordredDescriptors(expected.second);

        for (const auto& name : descriptor_names) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), 0.0, 1.0e-12) << name;
        }
    }
}

TEST(MordredDescriptorTest, TopologicalChargeDescriptorsUseHeavyAtomDistanceTerms) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "CCC",
            {
                {"GGI1", 0.5},
                {"GGI2", 0.0},
                {"GGI10", 0.0},
                {"JGI1", 0.25},
                {"JGI2", 0.0},
                {"JGI10", 0.0},
                {"JGT10", 0.25},
            },
        },
        {
            "CCC.C",
            {
                {"GGI1", 0.5},
                {"GGI2", 0.0},
                {"GGI10", 0.0},
                {"JGI1", 0.25},
                {"JGI2", 0.0},
                {"JGI10", 0.0},
                {"JGT10", 0.25},
            },
        },
        {
            "CCO",
            {
                {"GGI1", 0.5},
                {"GGI2", 0.0},
                {"GGI10", 0.0},
                {"JGI1", 0.25},
                {"JGI2", 0.0},
                {"JGI10", 0.0},
                {"JGT10", 0.25},
            },
        },
        {
            "CCCCCC",
            {
                {"GGI1", 0.5},
                {"GGI2", 0.22222222222222232},
                {"GGI3", 0.125},
                {"GGI4", 0.08000000000000002},
                {"GGI5", 0.0},
                {"GGI10", 0.0},
                {"JGI1", 0.1},
                {"JGI2", 0.05555555555555558},
                {"JGI3", 0.041666666666666664},
                {"JGI4", 0.04000000000000001},
                {"JGI5", 0.0},
                {"JGI10", 0.0},
                {"JGT10", 0.23722222222222228},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), value, 1.0e-12) << name;
        }
    }
}

TEST(MordredDescriptorTest, ABCIndexDescriptorsUseHeavyAtomGraphBonds) {
    struct Case {
        std::string smiles;
        double abc;
        double abcgg;
    };

    const std::vector<Case> cases{
        {"C", 0.0, 0.0},
        {"C.CC", 0.0, 0.0},
        {"CCO", 1.4142135623730951, 1.4142135623730951},
        {"c1ccncc1", 4.242640687119286, 3.9999999999999996},
        {"CCCCCC", 3.5355339059327378, 3.8697346110395934},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("ABC"));
        ASSERT_TRUE(descriptors.Has("ABCGG"));
        EXPECT_NEAR(descriptors.Float("ABC"), expected.abc, 1.0e-12);
        EXPECT_NEAR(descriptors.Float("ABCGG"), expected.abcgg, 1.0e-12);
    }
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

TEST(MordredDescriptorTest, PiPathCountDescriptorsMatchMordredPathSemantics) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "c1ccncc1",
            {
                {"piPC1", 2.302585092994046},
                {"piPC3", 3.056356895370426},
                {"piPC6", 0.0},
                {"piPC10", 0.0},
                {"TpiPC10", 4.833798667532871},
            },
        },
        {
            "CC#N",
            {
                {"piPC1", 1.6094379124341003},
                {"piPC2", 1.3862943611198906},
                {"piPC3", 0.0},
                {"piPC10", 0.0},
                {"TpiPC10", 2.3978952727983707},
            },
        },
        {
            "CCCCCC",
            {
                {"piPC1", 1.791759469228055},
                {"piPC3", 1.3862943611198906},
                {"piPC5", 0.6931471805599453},
                {"piPC10", 0.0},
                {"TpiPC10", 3.091042453358316},
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

TEST(MordredDescriptorTest, KappaShapeIndexDescriptorsMatchMordredPathSemantics) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
        std::vector<std::string> missing_values;
    };

    const std::vector<Case> cases{
        {
            "CCCC",
            {
                {"Kier1", 4.0},
                {"Kier2", 3.0},
                {"Kier3", 4.0},
            },
            {},
        },
        {
            "C1CCCCC1",
            {
                {"Kier1", 4.166666666666667},
                {"Kier2", 2.2222222222222223},
                {"Kier3", 1.3333333333333333},
            },
            {},
        },
        {
            "CC(C)(C)Cl",
            {
                {"Kier1", 5.0},
                {"Kier2", 1.0},
            },
            {"Kier3"},
        },
        {
            "C",
            {},
            {"Kier1", "Kier2", "Kier3"},
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
        }
        for (const auto& name : expected.missing_values) {
            EXPECT_FALSE(descriptors.Has(name)) << name;
        }
    }
}

TEST(MordredDescriptorTest, ChiPathDescriptorsMatchCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
        std::vector<std::string> missing_values;
    };

    const std::vector<Case> cases{
        {
            "CCO",
            {
                {"Xp-0d", 2.7071067811865475},
                {"Xp-1d", 1.4142135623730951},
                {"Xp-2d", 0.7071067811865476},
                {"Xp-3d", 0.0},
                {"AXp-0d", 0.9023689270621825},
                {"AXp-1d", 0.7071067811865476},
                {"AXp-2d", 0.7071067811865476},
                {"Xp-0dv", 2.1543203766865053},
                {"Xp-1dv", 1.0233345472033855},
                {"Xp-2dv", 0.31622776601683794},
                {"AXp-0dv", 0.7181067922288351},
                {"AXp-1dv", 0.5116672736016927},
                {"AXp-2dv", 0.31622776601683794},
            },
            {"AXp-3d", "AXp-3dv"},
        },
        {
            "c1ccncc1",
            {
                {"Xp-0d", 4.242640687119286},
                {"Xp-3d", 1.5},
                {"Xp-5d", 0.75},
                {"Xp-6d", 0.0},
                {"AXp-1d", 0.5},
                {"AXp-5d", 0.125},
                {"Xp-0dv", 3.3339649414480865},
                {"Xp-3dv", 0.5664874085517705},
                {"Xp-5dv", 0.17213259316477408},
                {"AXp-1dv", 0.3082885188046092},
                {"AXp-5dv", 0.028688765527462346},
            },
            {"AXp-6d", "AXp-7d", "AXp-6dv", "AXp-7dv"},
        },
        {
            "CCCC",
            {
                {"Xp-0d", 3.414213562373095},
                {"Xp-1d", 1.9142135623730951},
                {"Xp-2d", 1.0},
                {"Xp-3d", 0.5},
                {"Xp-4d", 0.0},
                {"AXp-0d", 0.8535533905932737},
                {"AXp-1d", 0.6380711874576984},
                {"AXp-2d", 0.5},
                {"AXp-3d", 0.5},
                {"Xp-0dv", 3.414213562373095},
                {"Xp-1dv", 1.9142135623730951},
                {"Xp-2dv", 1.0},
                {"Xp-3dv", 0.5},
                {"AXp-0dv", 0.8535533905932737},
                {"AXp-1dv", 0.6380711874576984},
                {"AXp-2dv", 0.5},
                {"AXp-3dv", 0.5},
            },
            {"AXp-4d", "AXp-4dv"},
        },
        {
            "CCCCCC",
            {
                {"Xp-0d", 4.82842712474619},
                {"Xp-5d", 0.25},
                {"Xp-6d", 0.0},
                {"AXp-0d", 0.8047378541243649},
                {"AXp-5d", 0.25},
                {"Xp-0dv", 4.82842712474619},
                {"Xp-5dv", 0.25},
                {"Xp-6dv", 0.0},
                {"AXp-0dv", 0.8047378541243649},
                {"AXp-5dv", 0.25},
            },
            {"AXp-6d", "AXp-7d", "AXp-6dv", "AXp-7dv"},
        },
        {
            "C",
            {
                {"Xp-1d", 0.0},
                {"Xp-7d", 0.0},
                {"Xp-1dv", 0.0},
                {"Xp-7dv", 0.0},
            },
            {"Xp-0d", "AXp-0d", "AXp-1d", "Xp-0dv", "AXp-0dv", "AXp-1dv"},
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
        }
        for (const auto& name : expected.missing_values) {
            EXPECT_FALSE(descriptors.Has(name)) << name;
        }
    }
}

TEST(MordredDescriptorTest, ChiValencePathDescriptorsDoNotDoubleCountExplicitHydrogens) {
    OEChem::OEGraphMol mol = mol_from_smiles("CCO");
    OEChem::OEAddExplicitHydrogens(mol);

    const auto descriptors = MakeMordredDescriptors(mol);

    const std::map<std::string, double> expected_values{
        {"Xp-0dv", 2.1543203766865053},
        {"Xp-1dv", 1.0233345472033855},
        {"Xp-2dv", 0.31622776601683794},
        {"AXp-0dv", 0.7181067922288351},
        {"AXp-1dv", 0.5116672736016927},
        {"AXp-2dv", 0.31622776601683794},
    };

    for (const auto& [name, value] : expected_values) {
        EXPECT_TRUE(descriptors.Has(name)) << name;
        EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
    }
}

TEST(MordredDescriptorTest, ChiNonPathDescriptorsMatchCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "c1ccncc1",
            {
                {"Xch-6d", 0.125},
            },
        },
        {
            "CC(C)(C)Cl",
            {
                {"Xc-3d", 2.0},
                {"Xc-4d", 0.5},
                {"Xc-3dv", 2.2008401285415227},
            },
        },
        {
            "O=S(=O)(N)C1=CC=CC=C1",
            {
                {"Xch-6d", 0.10206207261596575},
                {"Xc-3d", 1.510362971081845},
                {"Xpc-4d", 1.8618817185157401},
                {"Xpc-4dv", 0.9714045207910318},
            },
        },
        {
            "C1=CC2=C(C=C1)C=CC=C2",
            {
                {"Xch-7d", 0.23570226039551584},
                {"Xpc-6d", 2.121320343559642},
            },
        },
        {
            "C12C3C4C1C5C2C3C45",
            {
                {"Xch-4d", 0.6666666666666667},
                {"Xc-3d", 0.8888888888888891},
                {"Xpc-4d", 3.079201435678001},
            },
        },
        {
            "C1CC1C",
            {
                {"Xch-3d", 0.28867513459481287},
                {"Xc-3d", 0.28867513459481287},
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

TEST(MordredDescriptorTest, ChiNonPathDescriptorsReturnZeroForEmptyClasses) {
    const auto descriptors = MakeMordredDescriptors(mol_from_smiles("CCCC"));
    const std::vector<std::string> empty_non_path_names{
        "Xch-3d", "Xch-7d", "Xc-3d", "Xc-6d", "Xpc-4d", "Xpc-6d",
        "Xch-3dv", "Xch-7dv", "Xc-3dv", "Xc-6dv", "Xpc-4dv", "Xpc-6dv",
    };

    for (const auto& name : empty_non_path_names) {
        EXPECT_TRUE(descriptors.Has(name)) << name;
        EXPECT_DOUBLE_EQ(descriptors.Float(name), 0.0) << name;
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
