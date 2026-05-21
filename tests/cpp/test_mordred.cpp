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

DescriptorSet MakeMordredDetourDescriptorsForTesting(
    const OEChem::OEMolBase& mol,
    std::uint64_t max_search_operations);

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

const std::vector<std::string>& estate_count_names() {
    static const std::vector<std::string> names{
        "NsLi",    "NssBe",    "NssssBe", "NssBH",    "NsssB",   "NssssB",
        "NsCH3",   "NdCH2",    "NssCH2",  "NtCH",     "NdsCH",   "NaaCH",
        "NsssCH",  "NddC",     "NtsC",    "NdssC",    "NaasC",   "NaaaC",
        "NssssC",  "NsNH3",    "NsNH2",   "NssNH2",   "NdNH",    "NssNH",
        "NaaNH",   "NtN",      "NsssNH",  "NdsN",     "NaaN",    "NsssN",
        "NddsN",   "NaasN",    "NssssN",  "NsOH",     "NdO",     "NssO",
        "NaaO",    "NsF",      "NsSiH3",  "NssSiH2",  "NsssSiH", "NssssSi",
        "NsPH2",   "NssPH",    "NsssP",   "NdsssP",   "NsssssP", "NsSH",
        "NdS",     "NssS",     "NaaS",    "NdssS",    "NddssS",  "NsCl",
        "NsGeH3",  "NssGeH2",  "NsssGeH", "NssssGe",  "NsAsH2",  "NssAsH",
        "NsssAs",  "NsssdAs",  "NsssssAs", "NsSeH",   "NdSe",    "NssSe",
        "NaaSe",   "NdssSe",   "NddssSe", "NsBr",     "NsSnH3",  "NssSnH2",
        "NsssSnH", "NssssSn",  "NsI",     "NsPbH3",   "NssPbH2", "NsssPbH",
        "NssssPb",
    };
    return names;
}

const std::vector<std::string>& estate_sum_names() {
    static const std::vector<std::string> names = [] {
        auto sum_names = estate_count_names();
        for (auto& name : sum_names) {
            name[0] = 'S';
        }
        return sum_names;
    }();
    return names;
}

const std::vector<std::string>& estate_max_names() {
    static const std::vector<std::string> names = [] {
        auto max_names = estate_count_names();
        for (auto& name : max_names) {
            name = "MAX" + name.substr(1);
        }
        return max_names;
    }();
    return names;
}

const std::vector<std::string>& estate_min_names() {
    static const std::vector<std::string> names = [] {
        auto min_names = estate_count_names();
        for (auto& name : min_names) {
            name = "MIN" + name.substr(1);
        }
        return min_names;
    }();
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
    EXPECT_TRUE(descriptors.Has("SpAbs_Dt"));
    EXPECT_TRUE(descriptors.Has("VE1_A"));
    EXPECT_TRUE(descriptors.Has("VE1_D"));
    EXPECT_TRUE(descriptors.Has("VE1_Dt"));
    EXPECT_TRUE(descriptors.Has("VR1_A"));
    EXPECT_TRUE(descriptors.Has("VR1_D"));
    EXPECT_TRUE(descriptors.Has("VR1_Dt"));
    EXPECT_TRUE(descriptors.Has("DetourIndex"));
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

TEST(MordredDescriptorTest, EStateCountDescriptorsMatchRepresentativeTypes) {
    struct Case {
        std::string smiles;
        std::map<std::string, std::uint32_t> nonzero;
    };

    const std::vector<Case> cases{
        {
            "CCO",
            {
                {"NsCH3", 1},
                {"NssCH2", 1},
                {"NsOH", 1},
            },
        },
        {
            "c1ccncc1",
            {
                {"NaaCH", 5},
                {"NaaN", 1},
            },
        },
        {
            "CC(C)(C)Cl",
            {
                {"NsCH3", 3},
                {"NssssC", 1},
                {"NsCl", 1},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& name : estate_count_names()) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_EQ(
                descriptors.Int(name),
                static_cast<std::int64_t>(count_or_zero(expected.nonzero, name)))
                << name;
        }
    }
}

TEST(MordredDescriptorTest, EStateCountDescriptorsTreatExplicitHydrogensAsImplicit) {
    OEChem::OEGraphMol explicit_mol = mol_from_smiles("CCO");
    OEChem::OEAddExplicitHydrogens(explicit_mol);

    const auto implicit_descriptors = MakeMordredDescriptors(mol_from_smiles("CCO"));
    const auto explicit_descriptors = MakeMordredDescriptors(explicit_mol);

    for (const auto& name : estate_count_names()) {
        EXPECT_TRUE(explicit_descriptors.Has(name)) << name;
        EXPECT_EQ(explicit_descriptors.Int(name), implicit_descriptors.Int(name)) << name;
    }
}

TEST(MordredDescriptorTest, EStateSumDescriptorsMatchRepresentativeTypes) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> nonzero;
    };

    const std::vector<Case> cases{
        {
            "CCO",
            {
                {"SsCH3", 1.6805555555555556},
                {"SssCH2", 0.25},
                {"SsOH", 7.569444444444445},
            },
        },
        {
            "c1ccncc1",
            {
                {"SaaCH", 9.215277777777779},
                {"SaaN", 3.7847222222222223},
            },
        },
        {
            "CC(C)(C)Cl",
            {
                {"SsCH3", 5.858796296296297},
                {"SssssC", -0.02777777777777768},
                {"SsCl", 5.530092592592592},
            },
        },
        {
            "CC#N",
            {
                {"SsCH3", 1.4305555555555556},
                {"StN", 7.319444444444445},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, expected_value] : expected.nonzero) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), expected_value, 1.0e-12) << name;
        }
    }
}

TEST(MordredDescriptorTest, EStateSumDescriptorsUseZeroForAbsentTypes) {
    const auto descriptors = MakeMordredDescriptors(mol_from_smiles("C"));

    for (const auto& name : estate_sum_names()) {
        EXPECT_TRUE(descriptors.Has(name)) << name;
        EXPECT_NEAR(descriptors.Float(name), 0.0, 1.0e-12) << name;
    }
}

TEST(MordredDescriptorTest, EStateSumDescriptorsTreatExplicitHydrogensAsImplicit) {
    OEChem::OEGraphMol explicit_mol = mol_from_smiles("CCO");
    OEChem::OEAddExplicitHydrogens(explicit_mol);

    const auto implicit_descriptors = MakeMordredDescriptors(mol_from_smiles("CCO"));
    const auto explicit_descriptors = MakeMordredDescriptors(explicit_mol);

    for (const auto& name : estate_sum_names()) {
        EXPECT_TRUE(explicit_descriptors.Has(name)) << name;
        EXPECT_NEAR(explicit_descriptors.Float(name), implicit_descriptors.Float(name), 1.0e-12)
            << name;
    }
}

TEST(MordredDescriptorTest, EStateExtremaDescriptorsMatchRepresentativeTypes) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "CCO",
            {
                {"MAXsCH3", 1.6805555555555556},
                {"MINsCH3", 1.6805555555555556},
                {"MAXssCH2", 0.25},
                {"MINssCH2", 0.25},
                {"MAXsOH", 7.569444444444445},
                {"MINsOH", 7.569444444444445},
            },
        },
        {
            "c1ccncc1",
            {
                {"MAXaaCH", 1.9375},
                {"MINaaCH", 1.75},
                {"MAXaaN", 3.7847222222222223},
                {"MINaaN", 3.7847222222222223},
            },
        },
        {
            "CC(C)(C)Cl",
            {
                {"MAXsCH3", 1.9529320987654322},
                {"MINsCH3", 1.9529320987654322},
                {"MAXssssC", -0.02777777777777768},
                {"MINssssC", -0.02777777777777768},
                {"MAXsCl", 5.530092592592592},
                {"MINsCl", 5.530092592592592},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, expected_value] : expected.expected_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), expected_value, 1.0e-12) << name;
        }
    }
}

TEST(MordredDescriptorTest, EStateExtremaDescriptorsAreMissingForAbsentTypes) {
    const auto methane_descriptors = MakeMordredDescriptors(mol_from_smiles("C"));
    const auto ethanol_descriptors = MakeMordredDescriptors(mol_from_smiles("CCO"));

    for (const auto& name : estate_max_names()) {
        EXPECT_FALSE(methane_descriptors.Has(name)) << name;
    }
    for (const auto& name : estate_min_names()) {
        EXPECT_FALSE(methane_descriptors.Has(name)) << name;
    }
    EXPECT_FALSE(ethanol_descriptors.Has("MAXsLi"));
    EXPECT_FALSE(ethanol_descriptors.Has("MINsLi"));
}

TEST(MordredDescriptorTest, EStateExtremaDescriptorsTreatExplicitHydrogensAsImplicit) {
    OEChem::OEGraphMol explicit_mol = mol_from_smiles("CCO");
    OEChem::OEAddExplicitHydrogens(explicit_mol);

    const auto implicit_descriptors = MakeMordredDescriptors(mol_from_smiles("CCO"));
    const auto explicit_descriptors = MakeMordredDescriptors(explicit_mol);

    for (const auto& name : estate_max_names()) {
        EXPECT_EQ(explicit_descriptors.Has(name), implicit_descriptors.Has(name)) << name;
        if (implicit_descriptors.Has(name)) {
            EXPECT_NEAR(explicit_descriptors.Float(name), implicit_descriptors.Float(name), 1.0e-12)
                << name;
        }
    }
    for (const auto& name : estate_min_names()) {
        EXPECT_EQ(explicit_descriptors.Has(name), implicit_descriptors.Has(name)) << name;
        if (implicit_descriptors.Has(name)) {
            EXPECT_NEAR(explicit_descriptors.Float(name), implicit_descriptors.Float(name), 1.0e-12)
                << name;
        }
    }
}

TEST(MordredDescriptorTest, InformationContentDescriptorsMatchCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "C",
            {
                {"IC0", 0.7219280948873623},
                {"IC1", 0.7219280948873623},
                {"IC5", 0.7219280948873623},
                {"TIC0", 3.6096404744368114},
                {"TIC5", 3.6096404744368114},
                {"SIC0", 0.31091750708257115},
                {"BIC0", 0.36096404744368116},
                {"CIC0", 1.5999999999999999},
                {"MIC0", 5.837338485255589},
                {"MIC5", 5.837338485255589},
                {"ZMIC0", 3.816483617504394},
                {"ZMIC5", 3.816483617504394},
            },
        },
        {
            "CCO",
            {
                {"IC0", 1.224394445405986},
                {"IC1", 1.8799649487271108},
                {"IC2", 2.4193819456463714},
                {"IC5", 2.4193819456463714},
                {"TIC0", 11.019550008653875},
                {"TIC1", 16.919684538543997},
                {"TIC5", 21.774437510817343},
                {"SIC0", 0.3862534428571301},
                {"SIC1", 0.5930629109116868},
                {"SIC5", 0.7632300273809492},
                {"BIC0", 0.4081314818019954},
                {"BIC1", 0.626654982909037},
                {"BIC5", 0.8064606485487905},
                {"CIC0", 1.945530556036326},
                {"CIC1", 1.2899600527152013},
                {"CIC5", 0.7505430557959407},
                {"MIC0", 11.81993574300937},
                {"MIC1", 14.925861921468176},
                {"MIC5", 15.46959425436279},
                {"ZMIC0", 10.944027785790624},
                {"ZMIC1", 9.7520386326847},
                {"ZMIC5", 9.945865282505357},
            },
        },
        {
            "c1ccncc1",
            {
                {"IC0", 1.3485878960124222},
                {"IC1", 1.971747257128181},
                {"IC2", 3.0271691184406184},
                {"IC5", 3.4594316186372978},
                {"TIC0", 14.834466856136645},
                {"TIC1", 21.689219828409993},
                {"TIC5", 38.053747805010275},
                {"SIC0", 0.3898293259352366},
                {"SIC1", 0.5699627784245295},
                {"SIC5", 1.0000000000000002},
                {"BIC0", 0.3542059838444498},
                {"BIC1", 0.5178785002955785},
                {"BIC5", 0.9086181061280522},
                {"CIC0", 2.110843722624875},
                {"CIC1", 1.4876843615091162},
                {"CIC5", -4.440892098500626e-16},
                {"MIC0", 11.136550050977693},
                {"MIC1", 18.62131713733907},
                {"MIC5", 24.87708726340432},
                {"ZMIC0", 20.298103453336335},
                {"ZMIC1", 17.762556474120785},
                {"ZMIC5", 13.208738907524227},
            },
        },
        {
            "C[Na]",
            {
                {"IC0", 1.3709505944546687},
                {"IC5", 1.3709505944546687},
                {"TIC0", 6.854752972273344},
                {"SIC0", 0.5904362833084089},
                {"BIC0", 0.6854752972273344},
                {"CIC0", 0.9509775004326935},
                {"MIC0", 16.699677841182233},
                {"ZMIC0", 9.221093592116203},
            },
        },
        {
            "[13CH4]",
            {
                {"IC0", 0.7219280948873623},
                {"IC1", 0.7219280948873623},
                {"TIC0", 3.6096404744368114},
                {"SIC0", 0.31091750708257115},
                {"BIC0", 0.36096404744368116},
                {"CIC0", 1.5999999999999999},
                {"MIC0", 6.298173801874281},
                {"ZMIC0", 3.816483617504394},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, expected_value] : expected.expected_values) {
            ASSERT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), expected_value, 1.0e-12) << name;
        }
    }
}

TEST(MordredDescriptorTest, InformationContentDescriptorsAreMissingForInvalidDenominators) {
    const OEChem::OEGraphMol empty_mol;
    const auto empty_descriptors = MakeMordredDescriptors(empty_mol);

    EXPECT_FALSE(empty_descriptors.Has("IC0"));
    EXPECT_FALSE(empty_descriptors.Has("TIC0"));
    EXPECT_FALSE(empty_descriptors.Has("SIC0"));
    EXPECT_FALSE(empty_descriptors.Has("BIC0"));
    EXPECT_FALSE(empty_descriptors.Has("CIC0"));
    EXPECT_FALSE(empty_descriptors.Has("MIC0"));
    EXPECT_FALSE(empty_descriptors.Has("ZMIC0"));

    const auto helium_descriptors = MakeMordredDescriptors(mol_from_smiles("[He]"));

    EXPECT_TRUE(helium_descriptors.Has("IC0"));
    EXPECT_TRUE(helium_descriptors.Has("TIC0"));
    EXPECT_FALSE(helium_descriptors.Has("SIC0"));
    EXPECT_FALSE(helium_descriptors.Has("BIC0"));
    EXPECT_TRUE(helium_descriptors.Has("CIC0"));
    EXPECT_TRUE(helium_descriptors.Has("MIC0"));
    EXPECT_TRUE(helium_descriptors.Has("ZMIC0"));
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

TEST(MordredDescriptorTest, LabuteASAMatchesCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        double expected_value;
    };

    const std::vector<Case> cases{
        {"C", 8.739251027829551},
        {"CC", 15.104193142226158},
        {"CCO", 19.89842689442217},
        {"CC(=O)O", 24.059948994465934},
        {"c1ccncc1", 36.651051100443475},
        {"O=S(=O)(N)C1=CC=CC=C1", 59.53206970892476},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("LabuteASA"));
        EXPECT_NEAR(descriptors.Float("LabuteASA"), expected.expected_value, 1.0e-8);
    }
}

TEST(MordredDescriptorTest, LabuteASATreatsExplicitHydrogensAsImplicit) {
    OEChem::OEGraphMol explicit_mol = mol_from_smiles("CCO");
    OEChem::OEAddExplicitHydrogens(explicit_mol);

    const auto implicit_descriptors = MakeMordredDescriptors(mol_from_smiles("CCO"));
    const auto explicit_descriptors = MakeMordredDescriptors(explicit_mol);

    ASSERT_TRUE(implicit_descriptors.Has("LabuteASA"));
    ASSERT_TRUE(explicit_descriptors.Has("LabuteASA"));
    EXPECT_NEAR(
        explicit_descriptors.Float("LabuteASA"),
        implicit_descriptors.Float("LabuteASA"),
        1.0e-12);
}

TEST(MordredDescriptorTest, LabuteASAIsMissingForNonFiniteDummyAtomSurface) {
    const auto descriptors = MakeMordredDescriptors(mol_from_smiles("**"));

    EXPECT_FALSE(descriptors.Has("LabuteASA"));
}

TEST(MordredDescriptorTest, FragmentComplexityMatchesCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        double expected_value;
    };

    const std::vector<Case> cases{
        {"C", 0.0},
        {"CCO", 2.01},
        {"CC(C)(C)Cl", 4.01},
        {"C1=CC2=C(C=C1)C=CC=C2", 31.0},
        {"O=S(=O)(N)C1=CC=CC=C1", 10.04},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("fragCpx"));
        EXPECT_NEAR(descriptors.Float("fragCpx"), expected.expected_value, 1.0e-12);
    }
}

TEST(MordredDescriptorTest, MolecularIdDescriptorsMatchCopiedMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "C",
            {
                {"MID", 1.0},
                {"AMID", 1.0},
                {"MID_h", 0.0},
                {"AMID_h", 0.0},
                {"MID_C", 1.0},
                {"AMID_C", 1.0},
                {"MID_N", 0.0},
                {"AMID_N", 0.0},
                {"MID_O", 0.0},
                {"AMID_O", 0.0},
                {"MID_X", 0.0},
                {"AMID_X", 0.0},
            },
        },
        {
            "CCO",
            {
                {"MID", 4.914213562373095},
                {"AMID", 1.6380711874576983},
                {"MID_h", 1.6035533905932737},
                {"AMID_h", 0.5345177968644246},
                {"MID_C", 3.310660171779821},
                {"AMID_C", 1.1035533905932737},
                {"MID_N", 0.0},
                {"AMID_N", 0.0},
                {"MID_O", 1.6035533905932737},
                {"AMID_O", 0.5345177968644246},
                {"MID_X", 0.0},
                {"AMID_X", 0.0},
            },
        },
        {
            "c1ccncc1",
            {
                {"MID", 11.8125},
                {"AMID", 1.96875},
                {"MID_h", 1.96875},
                {"AMID_h", 0.328125},
                {"MID_C", 9.84375},
                {"AMID_C", 1.640625},
                {"MID_N", 1.96875},
                {"AMID_N", 0.328125},
                {"MID_O", 0.0},
                {"AMID_O", 0.0},
                {"MID_X", 0.0},
                {"AMID_X", 0.0},
            },
        },
        {
            "CC(C)(C)Cl",
            {
                {"MID", 8.5},
                {"AMID", 1.7},
                {"MID_h", 1.625},
                {"AMID_h", 0.325},
                {"MID_C", 6.875},
                {"AMID_C", 1.375},
                {"MID_X", 1.625},
                {"AMID_X", 0.325},
            },
        },
        {
            "O=S(=O)(N)C1=CC=CC=C1",
            {
                {"MID", 19.299499452491947},
                {"AMID", 1.9299499452491946},
                {"MID_h", 7.181685330138905},
                {"AMID_h", 0.7181685330138905},
                {"MID_C", 12.117814122353042},
                {"AMID_C", 1.2117814122353043},
                {"MID_N", 1.6863370660277812},
                {"AMID_N", 0.16863370660277813},
                {"MID_O", 3.3726741320555624},
                {"AMID_O", 0.33726741320555625},
            },
        },
        {
            "CI",
            {
                {"MID", 3.0},
                {"AMID", 1.5},
                {"MID_h", 1.5},
                {"AMID_h", 0.75},
                {"MID_C", 1.5},
                {"AMID_C", 0.75},
                {"MID_X", 1.5},
                {"AMID_X", 0.75},
            },
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, expected_value] : expected.expected_values) {
            ASSERT_TRUE(descriptors.Has(name)) << name;
            EXPECT_NEAR(descriptors.Float(name), expected_value, 1.0e-12) << name;
        }
    }
}

TEST(MordredDescriptorTest, MolecularIdDescriptorsAreMissingForDisconnectedHeavyAtomGraphs) {
    const std::vector<std::string> names{
        "MID",   "AMID",   "MID_h", "AMID_h", "MID_C", "AMID_C",
        "MID_N", "AMID_N", "MID_O", "AMID_O", "MID_X", "AMID_X",
    };
    const auto descriptors = MakeMordredDescriptors(mol_from_smiles("C.CC"));

    for (const auto& name : names) {
        EXPECT_FALSE(descriptors.Has(name)) << name;
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

TEST(MordredDescriptorTest, FrameworkDescriptorMatchesMordredReferences) {
    struct Case {
        std::string smiles;
        double expected_value;
    };

    const std::vector<Case> cases{
        {"CCO", 0.0},
        {"C1CCCCC1", 0.3333333333333333},
        {"C1=CC2=C(C=C1)C=CC=C2", 0.5555555555555556},
        {"C12C3C4C1C5C2C3C45", 0.5},
        {"FC(F)(F)c1ccc(Br)cc1", 0.4},
        {"c1ccccc1CCc2ccccc2", 0.5},
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        ASSERT_TRUE(descriptors.Has("fMF"));
        EXPECT_NEAR(descriptors.Float("fMF"), expected.expected_value, 1.0e-12);
    }
}

TEST(MordredDescriptorTest, FrameworkDescriptorDoesNotDoubleCountExplicitHydrogens) {
    OEChem::OEGraphMol mol = mol_from_smiles("C1CCCCC1");
    OEChem::OEAddExplicitHydrogens(mol);

    const auto descriptors = MakeMordredDescriptors(mol);

    ASSERT_TRUE(descriptors.Has("fMF"));
    EXPECT_NEAR(descriptors.Float("fMF"), 0.3333333333333333, 1.0e-12);
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

TEST(MordredDescriptorTest, DetourMatrixDescriptorsMatchMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_float_values;
        std::int64_t expected_detour_index;
    };

    const std::vector<Case> cases{
        {
            "C",
            {
                {"SpAbs_Dt", 0.0},
                {"SpMax_Dt", 0.0},
                {"SpDiam_Dt", 0.0},
                {"SpAD_Dt", 0.0},
                {"SpMAD_Dt", 0.0},
                {"LogEE_Dt", 0.6931471805599453},
                {"SM1_Dt", 0.0},
                {"VE1_Dt", 1.0},
                {"VE2_Dt", 1.0},
                {"VE3_Dt", -2.3025850929940455},
                {"VR1_Dt", 0.0},
                {"VR2_Dt", 0.0},
            },
            0,
        },
        {
            "CCO",
            {
                {"SpAbs_Dt", 5.464101615137755},
                {"SpMax_Dt", 2.7320508075688776},
                {"SpDiam_Dt", 4.732050807568878},
                {"SpAD_Dt", 5.464101615137755},
                {"SpMAD_Dt", 1.8213672050459184},
                {"LogEE_Dt", 2.832072756761435},
                {"SM1_Dt", 0.0},
                {"VE1_Dt", 1.7156269037800915},
                {"VE2_Dt", 0.5718756345933639},
                {"VE3_Dt", -0.6641942489393373},
                {"VR1_Dt", 3.7224194364083996},
                {"VR2_Dt", 1.2408064788028},
                {"VR3_Dt", 0.11040103868100991},
            },
            4,
        },
        {
            "C1CCCCC1",
            {
                {"SpAbs_Dt", 42.0},
                {"SpMax_Dt", 21.0},
                {"SpDiam_Dt", 27.0},
                {"SpAD_Dt", 42.0},
                {"SpMAD_Dt", 7.0},
                {"LogEE_Dt", 21.000000000972364},
                {"SM1_Dt", 0.0},
                {"VE1_Dt", 2.4494897427831788},
                {"VE2_Dt", 0.40824829046386313},
                {"VE3_Dt", 0.3850541108480373},
                {"VR1_Dt", 14.696938456699067},
                {"VR2_Dt", 2.449489742783178},
                {"VR3_Dt", 2.1768135800760917},
            },
            63,
        },
        {
            "c1ccncc1",
            {
                {"SpAbs_Dt", 42.0},
                {"SpMax_Dt", 21.0},
                {"SpDiam_Dt", 27.0},
                {"SpAD_Dt", 42.0},
                {"SpMAD_Dt", 7.0},
                {"LogEE_Dt", 21.000000000972364},
                {"SM1_Dt", 0.0},
                {"VE1_Dt", 2.4494897427831788},
                {"VE2_Dt", 0.40824829046386313},
                {"VE3_Dt", 0.3850541108480373},
                {"VR1_Dt", 14.696938456699067},
                {"VR2_Dt", 2.449489742783178},
                {"VR3_Dt", 2.1768135800760917},
            },
            63,
        },
        {
            "C12C3C4C1C5C2C3C45",
            {
                {"SpAbs_Dt", 91.99999999999999},
                {"SpMax_Dt", 45.999999999999986},
                {"SpDiam_Dt", 55.999999999999986},
                {"SpAD_Dt", 91.99999999999997},
                {"SpMAD_Dt", 11.499999999999996},
                {"LogEE_Dt", 45.999999999999986},
                {"SM1_Dt", 0.0},
                {"VE1_Dt", 2.8284271247461907},
                {"VE2_Dt", 0.35355339059327384},
                {"VE3_Dt", 0.8165772195257084},
                {"VR1_Dt", 33.941125496954264},
                {"VR2_Dt", 4.242640687119283},
                {"VR3_Dt", 3.3014838693137083},
            },
            184,
        },
    };

    for (const auto& expected : cases) {
        SCOPED_TRACE(expected.smiles);
        const auto descriptors = MakeMordredDescriptors(mol_from_smiles(expected.smiles));

        for (const auto& [name, value] : expected.expected_float_values) {
            EXPECT_TRUE(descriptors.Has(name)) << name;
            if (descriptors.Has(name)) {
                EXPECT_NEAR(descriptors.Float(name), value, 1.0e-8) << name;
            }
        }
        ASSERT_TRUE(descriptors.Has("DetourIndex"));
        EXPECT_EQ(descriptors.Int("DetourIndex"), expected.expected_detour_index);
        if (expected.smiles == "C") {
            EXPECT_FALSE(descriptors.Has("VR3_Dt"));
        }
    }
}

TEST(MordredDescriptorTest, DetourMatrixDescriptorsAreMissingWhenRequiredMatrixIsMissing) {
    const std::vector<std::string> descriptor_names{
        "SpAbs_Dt",
        "SpMax_Dt",
        "SpDiam_Dt",
        "SpAD_Dt",
        "SpMAD_Dt",
        "LogEE_Dt",
        "SM1_Dt",
        "VE1_Dt",
        "VE2_Dt",
        "VE3_Dt",
        "VR1_Dt",
        "VR2_Dt",
        "VR3_Dt",
        "DetourIndex",
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

TEST(MordredDescriptorTest, DetourMatrixDescriptorsAreMissingWhenSearchBudgetIsExceeded) {
    const auto descriptors =
        MakeMordredDetourDescriptorsForTesting(mol_from_smiles("C1CCCCC1"), 1u);
    const std::vector<std::string> descriptor_names{
        "SpAbs_Dt",
        "SpMax_Dt",
        "SpDiam_Dt",
        "SpAD_Dt",
        "SpMAD_Dt",
        "LogEE_Dt",
        "SM1_Dt",
        "VE1_Dt",
        "VE2_Dt",
        "VE3_Dt",
        "VR1_Dt",
        "VR2_Dt",
        "VR3_Dt",
        "DetourIndex",
    };

    for (const auto& name : descriptor_names) {
        EXPECT_FALSE(descriptors.Has(name)) << name;
    }
}

TEST(MordredDescriptorTest, BaryszMatrixDescriptorsMatchMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
    };

    const std::vector<Case> cases{
        {
            "C",
            {
                {"SpAbs_DzZ", 0.0},
                {"SpMax_DzZ", 0.0},
                {"SpDiam_DzZ", 0.0},
                {"SpAD_DzZ", 0.0},
                {"SpMAD_DzZ", 0.0},
                {"LogEE_DzZ", 0.6931471805599453},
                {"SM1_DzZ", 0.0},
                {"VE1_DzZ", 1.0},
                {"VE2_DzZ", 1.0},
                {"VE3_DzZ", -2.3025850929940455},
                {"VR1_DzZ", 0.0},
                {"VR2_DzZ", 0.0},
            },
        },
        {
            "CCO",
            {
                {"SpAbs_DzZ", 4.730475337133475},
                {"SpMax_DzZ", 2.4902376685667376},
                {"SpDiam_DzZ", 4.158938287189794},
                {"SpAD_DzZ", 4.813808670466809},
                {"SpMAD_DzZ", 1.604602890155603},
                {"LogEE_DzZ", 2.625920832582195},
                {"SM1_DzZ", 0.25},
                {"VE1_DzZ", 1.7112782949565715},
                {"VE2_DzZ", 0.5704260983188572},
                {"VE3_DzZ", -0.6667321721706297},
                {"VR1_DzZ", 3.76932889054404},
                {"VR2_DzZ", 1.25644296351468},
                {"VR3_DzZ", 0.12292416816947607},
                {"SpAbs_Dzm", 4.732622816951536},
                {"SpMax_Dzm", 2.4909441980251152},
                {"SpDiam_Dzm", 4.160470659917556},
                {"SpAD_Dzm", 4.815711343317767},
                {"SpMAD_Dzm", 1.6052371144392559},
                {"LogEE_Dzm", 2.626501375966335},
                {"SM1_Dzm", 0.24926557909869373},
                {"VE1_Dzm", 1.711294139372221},
                {"VE2_Dzm", 0.5704313797907403},
                {"VE3_Dzm", -0.6667229133946923},
                {"VR1_Dzm", 3.769161797350935},
                {"VR2_Dzm", 1.256387265783645},
                {"VR3_Dzm", 0.1228798374937347},
                {"SpAbs_Dzv", 6.642353014604328},
                {"SpMax_Dzv", 3.1216792682731516},
                {"SpDiam_Dzv", 5.787958588915702},
                {"SpAD_Dzv", 6.509354855251654},
                {"SpMAD_Dzv", 2.169784951750551},
                {"LogEE_Dzv", 3.185504840783211},
                {"SM1_Dzv", -0.3989944780580257},
                {"VE1_Dzv", 1.7199058636419866},
                {"VE2_Dzv", 0.5733019545473289},
                {"VE3_Dzv", -0.6617032454390448},
                {"VR1_Dzv", 3.6735819004455097},
                {"VR2_Dzv", 1.2245273001485033},
                {"VR3_Dzv", 0.09719437643921745},
                {"SpAbs_Dzse", 4.734876666774441},
                {"SpMax_Dzse", 2.4916857334966895},
                {"SpDiam_Dzse", 4.162079956498632},
                {"SpAD_Dzse", 4.81770826684742},
                {"SpMAD_Dzse", 1.6059027556158068},
                {"LogEE_Dzse", 2.6271108444592195},
                {"SM1_Dzse", 0.2484948002189381},
                {"VE1_Dzse", 1.7113107440203548},
                {"VE2_Dzse", 0.5704369146734516},
                {"VE3_Dzse", -0.6667132104644975},
                {"VR1_Dzse", 3.768986660326327},
                {"VR2_Dzse", 1.2563288867754423},
                {"VR3_Dzse", 0.1228333706390096},
                {"SpAbs_Dzpe", 4.704979072146131},
                {"SpMax_Dzpe", 2.4818500011893443},
                {"SpDiam_Dzpe", 4.140817758452723},
                {"SpAD_Dzpe", 4.79121938222365},
                {"SpMAD_Dzpe", 1.5970731274078833},
                {"LogEE_Dzpe", 2.6190403624769565},
                {"SM1_Dzpe", 0.25872093023255816},
                {"VE1_Dzpe", 1.7110884153153094},
                {"VE2_Dzpe", 0.5703628051051032},
                {"VE3_Dzpe", -0.6668431361073036},
                {"VR1_Dzpe", 3.7713294262855883},
                {"VR2_Dzpe", 1.2571098087618628},
                {"VR3_Dzpe", 0.1234547679405911},
                {"SpAbs_Dzare", 4.626111677620375},
                {"SpMax_Dzare", 2.45591298166733},
                {"SpDiam_Dzare", 4.085623233525894},
                {"SpAD_Dzare", 4.72134977285847},
                {"SpMAD_Dzare", 1.57378325761949},
                {"LogEE_Dzare", 2.597902051813245},
                {"SM1_Dzare", 0.2857142857142857},
                {"VE1_Dzare", 1.7104796754470062},
                {"VE2_Dzare", 0.5701598918156687},
                {"VE3_Dzare", -0.667198961190177},
                {"VR1_Dzare", 3.7777197310165604},
                {"VR2_Dzare", 1.2592399103388534},
                {"VR3_Dzare", 0.1251477775168126},
                {"SpAbs_Dzp", 8.671219128493659},
                {"SpMax_Dzp", 3.7944624320772533},
                {"SpDiam_Dzp", 7.7419816714401195},
                {"SpAD_Dzp", 8.310454373713942},
                {"SpMAD_Dzp", 2.7701514579046473},
                {"LogEE_Dzp", 3.8257783883922456},
                {"SM1_Dzp", -1.082294264339152},
                {"VE1_Dzp", 1.7239111927563364},
                {"VE2_Dzp", 0.5746370642521121},
                {"VE3_Dzp", -0.6593771457335504},
                {"VR1_Dzp", 3.6241090761576578},
                {"VR2_Dzp", 1.2080363587192193},
                {"VR3_Dzp", 0.08363568167935242},
                {"SpAbs_Dzi", 4.955509714837185},
                {"SpMax_Dzi", 2.5643219503577432},
                {"SpDiam_Dzi", 4.324553093799054},
                {"SpAD_Dzi", 5.013221110129952},
                {"SpMAD_Dzi", 1.6710737033766507},
                {"LogEE_Dzi", 2.687570851662355},
                {"SM1_Dzi", 0.1731341858783012},
                {"VE1_Dzi", 1.7128230733907528},
                {"VE2_Dzi", 0.5709410244635843},
                {"VE3_Dzi", -0.6658298749703967},
                {"VR1_Dzi", 3.752916640530884},
                {"VR2_Dzi", 1.2509722135102945},
                {"VR3_Dzi", 0.11856050415764435},
            },
        },
        {
            "c1ccncc1",
            {
                {"SpAbs_DzZ", 11.261331130232497},
                {"SpMax_DzZ", 5.676981473926306},
                {"SpDiam_DzZ", 8.210668066766813},
                {"SpAD_DzZ", 11.308950177851543},
                {"SpMAD_DzZ", 1.8848250296419238},
                {"LogEE_DzZ", 5.689526220366282},
                {"SM1_DzZ", 0.1428571428571429},
                {"VE1_DzZ", 2.449067149645903},
                {"VE2_DzZ", 0.4081778582743172},
                {"VE3_DzZ", 0.3848815730383908},
                {"VR1_DzZ", 14.703129179892782},
                {"VR2_DzZ", 2.4505215299821304},
                {"VR3_DzZ", 2.17723471674569},
                {"SpAbs_Dzm", 11.263113736984625},
                {"SpMax_Dzm", 5.677782465895757},
                {"SpDiam_Dzm", 8.211775542111347},
                {"SpAD_Dzm", 11.310613796478687},
                {"SpMAD_Dzm", 1.8851022994131146},
                {"LogEE_Dzm", 5.690316735415072},
                {"SM1_Dzm", 0.14250017848218755},
                {"VE1_Dzm", 2.4490693384849758},
                {"VE2_Dzm", 0.4081782230808293},
                {"VE3_Dzm", 0.3848824667819922},
                {"VR1_Dzm", 14.7030971719931},
                {"VR2_Dzm", 2.4505161953321832},
                {"VR3_Dzm", 2.177232539798583},
                {"SpAbs_Dzv", 13.218419024183428},
                {"SpMax_Dzv", 6.4495471979699985},
                {"SpDiam_Dzv", 9.37173082586626},
                {"SpAD_Dzv", 13.005535938687807},
                {"SpMAD_Dzv", 2.1675893231146346},
                {"LogEE_Dzv", 6.454819011453754},
                {"SM1_Dzv", -0.3193246282434288},
                {"VE1_Dzv", 2.4486607656137274},
                {"VE2_Dzv", 0.4081101276022879},
                {"VE3_Dzv", 0.38471562505592244},
                {"VR1_Dzv", 14.710198029768193},
                {"VR2_Dzv", 2.4516996716280324},
                {"VR3_Dzv", 2.177715373011546},
                {"SpAbs_Dzse", 11.274293594541145},
                {"SpMax_Dzse", 5.682803211194224},
                {"SpDiam_Dzse", 8.218720170180022},
                {"SpAD_Dzse", 11.321047925578505},
                {"SpMAD_Dzse", 1.8868413209297508},
                {"LogEE_Dzse", 5.695272031686407},
                {"SM1_Dzse", 0.14026299311208512},
                {"VE1_Dzse", 2.44908291789815},
                {"VE2_Dzse", 0.40818048631635834},
                {"VE3_Dzse", 0.38488801149047164},
                {"VR1_Dzse", 14.702898588499734},
                {"VR2_Dzse", 2.450483098083289},
                {"VR3_Dzse", 2.177219033471847},
                {"SpAbs_Dzpe", 11.170276038514366},
                {"SpMax_Dzpe", 5.635900544792415},
                {"SpDiam_Dzpe", 8.154034362829579},
                {"SpAD_Dzpe", 11.224004108689806},
                {"SpMAD_Dzpe", 1.870667351448301},
                {"LogEE_Dzpe", 5.64899596424817},
                {"SM1_Dzpe", 0.16118421052631582},
                {"VE1_Dzpe", 2.448946507971185},
                {"VE2_Dzpe", 0.4081577513285308},
                {"VE3_Dzpe", 0.38483231156907877},
                {"VR1_Dzpe", 14.70489280529544},
                {"VR2_Dzpe", 2.45081546754924},
                {"VR3_Dzpe", 2.177354658536107},
                {"SpAbs_Dzare", 11.050096353686108},
                {"SpMax_Dzare", 5.581153204627513},
                {"SpDiam_Dzare", 8.079075457301386},
                {"SpAD_Dzare", 11.111985604500438},
                {"SpMAD_Dzare", 1.851997600750073},
                {"LogEE_Dzare", 5.595024922846496},
                {"SM1_Dzare", 0.18566775244299671},
                {"VE1_Dzare", 2.4487594922644638},
                {"VE2_Dzare", 0.4081265820440773},
                {"VE3_Dzare", 0.3847559428745089},
                {"VR1_Dzare", 14.707624957624027},
                {"VR2_Dzare", 2.451270826270671},
                {"VR3_Dzare", 2.1775404401382445},
                {"SpAbs_Dzp", 13.988334483721768},
                {"SpMax_Dzp", 6.7350763327699745},
                {"SpDiam_Dzp", 9.884555751088165},
                {"SpAD_Dzp", 13.642879938267223},
                {"SpMAD_Dzp", 2.273813323044537},
                {"LogEE_Dzp", 6.7389087466929665},
                {"SM1_Dzp", -0.5181818181818181},
                {"VE1_Dzp", 2.447440398429793},
                {"VE2_Dzp", 0.40790673307163217},
                {"VE3_Dzp", 0.384217119339509},
                {"VR1_Dzp", 14.72968247741355},
                {"VR2_Dzp", 2.454947079568925},
                {"VR3_Dzp", 2.1790390502924732},
                {"SpAbs_Dzi", 10.859504605114877},
                {"SpMax_Dzi", 5.492974979590288},
                {"SpDiam_Dzi", 7.959622282974812},
                {"SpAD_Dzi", 10.934587800267424},
                {"SpMAD_Dzi", 1.8224313000445707},
                {"LogEE_Dzi", 5.508207612108137},
                {"SM1_Dzi", 0.22524958545764784},
                {"VE1_Dzi", 2.4483919942977224},
                {"VE2_Dzi", 0.4080653323829537},
                {"VE3_Dzi", 0.3846058564544995},
                {"VR1_Dzi", 14.712989319034026},
                {"VR2_Dzi", 2.452164886505671},
                {"VR3_Dzi", 2.1779051069952313},
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
            EXPECT_FALSE(descriptors.Has("VR3_DzZ"));
        }
    }
}

TEST(MordredDescriptorTest, BaryszMatrixDescriptorsAreMissingWhenRequiredMatrixIsMissing) {
    const std::vector<std::string> descriptor_names{
        "SpAbs_DzZ",
        "SpMax_DzZ",
        "SpDiam_DzZ",
        "SpAD_DzZ",
        "SpMAD_DzZ",
        "LogEE_DzZ",
        "SM1_DzZ",
        "VE1_DzZ",
        "VE2_DzZ",
        "VE3_DzZ",
        "VR1_DzZ",
        "VR2_DzZ",
        "VR3_DzZ",
        "SpAbs_Dzm",
        "SpMax_Dzm",
        "SpDiam_Dzm",
        "SpAD_Dzm",
        "SpMAD_Dzm",
        "LogEE_Dzm",
        "SM1_Dzm",
        "VE1_Dzm",
        "VE2_Dzm",
        "VE3_Dzm",
        "VR1_Dzm",
        "VR2_Dzm",
        "VR3_Dzm",
        "SpAbs_Dzv",
        "SpMax_Dzv",
        "SpDiam_Dzv",
        "SpAD_Dzv",
        "SpMAD_Dzv",
        "LogEE_Dzv",
        "SM1_Dzv",
        "VE1_Dzv",
        "VE2_Dzv",
        "VE3_Dzv",
        "VR1_Dzv",
        "VR2_Dzv",
        "VR3_Dzv",
        "SpAbs_Dzse",
        "SpMax_Dzse",
        "SpDiam_Dzse",
        "SpAD_Dzse",
        "SpMAD_Dzse",
        "LogEE_Dzse",
        "SM1_Dzse",
        "VE1_Dzse",
        "VE2_Dzse",
        "VE3_Dzse",
        "VR1_Dzse",
        "VR2_Dzse",
        "VR3_Dzse",
        "SpAbs_Dzpe",
        "SpMax_Dzpe",
        "SpDiam_Dzpe",
        "SpAD_Dzpe",
        "SpMAD_Dzpe",
        "LogEE_Dzpe",
        "SM1_Dzpe",
        "VE1_Dzpe",
        "VE2_Dzpe",
        "VE3_Dzpe",
        "VR1_Dzpe",
        "VR2_Dzpe",
        "VR3_Dzpe",
        "SpAbs_Dzare",
        "SpMax_Dzare",
        "SpDiam_Dzare",
        "SpAD_Dzare",
        "SpMAD_Dzare",
        "LogEE_Dzare",
        "SM1_Dzare",
        "VE1_Dzare",
        "VE2_Dzare",
        "VE3_Dzare",
        "VR1_Dzare",
        "VR2_Dzare",
        "VR3_Dzare",
        "SpAbs_Dzp",
        "SpMax_Dzp",
        "SpDiam_Dzp",
        "SpAD_Dzp",
        "SpMAD_Dzp",
        "LogEE_Dzp",
        "SM1_Dzp",
        "VE1_Dzp",
        "VE2_Dzp",
        "VE3_Dzp",
        "VR1_Dzp",
        "VR2_Dzp",
        "VR3_Dzp",
        "SpAbs_Dzi",
        "SpMax_Dzi",
        "SpDiam_Dzi",
        "SpAD_Dzi",
        "SpMAD_Dzi",
        "LogEE_Dzi",
        "SM1_Dzi",
        "VE1_Dzi",
        "VE2_Dzi",
        "VE3_Dzi",
        "VR1_Dzi",
        "VR2_Dzi",
        "VR3_Dzi",
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

TEST(MordredDescriptorTest, CarbonMolecularDistanceEdgeDescriptorsMatchMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
        std::vector<std::string> missing_values;
    };

    const std::vector<Case> cases{
        {
            "C[N+](C)(C)CC(=O)[O-]",
            {
                {"MDEC-11", 1.5000000000000004},
                {"MDEC-12", 1.5000000000000004},
                {"MDEC-13", 1.0000000000000002},
                {"MDEC-23", 1.0},
            },
            {"MDEC-14", "MDEC-22", "MDEC-24", "MDEC-33", "MDEC-34", "MDEC-44"},
        },
        {
            "FC(F)(F)c1ccc(Br)cc1",
            {
                {"MDEC-22", 3.3019272488946267},
                {"MDEC-23", 5.656854249492381},
                {"MDEC-24", 1.6329931618554518},
                {"MDEC-33", 0.33333333333333337},
                {"MDEC-34", 0.9999999999999998},
            },
            {"MDEC-11", "MDEC-12", "MDEC-13", "MDEC-14", "MDEC-44"},
        },
        {
            "CC(C)(C)C(C)(C)C",
            {
                {"MDEC-11", 5.880395112623368},
                {"MDEC-14", 8.485281374238571},
                {"MDEC-44", 1.0},
            },
            {"MDEC-12", "MDEC-13", "MDEC-22", "MDEC-23", "MDEC-24", "MDEC-33", "MDEC-34"},
        },
        {
            "C",
            {},
            {
                "MDEC-11",
                "MDEC-12",
                "MDEC-13",
                "MDEC-14",
                "MDEC-22",
                "MDEC-23",
                "MDEC-24",
                "MDEC-33",
                "MDEC-34",
                "MDEC-44",
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
        for (const auto& name : expected.missing_values) {
            EXPECT_FALSE(descriptors.Has(name)) << name;
        }
    }
}

TEST(MordredDescriptorTest, OxygenAndNitrogenMolecularDistanceEdgeDescriptorsMatchMordredReferences) {
    struct Case {
        std::string smiles;
        std::map<std::string, double> expected_values;
        std::vector<std::string> missing_values;
    };

    const std::vector<Case> cases{
        {
            "COC(=O)c1ccc(OCC)c(O)c1C(=O)OCC",
            {
                {"MDEO-11", 0.646330407009565},
                {"MDEO-12", 2.345974797336829},
                {"MDEO-22", 0.5646216173286172},
            },
            {},
        },
        {
            "O=P(O)(O)O",
            {{"MDEO-11", 3.000000000000001}},
            {"MDEO-12", "MDEO-22"},
        },
        {
            "NNN",
            {
                {"MDEN-11", 0.4999999999999999},
                {"MDEN-12", 2.0},
            },
            {"MDEN-13", "MDEN-22", "MDEN-23", "MDEN-33"},
        },
        {
            "CN(C)N",
            {{"MDEN-13", 1.0}},
            {"MDEN-11", "MDEN-12", "MDEN-22", "MDEN-23", "MDEN-33"},
        },
        {
            "N1CCNCC1",
            {{"MDEN-22", 0.33333333333333337}},
            {"MDEN-11", "MDEN-12", "MDEN-13", "MDEN-23", "MDEN-33"},
        },
        {
            "CN(C)NC",
            {{"MDEN-23", 1.0}},
            {"MDEN-11", "MDEN-12", "MDEN-13", "MDEN-22", "MDEN-33"},
        },
        {
            "CN(C)N(C)C",
            {{"MDEN-33", 1.0}},
            {"MDEN-11", "MDEN-12", "MDEN-13", "MDEN-22", "MDEN-23"},
        },
        {
            "O.N",
            {},
            {
                "MDEO-11",
                "MDEO-12",
                "MDEO-22",
                "MDEN-11",
                "MDEN-12",
                "MDEN-13",
                "MDEN-22",
                "MDEN-23",
                "MDEN-33",
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
        for (const auto& name : expected.missing_values) {
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
