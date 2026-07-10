#include "oefp/kallisto_data.h"

#include <gtest/gtest.h>
#include <cmath>

// Spot-check test for kallisto parameter tables.
// Verifies a handful of known values from kallisto 1.0.10 to guard against
// generation drift.

namespace OEFP {
namespace {

TEST(KallistoData, BohrConstant) {
    // Verify BOHR_RADIUS_ANGSTROM matches kallisto.units.Bohr to 1e-11
    constexpr double expected_bohr = 0.5291772105437147;
    EXPECT_NEAR(kallisto::BOHR_RADIUS_ANGSTROM, expected_bohr, 1e-11);
}

TEST(KallistoData, CovalentRadius) {
    // covalent_radius[0] (H, Z=1 -> index 0 in Z-1 indexing)
    EXPECT_NEAR(kallisto::COVALENT_RADIUS[0], 0.80628308, 1e-8);
}

TEST(KallistoData, PaulingElectronegativity) {
    // pauling_en[0] (H, Z=1 -> index 0)
    EXPECT_NEAR(kallisto::PAULING_EN[0], 2.20, 1e-8);
}

TEST(KallistoData, EEQParameters) {
    // eeq_en[0] (H, Z=1 -> index 0)
    EXPECT_NEAR(kallisto::EEQ_EN[0], 1.23695041, 1e-8);
    // eeq_gamm[0] (H)
    EXPECT_NEAR(kallisto::EEQ_GAMM[0], -0.35015861, 1e-8);
    // eeq_cnfak[0] (H)
    EXPECT_NEAR(kallisto::EEQ_CNFAK[0], 0.04916110, 1e-8);
    // eeq_alp[0] (H)
    EXPECT_NEAR(kallisto::EEQ_ALP[0], 0.55159092, 1e-8);
}

TEST(KallistoData, ChemicalSymbols) {
    // chemical_symbols is Z-indexed (index 0 = "X", index 1 = "H", ...)
    EXPECT_STREQ(kallisto::CHEMICAL_SYMBOLS[0], "X");
    EXPECT_STREQ(kallisto::CHEMICAL_SYMBOLS[1], "H");
    EXPECT_STREQ(kallisto::CHEMICAL_SYMBOLS[6], "C");
}

TEST(KallistoData, VdWRadii) {
    // vdw_rahm and vdw_truhlar are Z-indexed (index 0 = sentinel, 1 = H, ...)
    // rahm['H'] = 0.91
    EXPECT_NEAR(kallisto::VDW_RAHM[1], 0.91, 1e-8);
    // truhlar['H'] = 0.65
    EXPECT_NEAR(kallisto::VDW_TRUHLAR[1], 0.65, 1e-8);
}

TEST(KallistoData, D4TableDimensions) {
    // Verify shapes by checking array bounds compile and spot-checking values
    // refn length 86 (Z-1 indexed)
    EXPECT_EQ(kallisto::D4_REFN[0], 2);  // H has 2 references

    // alphaiw dims 23 x 7 x 86 (just verify it compiles and check a known value)
    // alphaiw[0][0][0] is the first element (H, first reference, first CN bin)
    EXPECT_GT(kallisto::D4_ALPHAIW[0][0][0], 0.0);

    // sscale length 4 (NOT 86 as a stale brief annotation said)
    EXPECT_GT(kallisto::D4_SSCALE[0], 0.0);
    EXPECT_GT(kallisto::D4_SSCALE[3], 0.0);

    // seciw dims 23 x 8
    EXPECT_GT(kallisto::D4_SECIW[0][0], 0.0);
}

} // namespace
} // namespace OEFP
