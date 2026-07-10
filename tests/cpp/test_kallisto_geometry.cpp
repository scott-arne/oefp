#include "oefp/kallisto_geometry.h"
#include "oefp/kallisto_data.h"

#include <gtest/gtest.h>
#include <oechem.h>
#include <cmath>
#include <vector>
#include <array>
#include <optional>

namespace OEFP {
namespace {

// Test solve_dense_linear with a known 3x3 system
TEST(SolveDenseLinear, KnownSystem3x3) {
    // System: 2x + y + z = 5
    //         x + 2y + z = 5
    //         x + y + 2z = 5
    // Solution: x = y = z = 5/4 = 1.25
    std::vector<double> a = {
        2.0, 1.0, 1.0,
        1.0, 2.0, 1.0,
        1.0, 1.0, 2.0
    };
    std::vector<double> b = {5.0, 5.0, 5.0};
    std::size_t n = 3;

    auto result = solve_dense_linear(a, b, n);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3u);
    EXPECT_NEAR((*result)[0], 1.25, 1e-12);
    EXPECT_NEAR((*result)[1], 1.25, 1e-12);
    EXPECT_NEAR((*result)[2], 1.25, 1e-12);
}

// Test solve_dense_linear with a singular matrix
TEST(SolveDenseLinear, SingularMatrix) {
    // Singular matrix (row 2 = row 1)
    std::vector<double> a = {
        1.0, 2.0, 3.0,
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    std::vector<double> b = {1.0, 1.0, 1.0};
    std::size_t n = 3;

    auto result = solve_dense_linear(a, b, n);
    EXPECT_FALSE(result.has_value());
}

// Test solve_dense_linear with invalid dimensions
TEST(SolveDenseLinear, InvalidDimensions) {
    std::vector<double> a = {1.0, 2.0, 3.0};  // size 3, not 9
    std::vector<double> b = {1.0, 2.0, 3.0};
    std::size_t n = 3;

    EXPECT_THROW(solve_dense_linear(a, b, n), std::invalid_argument);

    std::vector<double> a2 = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    std::vector<double> b2 = {1.0, 2.0};  // size 2, not 3
    EXPECT_THROW(solve_dense_linear(a2, b2, n), std::invalid_argument);
}

// Test eligibility: 2D molecule should be ineligible
TEST(KallistoGeometryContext, Ineligible2D) {
    OEChem::OEGraphMol mol;
    OEChem::OESmilesToMol(mol, "CC");

    // Set 2D coordinates
    unsigned int i = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom, ++i) {
        const double coords[3] = {static_cast<double>(i) * 1.5, 0.0, 0.0};
        mol.SetCoords(atom, coords);
    }
    mol.SetDimension(2);

    KallistoGeometryContext ctx(mol);
    EXPECT_FALSE(ctx.Eligible());
}

// Test eligibility: molecule with Z > 86 should be ineligible
TEST(KallistoGeometryContext, IneligibleUnsupportedElement) {
    OEChem::OEGraphMol mol;
    // Create a molecule with uranium (Z=92)
    OEChem::OESmilesToMol(mol, "[U]");

    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const double coords[3] = {0.0, 0.0, 0.0};
        mol.SetCoords(atom, coords);
    }
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    EXPECT_FALSE(ctx.Eligible());
}

// Test eligibility: valid 3D molecule with supported elements
TEST(KallistoGeometryContext, EligibleValid3D) {
    OEChem::OEGraphMol mol;
    OEChem::OESmilesToMol(mol, "CC");

    // Set 3D coordinates in Angstrom
    unsigned int i = 0;
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom, ++i) {
        const double coords[3] = {static_cast<double>(i) * 1.5, 0.0, 0.0};
        mol.SetCoords(atom, coords);
    }
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());
    EXPECT_EQ(ctx.AtomCount(), 2u);

    // Check atomic numbers (C = 6)
    const auto& atomic_nums = ctx.AtomicNumbers();
    ASSERT_EQ(atomic_nums.size(), 2u);
    EXPECT_EQ(atomic_nums[0], 6);
    EXPECT_EQ(atomic_nums[1], 6);

    // Check coordinates are converted to Bohr
    const auto& coords_bohr = ctx.CoordsBohr();
    ASSERT_EQ(coords_bohr.size(), 2u);
    EXPECT_NEAR(coords_bohr[0][0], 0.0, 1e-12);
    EXPECT_NEAR(coords_bohr[0][1], 0.0, 1e-12);
    EXPECT_NEAR(coords_bohr[0][2], 0.0, 1e-12);
    EXPECT_NEAR(coords_bohr[1][0], 1.5 / kallisto::BOHR_RADIUS_ANGSTROM, 1e-12);
    EXPECT_NEAR(coords_bohr[1][1], 0.0, 1e-12);
    EXPECT_NEAR(coords_bohr[1][2], 0.0, 1e-12);
}

// Test charge: defaults to summed formal charge
TEST(KallistoGeometryContext, ChargeDefaultsSummedFormal) {
    OEChem::OEGraphMol mol;
    OEChem::OESmilesToMol(mol, "[NH4+]");

    // Set 3D coordinates
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const double coords[3] = {0.0, 0.0, 0.0};
        mol.SetCoords(atom, coords);
    }
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());
    EXPECT_EQ(ctx.Charge(), 1);  // NH4+ has formal charge +1
}

// Test charge: honors override
TEST(KallistoGeometryContext, ChargeHonorsOverride) {
    OEChem::OEGraphMol mol;
    OEChem::OESmilesToMol(mol, "CC");

    // Set 3D coordinates
    for (OESystem::OEIter<OEChem::OEAtomBase> atom = mol.GetAtoms(); atom; ++atom) {
        const double coords[3] = {0.0, 0.0, 0.0};
        mol.SetCoords(atom, coords);
    }
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol, 2);  // Override charge to +2
    ASSERT_TRUE(ctx.Eligible());
    EXPECT_EQ(ctx.Charge(), 2);
}

} // namespace
} // namespace OEFP
