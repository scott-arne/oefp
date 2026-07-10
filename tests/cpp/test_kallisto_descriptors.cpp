#include "oefp/kallisto_descriptors.h"
#include "oefp/kallisto_geometry.h"
#include "oefp/kallisto_data.h"

#include <gtest/gtest.h>
#include <oechem.h>
#include <cmath>
#include <vector>
#include <utility>

namespace OEFP {
namespace {

// Test coordination_numbers with cntype "erf" on H2
TEST(KallistoDescriptors, CoordinationNumbersErfH2) {
    // H2 molecule: H at (0,0,0), H at (1.4,0,0) Bohr
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);  // Hydrogen
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(h1, h2, 1);

    // Set coordinates in Angstrom (context converts to Bohr)
    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_h1[3] = {0.0, 0.0, 0.0};
    const double coords_h2[3] = {1.4 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto cn_erf = coordination_numbers(ctx, "erf");
    ASSERT_EQ(cn_erf.size(), 2u);
    EXPECT_NEAR(cn_erf[0], 0.91896554, 1e-6);
    EXPECT_NEAR(cn_erf[1], 0.91896554, 1e-6);
}

// Test coordination_numbers with cntype "cov" on H2
TEST(KallistoDescriptors, CoordinationNumbersCovH2) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(h1, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_h1[3] = {0.0, 0.0, 0.0};
    const double coords_h2[3] = {1.4 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto cn_cov = coordination_numbers(ctx, "cov");
    ASSERT_EQ(cn_cov.size(), 2u);
    EXPECT_NEAR(cn_cov[0], 0.90137656, 1e-6);
    EXPECT_NEAR(cn_cov[1], 0.90137656, 1e-6);
}

// Test coordination_numbers with cntype "exp" on H2
TEST(KallistoDescriptors, CoordinationNumbersExpH2) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(h1, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_h1[3] = {0.0, 0.0, 0.0};
    const double coords_h2[3] = {1.4 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto cn_exp = coordination_numbers(ctx, "exp");
    ASSERT_EQ(cn_exp.size(), 2u);
    EXPECT_NEAR(cn_exp[0], 0.91903651, 1e-6);
    EXPECT_NEAR(cn_exp[1], 0.91903651, 1e-6);
}

// Test proximity_shells on H2
TEST(KallistoDescriptors, ProximityShellsH2) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(h1, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_h1[3] = {0.0, 0.0, 0.0};
    const double coords_h2[3] = {1.4 * bohr_to_angstrom, 0.0, 0.0};
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto prox = proximity_shells(ctx, {2, 3});
    ASSERT_EQ(prox.size(), 2u);
    EXPECT_NEAR(prox[0], 9.53630841e-10, 1e-6);
    EXPECT_NEAR(prox[1], 9.53630841e-10, 1e-6);
}

// Test coordination_numbers on H2O triatomic
TEST(KallistoDescriptors, CoordinationNumbersH2O) {
    // O at (0,0,0), H at (1.8,0,0), H at (0,1.8,0) Bohr
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);   // Oxygen
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);  // Hydrogen
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    // Test all three CN types
    auto cn_erf = coordination_numbers(ctx, "erf");
    ASSERT_EQ(cn_erf.size(), 3u);
    EXPECT_NEAR(cn_erf[0], 1.99147549, 1e-6);
    EXPECT_NEAR(cn_erf[1], 0.99573774, 1e-6);
    EXPECT_NEAR(cn_erf[2], 0.99573774, 1e-6);

    auto cn_cov = coordination_numbers(ctx, "cov");
    ASSERT_EQ(cn_cov.size(), 3u);
    EXPECT_NEAR(cn_cov[0], 1.61210366, 1e-6);
    EXPECT_NEAR(cn_cov[1], 0.80605183, 1e-6);
    EXPECT_NEAR(cn_cov[2], 0.80605183, 1e-6);

    auto cn_exp = coordination_numbers(ctx, "exp");
    ASSERT_EQ(cn_exp.size(), 3u);
    EXPECT_NEAR(cn_exp[0], 1.98983559, 1e-6);
    EXPECT_NEAR(cn_exp[1], 0.99774852, 1e-6);
    EXPECT_NEAR(cn_exp[2], 0.99774852, 1e-6);
}

// Test proximity_shells on H2O
TEST(KallistoDescriptors, ProximityShellsH2O) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    auto prox = proximity_shells(ctx, {2, 3});
    ASSERT_EQ(prox.size(), 3u);
    EXPECT_NEAR(prox[0], 2.93567393e-11, 1e-6);
    EXPECT_NEAR(prox[1], 1.24699443e-02, 1e-6);
    EXPECT_NEAR(prox[2], 1.24699443e-02, 1e-6);
}

// Test that CovalentCoordinationNumbers() returns the same as coordination_numbers(..., "cov")
TEST(KallistoDescriptors, CovalentCoordinationNumbersCaching) {
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* o = mol.NewAtom(8);
    OEChem::OEAtomBase* h1 = mol.NewAtom(1);
    OEChem::OEAtomBase* h2 = mol.NewAtom(1);
    mol.NewBond(o, h1, 1);
    mol.NewBond(o, h2, 1);

    constexpr double bohr_to_angstrom = 0.5291772105437147;
    const double coords_o[3] = {0.0, 0.0, 0.0};
    const double coords_h1[3] = {1.8 * bohr_to_angstrom, 0.0, 0.0};
    const double coords_h2[3] = {0.0, 1.8 * bohr_to_angstrom, 0.0};
    mol.SetCoords(o, coords_o);
    mol.SetCoords(h1, coords_h1);
    mol.SetCoords(h2, coords_h2);
    mol.SetDimension(3);

    KallistoGeometryContext ctx(mol);
    ASSERT_TRUE(ctx.Eligible());

    // Get covalent CN via the kernel
    auto cn_cov_kernel = coordination_numbers(ctx, "cov");

    // Get covalent CN via the cached accessor
    const auto& cn_cov_cached = ctx.CovalentCoordinationNumbers();

    // They should match
    ASSERT_EQ(cn_cov_kernel.size(), cn_cov_cached.size());
    for (std::size_t i = 0; i < cn_cov_kernel.size(); ++i) {
        EXPECT_NEAR(cn_cov_kernel[i], cn_cov_cached[i], 1e-12);
    }

    // Call again to verify caching works
    const auto& cn_cov_cached2 = ctx.CovalentCoordinationNumbers();
    ASSERT_EQ(cn_cov_cached.size(), cn_cov_cached2.size());
    for (std::size_t i = 0; i < cn_cov_cached.size(); ++i) {
        EXPECT_EQ(cn_cov_cached[i], cn_cov_cached2[i]);
    }
}

// Test KallistoAtomDescriptorSchema
TEST(KallistoDescriptors, AtomDescriptorSchema) {
    auto schema = KallistoAtomDescriptorSchema();
    ASSERT_NE(schema, nullptr);

    // Should have 4 columns for this task
    ASSERT_EQ(schema->Size(), 4u);

    // Check each column
    EXPECT_EQ(schema->Definition(0).name, "cn_erf");
    EXPECT_EQ(schema->Definition(0).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(0).group, "kallisto");
    EXPECT_EQ(schema->Definition(0).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(0).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(1).name, "cn_cov");
    EXPECT_EQ(schema->Definition(1).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(1).group, "kallisto");
    EXPECT_EQ(schema->Definition(1).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(1).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(2).name, "cn_exp");
    EXPECT_EQ(schema->Definition(2).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(2).group, "kallisto");
    EXPECT_EQ(schema->Definition(2).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(2).prerequisites, kDescriptorPrerequisiteCoordinates3D);

    EXPECT_EQ(schema->Definition(3).name, "prox");
    EXPECT_EQ(schema->Definition(3).value_kind, DescriptorValueKind::Float);
    EXPECT_EQ(schema->Definition(3).group, "kallisto");
    EXPECT_EQ(schema->Definition(3).source_name, "kallisto");
    EXPECT_EQ(schema->Definition(3).prerequisites, kDescriptorPrerequisiteCoordinates3D);
}

// Test that kernels return empty vectors for ineligible contexts (2D molecule)
TEST(KallistoDescriptors, IneligibleContext2D) {
    // Create a 2D molecule (no 3D coords)
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* c1 = mol.NewAtom(6);  // Carbon
    OEChem::OEAtomBase* c2 = mol.NewAtom(6);
    mol.NewBond(c1, c2, 1);

    // Set 2D coords to ensure dimension is 2
    const double coords_c1[3] = {0.0, 0.0, 0.0};
    const double coords_c2[3] = {1.5, 0.0, 0.0};
    mol.SetCoords(c1, coords_c1);
    mol.SetCoords(c2, coords_c2);
    mol.SetDimension(2);

    ASSERT_EQ(mol.GetDimension(), 2);

    KallistoGeometryContext ctx(mol);
    EXPECT_FALSE(ctx.Eligible());
    EXPECT_EQ(ctx.AtomCount(), 2u);
    EXPECT_EQ(ctx.AtomicNumbers().size(), 0u);
    EXPECT_EQ(ctx.CoordsBohr().size(), 0u);

    // All kernels should return empty
    auto cn_erf = coordination_numbers(ctx, "erf");
    EXPECT_EQ(cn_erf.size(), 0u);

    auto cn_cov = coordination_numbers(ctx, "cov");
    EXPECT_EQ(cn_cov.size(), 0u);

    auto cn_exp = coordination_numbers(ctx, "exp");
    EXPECT_EQ(cn_exp.size(), 0u);

    auto prox = proximity_shells(ctx, {2, 3});
    EXPECT_EQ(prox.size(), 0u);

    // Cached covalent CN should also return empty
    const auto& cn_cov_cached = ctx.CovalentCoordinationNumbers();
    EXPECT_EQ(cn_cov_cached.size(), 0u);
}

// Test that kernels return empty vectors for ineligible contexts (Z > 86)
TEST(KallistoDescriptors, IneligibleContextHeavyElement) {
    // Create a molecule with Z > 86 (radon is 86, francium is 87)
    OEChem::OEGraphMol mol;
    OEChem::OEAtomBase* fr = mol.NewAtom(87);  // Francium, Z=87 > 86
    mol.SetDimension(3);

    constexpr double coords[3] = {0.0, 0.0, 0.0};
    mol.SetCoords(fr, coords);

    KallistoGeometryContext ctx(mol);
    EXPECT_FALSE(ctx.Eligible());
    EXPECT_EQ(ctx.AtomCount(), 1u);
    EXPECT_EQ(ctx.AtomicNumbers().size(), 0u);
    EXPECT_EQ(ctx.CoordsBohr().size(), 0u);

    // All kernels should return empty
    auto cn_erf = coordination_numbers(ctx, "erf");
    EXPECT_EQ(cn_erf.size(), 0u);

    auto cn_cov = coordination_numbers(ctx, "cov");
    EXPECT_EQ(cn_cov.size(), 0u);

    auto cn_exp = coordination_numbers(ctx, "exp");
    EXPECT_EQ(cn_exp.size(), 0u);

    auto prox = proximity_shells(ctx, {2, 3});
    EXPECT_EQ(prox.size(), 0u);

    // Cached covalent CN should also return empty
    const auto& cn_cov_cached = ctx.CovalentCoordinationNumbers();
    EXPECT_EQ(cn_cov_cached.size(), 0u);
}

}  // namespace
}  // namespace OEFP
